# -*- coding: utf-8 -*-
# Standalone ZCode speech watcher: speaks the agent's reply through the TTS
# relay sentence by sentence WHILE it is being generated, instead of waiting
# for the whole turn (the old Stop-hook approach).
#
# Two sources, pick with --mode:
#
#   uia (default)  Reads ZCode's window text live via UI Automation. Chromium
#                  builds its accessibility tree once a screen-reader client
#                  sends WM_GETOBJECT; the tree then tracks every streamed
#                  token. Latency ~ the polling interval. The raw text mixes
#                  reply paragraphs with thought headers, tool blocks and the
#                  status bar, so a small line-level state machine filters
#                  everything that is not spoken prose.
#
#   rollout        Tails ~/.zcode/cli/rollout/model-io-sess_*.jsonl - one
#                  JSON line per COMPLETED model call (querySource=main_turn).
#                  ZCode persists nothing during token streaming, so this is
#                  coarser: a narration block only lands when the whole model
#                  call (incl. tool-call parameters) finished. Kept as a
#                  fallback for when the UIA path cannot see the window
#                  (hidden windows, other sessions).
#
# The on/off switch is the "read_zcode" key of the relay config
# (tts_relay_config.json) - the tray menu item toggles it, no restart needed.
#
# Run:  python zcode_speech_watcher.py [--mode uia] [--relay-url ...]
import argparse
import codecs
import ctypes
import glob
import json
import os
import re
import time

import requests

DEFAULT_RELAY = "http://127.0.0.1:9881"
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tts_relay_config.json")
DEFAULT_ROLLOUT_DIR = os.path.join(os.path.expanduser("~"), ".zcode", "cli", "rollout")

SENTENCE_RE = re.compile(r"[。！？；!?…\n]")

# localhost only; never route through a system proxy (Fiddler etc.)
HTTP = requests.Session()
HTTP.trust_env = False


def clean_for_speech(text):
    """Strip constructs that are unreadable as speech."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)   # fenced code blocks
    text = re.sub(r"`[^`]*`", " ", text)                  # inline code
    text = re.sub(r"https?://\S+", " ", text)             # urls
    text = re.sub(r"^\|.*$", " ", text, flags=re.M)       # table rows
    text = re.sub(r"\|", " ", text)                       # leftover pipes
    text = re.sub(r"[→←↔↑↓⇒⇔]", " 到 ", text)             # arrows: speak as words
    text = re.sub(r"[≈≠≤≥±×÷·°]", " ", text)              # math signs: drop
    text = re.sub(r"[*_~#>\[\]()]|::\w+\{[^}]*\}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class Speaker:
    """Sentence splitter + relay client: speaks whole sentences, carries an
    unfinished tail over to the next chunk."""

    def __init__(self, relay_url):
        self.relay_url = relay_url.rstrip("/")
        self.buf = ""

    def say(self, text):
        try:
            HTTP.post(self.relay_url + "/say", json={"text": text}, timeout=5)
        except Exception as e:
            print("[watcher] relay unreachable:", e, flush=True)

    def feed(self, text):
        self.buf += text
        while True:
            m = SENTENCE_RE.search(self.buf)
            if not m:
                break
            sent = self.buf[: m.end()].strip()
            self.buf = self.buf[m.end():]
            if len(sent) >= 2:
                print("[watcher] %s say: %s" % (time.strftime("%H:%M:%S"), sent[:60]), flush=True)
                self.say(sent)
        if len(self.buf) > 120:  # no punctuation at all: cut anyway
            sent, self.buf = self.buf.strip(), ""
            if len(sent) >= 2:
                print("[watcher] %s say: %s" % (time.strftime("%H:%M:%S"), sent[:60]), flush=True)
                self.say(sent)

    def flush(self):
        tail, self.buf = self.buf.strip(), ""
        if len(tail) >= 4:
            print("[watcher] %s say(tail): %s" % (time.strftime("%H:%M:%S"), tail[:60]), flush=True)
            self.say(tail)


def load_enabled(config_path):
    try:
        with open(config_path, encoding="utf-8") as f:
            return bool(json.load(f).get("read_zcode"))
    except Exception:
        return False


class UiSource:
    """ZCode UI Automation reader. Chromium exposes its accessibility tree to
    UIA clients once WM_GETOBJECT(OBJID_CLIENT) hits its windows; the tree
    then tracks streamed tokens in real time.

    Strategy: re-parse the WHOLE chat text every poll and speak only lines
    that are prose.  Thought blocks (header line 思考 ... until 持续了 X 秒),
    tool blocks (header line 终端/查阅/正在执行/... until the result card),
    and the status area are dropped wholesale.  Spoken lines are compared
    with the previous poll line-by-line, so UI noise or streamed rewrites
    can never replay content, and the last line is always withheld one poll
    (only lines followed by another line are considered final).  A cap on
    new lines per poll is the flood guard."""

    # exact lines that open a non-speech block
    SKIP_HEADERS = {"终端", "查阅", "正在执行", "思考", "搜索", "读取", "编辑",
                    "写入", "任务", "子智能体", "工作流", "命令面板", "网页",
                    "图片", "文件"}
    # exact lines of the bottom status area - always dropped
    DROP_LINES = {"添加上下文", "切换模式", "电脑操作", "后退", "前进",
                  "停止生成", "最高", "显示更多", "加载中...", "撤销", "￼",
                  "复制", "GLM-5.3-Flash"}
    THINK_END_RE = re.compile(r"^(持续了 \d+ 秒|已工作 \d+ 分 \d+ 秒|已工作 \d+ 秒|工作中 \d+ 分 \d+ 秒)$")
    CARD_RE = re.compile(r"^(\d+ 个文件已更改|\+\d+ -\d+|\d+ 文件)$")
    DROP_PREFIX = ("上下文已用", "打开运行中的后台任务", "正在等待")
    # status-area anchors: everything after the LAST of these in the document
    # is the input box / status bar (the input box text must not be spoken)
    BAR_MARKERS = ("添加上下文", "切换模式", "电脑操作", "停止生成", "后退",
                   "前进", "上下文已用")

    MAX_NEW_LINES = 20  # flood guard: more than this per poll = discard

    def __init__(self):
        import uiautomation as uia
        self.uia = uia
        uia.SetGlobalSearchTimeout(3)
        self.user32 = ctypes.windll.user32
        self.WM_GETOBJECT = 0x003D
        self.OBJID_CLIENT = 0xFFFFFFFC
        self.prev_lines = []  # spoken-prose lines already consumed
        self._no_read = 0
        self._enum_cb = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)

    def _activate(self):
        win = self.uia.WindowControl(searchDepth=1, ClassName="Chrome_WidgetWin_1",
                                     Name="ZCode")
        if not win.Exists(2):
            return None
        top = win.NativeWindowHandle
        hit = [0]

        def cb(hwnd, _):
            buf = ctypes.create_unicode_buffer(64)
            self.user32.GetClassNameW(hwnd, buf, 64)
            if "Chrome" in buf.value:
                if self.user32.SendMessageW(hwnd, self.WM_GETOBJECT, 0, self.OBJID_CLIENT):
                    hit[0] += 1
            return 1

        self.user32.EnumChildWindows(top, self._enum_cb(cb), None)
        if not hit[0]:
            return None
        return win

    def read_text(self):
        """Return the document text (or None if the window is not readable)."""
        win = self._activate()
        if win is None:
            return None
        docs = []

        def collect(c):
            if c.ControlTypeName == "DocumentControl":
                docs.append(c)
                return
            try:
                for k in c.GetChildren():
                    collect(k)
            except Exception:
                pass

        collect(win)
        for d in reversed(docs):
            try:
                tp = d.GetPattern(self.uia.PatternId.TextPattern)
                if tp:
                    return tp.DocumentRange.GetText(300000)
            except Exception:
                pass
        return None

    def parse_lines(self, chat):
        """Whole-text line filter -> only speakable prose lines, in order."""
        # cut everything after the last status-area anchor: input box text
        # arrives here too while the user types, and must never be spoken
        cut = max((chat.rfind(m) for m in self.BAR_MARKERS), default=-1)
        if cut >= 0:
            chat = chat[:cut]
        # the input box sits directly ABOVE the status block: walk back from
        # 添加上下文 over the free-text lines until a fixed UI line shows up
        p = chat.rfind("添加上下文")
        if p >= 0:
            lines = chat[:p].splitlines()
            while lines:
                t = lines[-1].strip()
                if not t or t in self.DROP_LINES or t.startswith(self.DROP_PREFIX):
                    break
                lines.pop()
            chat = "\n".join(lines)
        raw = [l.strip() for l in chat.splitlines()]
        # user messages are followed by their hover button line "复制"
        user_msg = {i for i, s in enumerate(raw) if s and i + 1 < len(raw)
                    and raw[i + 1] == "复制"}
        out = []
        state = "normal"
        for i, s in enumerate(raw):
            if not s or s in self.DROP_LINES or s.startswith(self.DROP_PREFIX):
                continue
            if i in user_msg:
                continue  # the user's own words are not spoken
            if self.THINK_END_RE.match(s):
                # thinking-duration line, wherever it appears: never prose
                continue
            if s in self.SKIP_HEADERS:
                state = "skip"
                continue
            if state == "skip":
                if self.THINK_END_RE.match(s) or self.CARD_RE.match(s):
                    state = "normal"
                continue
            if self.CARD_RE.match(s):
                continue
            out.append(s)
        return out

    def pump(self, speaker):
        """One poll: read, re-parse everything, feed only genuinely new lines."""
        full = self.read_text()
        if not full:
            self._no_read += 1
            if self._no_read % 25 == 1:
                print("[watcher] ui read failed x%d" % self._no_read, flush=True)
            return False
        self._no_read = 0
        lines = self.parse_lines(full)
        # withhold the last line: it may still be streaming
        final = lines[:-1] if lines else []
        prev = self.prev_lines
        # common prefix with what we already consumed
        k = 0
        for a, b in zip(prev, final):
            if a != b:
                break
            k += 1
        if k < len(prev):
            # virtual list may have dropped old messages from the top:
            # try to re-align prev as a tail of final instead of replaying
            for j in range(1, len(prev) + 1):
                tail = prev[j:]
                if tail and final[: len(tail)] == tail:
                    k = len(tail)
                    break
        new = final[k:]
        self.prev_lines = final
        if not new:
            return True
        if len(new) > self.MAX_NEW_LINES:
            print(f"[watcher] flood guard: {len(new)} new lines dropped", flush=True)
            return True
        for line in new:
            text = clean_for_speech(line)
            if text:
                speaker.feed(text + "\n")
        return True


class RolloutSource:
    """Tails ZCode's model-io rollout files (one JSON line per completed
    model call).  Coarser than UIA but works without reading the window."""

    def __init__(self, rollout_dir):
        self.rollout_dir = rollout_dir
        self.dec = codecs.getincrementaldecoder("utf-8")()
        self.state = {}  # path -> [offset, tail]

    @staticmethod
    def _init_offset(path):
        size = os.path.getsize(path)
        if not size:
            return 0
        with open(path, "rb") as f:
            f.seek(max(0, size - 1))
            if f.read(1) == b"\n":
                return size
            f.seek(max(0, size - 2_000_000))
            data = f.read()
        idx = data.rfind(b"\n")
        return size - len(data) + idx + 1 if idx >= 0 else 0

    def pump(self, speaker):
        try:
            paths = set(glob.glob(os.path.join(self.rollout_dir, "model-io-sess_*.jsonl")))
            for p in list(self.state):
                if p not in paths:
                    self.state.pop(p, None)
            for p in sorted(paths):
                if p not in self.state:
                    self.state[p] = [self._init_offset(p), ""]
                    continue
                off, tail = self.state[p]
                size = os.path.getsize(p)
                if size < off:  # truncated/rotated
                    self.state[p] = [size, ""]
                    continue
                if size == off:
                    continue
                with open(p, "rb") as f:
                    f.seek(off)
                    chunk = f.read()
                self.state[p][0] = off + len(chunk)
                data = self.dec.decode(chunk)
                lines = (tail + data).split("\n")
                tail = lines.pop()
                self.state[p][1] = tail
                for line in lines:
                    self._speak_line(line, speaker)
                speaker.flush()
        except Exception as e:
            print("[watcher] rollout error:", e, flush=True)

    def _speak_line(self, line, speaker):
        line = line.strip()
        if not line:
            return
        try:
            rec = json.loads(line)
        except Exception:
            return
        if rec.get("querySource") != "main_turn":
            return
        text = clean_for_speech((rec.get("response") or {}).get("text") or "")
        if text:
            speaker.feed(text)
            speaker.flush()


def main():
    ap = argparse.ArgumentParser(description="Speak ZCode replies via the TTS relay")
    ap.add_argument("--mode", choices=["uia", "rollout"], default="uia",
                    help="uia: live window text (near real-time); "
                         "rollout: completed model calls (fallback)")
    ap.add_argument("--relay-url", default=DEFAULT_RELAY)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--rollout-dir", default=DEFAULT_ROLLOUT_DIR)
    ap.add_argument("--interval", type=float, default=0.4)
    args = ap.parse_args()

    speaker = Speaker(args.relay_url)
    try:
        src = UiSource() if args.mode == "uia" else RolloutSource(args.rollout_dir)
        print(f"[watcher] mode={args.mode} relay={args.relay_url}", flush=True)
    except Exception as e:
        print(f"[watcher] {args.mode} source unavailable ({e}), nothing to do", flush=True)
        return

    cfg_mtime = None
    enabled = False
    while True:
        try:
            if os.path.exists(args.config):
                m = os.path.getmtime(args.config)
                if m != cfg_mtime:
                    cfg_mtime = m
                    enabled = load_enabled(args.config)
            if enabled:
                ok = src.pump(speaker)
                if args.mode == "rollout":
                    pass
                elif ok is False:
                    time.sleep(1.0)  # window gone / not readable
        except Exception as e:
            print("[watcher] error:", e, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
