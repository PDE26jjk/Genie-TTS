# -*- coding: utf-8 -*-
# Standalone ZCode speech watcher: tails ZCode's model-io rollout files and
# pushes the agent's reply text into the TTS relay sentence by sentence.
#
# Scope: FILE ACCESS ONLY. ZCode appends one JSON line per completed model
# call to ~/.zcode/cli/rollout/model-io-sess_<id>.jsonl (response.text =
# that call's full text, querySource=="main_turn" marks the main agent;
# subagents and background compaction are skipped). Nothing is written
# during token streaming, so latency is one model call - narration blocks
# between tool calls land milliseconds after the call finishes.
#
# The on/off switch is the "read_zcode" key of the relay config
# (tts_relay_config.json) - the tray menu item toggles it, no restart
# needed. Disabled mode still advances the file offsets, so re-enabling
# never replays old content.
#
# Run:  python zcode_speech_watcher.py [--relay-url http://127.0.0.1:9881]
import argparse
import codecs
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
    unfinished tail over to the next model call."""

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


class RolloutSource:
    """Tails ZCode's model-io rollout files (one JSON line per completed
    model call). Per-file offsets so two ZCode windows both work."""

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
    ap.add_argument("--relay-url", default=DEFAULT_RELAY)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--rollout-dir", default=DEFAULT_ROLLOUT_DIR)
    ap.add_argument("--interval", type=float, default=0.25)
    args = ap.parse_args()

    speaker = Speaker(args.relay_url)
    src = RolloutSource(args.rollout_dir)
    print(f"[watcher] relay={args.relay_url} rollout_dir={args.rollout_dir}", flush=True)

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
                src.pump(speaker)
        except Exception as e:
            print("[watcher] error:", e, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
