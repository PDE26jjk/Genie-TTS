# -*- coding: utf-8 -*-
# Standalone ZCode speech watcher: tails ZCode's model-io rollout files and
# pushes the agent's reply text into the TTS relay sentence by sentence, so
# speech follows the reply as it is generated instead of waiting for the
# whole turn (the old Stop-hook approach).
#
# Why file watching: ZCode persists nothing during token streaming; every
# completed model call is appended as one JSON line to
# ~/.zcode/cli/rollout/model-io-sess_<id>.jsonl, milliseconds after its text
# finished generating. Per-model-call is the finest granularity available
# from disk, and for tool-heavy turns it means every narration block between
# tool calls is spoken almost immediately.  Only querySource=="main_turn"
# lines are used; subagents and background compaction are skipped.
#
# The on/off switch is the "read_zcode" key of the relay config
# (tts_relay_config.json) - the tray menu item toggles it, no restart needed.
# Disabled mode still advances the file offsets, so re-enabling never
# replays old content.
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
                print("[watcher] say:", sent[:60], flush=True)
                self.say(sent)
        if len(self.buf) > 120:  # no punctuation at all: cut anyway
            sent, self.buf = self.buf.strip(), ""
            if len(sent) >= 2:
                print("[watcher] say:", sent[:60], flush=True)
                self.say(sent)

    def flush(self):
        tail, self.buf = self.buf.strip(), ""
        if len(tail) >= 4:
            print("[watcher] say(tail):", tail[:60], flush=True)
            self.say(tail)


def init_offset(path):
    """Start at EOF, but back up to the last complete line so a line that is
    mid-write at startup is not lost."""
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


def load_enabled(config_path):
    try:
        with open(config_path, encoding="utf-8") as f:
            return bool(json.load(f).get("read_zcode"))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Speak ZCode replies via the TTS relay")
    ap.add_argument("--relay-url", default=DEFAULT_RELAY)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--rollout-dir", default=DEFAULT_ROLLOUT_DIR)
    ap.add_argument("--interval", type=float, default=0.25)
    args = ap.parse_args()

    speaker = Speaker(args.relay_url)
    dec = codecs.getincrementaldecoder("utf-8")()
    state = {}  # path -> [offset, tail]
    cfg_mtime = None
    enabled = False

    print(f"[watcher] relay={args.relay_url} rollout_dir={args.rollout_dir}", flush=True)
    while True:
        try:
            if os.path.exists(args.config):
                m = os.path.getmtime(args.config)
                if m != cfg_mtime:
                    cfg_mtime = m
                    enabled = load_enabled(args.config)
            paths = set(glob.glob(os.path.join(args.rollout_dir, "model-io-sess_*.jsonl")))
            for p in list(state):
                if p not in paths:
                    state.pop(p, None)
            for p in sorted(paths):
                if p not in state:
                    state[p] = [init_offset(p), ""]
                    continue
                off, tail = state[p]
                size = os.path.getsize(p)
                if size < off:  # truncated/rotated
                    state[p] = [size, ""]
                    continue
                if size == off:
                    continue
                with open(p, "rb") as f:
                    f.seek(off)
                    chunk = f.read()
                state[p][0] = off + len(chunk)
                data = dec.decode(chunk)
                lines = (tail + data).split("\n")
                tail = lines.pop()
                if not enabled:
                    state[p][1] = ""
                    continue
                state[p][1] = tail
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("querySource") != "main_turn":
                        continue
                    text = clean_for_speech((rec.get("response") or {}).get("text") or "")
                    if text:
                        speaker.feed(text)
                speaker.flush()
        except Exception as e:
            print("[watcher] error:", e, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
