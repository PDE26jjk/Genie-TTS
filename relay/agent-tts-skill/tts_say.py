# -*- coding: utf-8 -*-
# agent-tts helper: push text to the local TTS relay (tts_relay_tray.py).
# Standard library only, so it runs under any Python installed on this
# machine (system python, Genie venv, GPT-SoVITS runtime, ...).
#
# Usage:
#   python tts_say.py "要朗读的文本"          # enqueue and play
#   echo 文本 | python tts_say.py --stdin     # read text from stdin
#   python tts_say.py --status                # relay status JSON
#   python tts_say.py --stop                  # abort current speech + clear queue
import argparse
import json
import sys
import urllib.request

DEFAULT_URL = "http://127.0.0.1:9881"


def post(url, payload, timeout=15):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser(description="Speak text through the local TTS relay")
    ap.add_argument("text", nargs="*", help="text to speak")
    ap.add_argument("--url", default=DEFAULT_URL, help="relay base URL")
    ap.add_argument("--stop", action="store_true",
                    help="abort the current utterance and clear the queue")
    ap.add_argument("--status", action="store_true", help="print relay status JSON")
    ap.add_argument("--stdin", action="store_true", help="read text from stdin")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    try:
        if args.stop:
            print(post(url + "/stop", {}))
        elif args.status:
            print(get(url + "/status"))
        else:
            text = " ".join(args.text).strip()
            if args.stdin:
                text = sys.stdin.read().strip()
            if not text:
                print("nothing to say", file=sys.stderr)
                sys.exit(2)
            print(post(url + "/say", {"text": text}))
    except Exception as e:
        print(f"relay unreachable at {url}: {e}", file=sys.stderr)
        print("start the relay with: pythonw tts_relay_tray.py "
              "(Genie-TTS repo, relay/ directory, run with its .venv python)",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
