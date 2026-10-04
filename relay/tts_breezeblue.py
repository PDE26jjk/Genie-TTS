# -*- coding: utf-8 -*-
# Standalone breeze.blue TTS test client (cloud provider probe).
#
# Usage:
#   python tts_breezeblue.py balance
#   python tts_breezeblue.py tts --text "你好世界" [--out out.wav] [--voice voc_xxx]
#                                [--model breeze-tts-2] [--lang zh]
#                                [--speed 1.0] [--no-stream]
#
# API key resolution: --key > BREEZE_API_KEY env var. Never hardcode keys.
#
# Streaming wire format: POST /v1/text-to-speech/{voice_id}/stream returns
# raw s16le mono PCM at 24 kHz when output_format=pcm (headerless chunks,
# same shape as the relay tray's local backends).
#
# Billing (pricing docs): CJK chars (Han / kana / Hangul) bill 2 units,
# everything else 1 unit; 10 units = 1 credit. This script estimates the
# cost locally and (by default) confirms it via a free balance read.
import argparse
import os
import sys
import time
import wave

import requests

API_BASE = "https://api.breeze.blue"
DEFAULT_VOICE = "voc_kwje43kewnst"
PCM_SR = 24000


def billable_units(text: str) -> int:
    # breeze.blue pricing: CJK chars bill 2 units, everything else 1;
    # 10 units = 1 credit
    total = 0
    for ch in text:
        o = ord(ch)
        if (0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF
                or 0xAC00 <= o <= 0xD7AF):
            total += 2
        else:
            total += 1
    return total


def get_balance(key: str) -> dict:
    r = requests.get(API_BASE + "/v1/balance",
                     headers={"xi-api-key": key}, timeout=15)
    r.raise_for_status()
    return r.json()


def cmd_balance(args) -> int:
    data = get_balance(args.key)
    print(f"balance: {data.get('balance')} credits "
          f"({data.get('balance_millicredits')} millicredits)")
    for ch in data.get("charges") or []:
        print("  charge:", ch)
    for tp in data.get("topups") or []:
        print(f"  topup: +{tp.get('amount')} ({tp.get('description')}, "
              f"{tp.get('created_at')})")
    return 0


def cmd_tts(args) -> int:
    before = get_balance(args.key).get("balance")
    units = billable_units(args.text)
    print(f"local estimate: {units} units = {units / 10:.1f} credits "
          f"(balance before: {before})")

    url = f"{API_BASE}/v1/text-to-speech/{args.voice}/stream"
    headers = {"xi-api-key": args.key, "Accept": "audio/*"}
    payload = {
        "text": args.text,
        "model_id": args.model,
        "voice_settings": {"speed": args.speed},
    }
    if args.lang:
        payload["language_code"] = args.lang

    t0 = time.perf_counter()
    first_byte = None
    pcm = bytearray()
    with requests.post(url, headers=headers, json=payload,
                       stream=True, timeout=(15, 120)) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "?")
        for piece in r.iter_content(chunk_size=4096):
            if first_byte is None:
                first_byte = time.perf_counter()
                print(f"first audio byte: {first_byte - t0:.2f}s "
                      f"(Content-Type: {ctype})")
            pcm.extend(piece)
    total = time.perf_counter() - t0
    if not pcm:
        print("no audio returned", file=sys.stderr)
        return 1
    dur = len(pcm) / 2 / PCM_SR
    print(f"audio: {len(pcm)} bytes s16le mono {PCM_SR}Hz = {dur:.2f}s, "
          f"total {total:.2f}s, TTFA {first_byte - t0:.2f}s, "
          f"RTF {total / dur:.2f}")

    if args.out:
        with wave.open(args.out, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(PCM_SR)
            wf.writeframes(bytes(pcm))
        print(f"saved: {args.out}")

    after = get_balance(args.key).get("balance")
    spent = (before or 0) - (after or 0)
    print(f"balance after: {after} (spent {spent:.1f} credits, "
          f"actual cost {spent / max(units / 10, 1e-9) * 100:.0f}% of estimate)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="breeze.blue TTS test client")
    p.add_argument("--key", default=os.environ.get("BREEZE_API_KEY", ""),
                   help="API key (default: BREEZE_API_KEY env)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance", help="show credit balance")
    t = sub.add_parser("tts", help="synthesize text (streaming by default)")
    t.add_argument("--text", required=True)
    t.add_argument("--out", default="", help="save to .wav file")
    t.add_argument("--voice", default=DEFAULT_VOICE)
    t.add_argument("--model", default="breeze-tts-2")
    t.add_argument("--lang", default="zh", help="language_code (empty = auto)")
    t.add_argument("--speed", type=float, default=1.0)
    args = p.parse_args()
    if not args.key:
        print("no API key: pass --key or set BREEZE_API_KEY", file=sys.stderr)
        return 2
    return cmd_balance(args) if args.cmd == "balance" else cmd_tts(args)


if __name__ == "__main__":
    sys.exit(main())
