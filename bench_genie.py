# -*- coding: utf-8 -*-
# Benchmark Genie /tts streaming: report chunk latency and RTFx.
import json
import sys
import time

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9882"
TEXT = "你好，我是基于 ONNX 的轻量后端，现在 testing streaming output，延迟应该比大运行时低不少。"

payload = {"character_name": "dq", "text": TEXT}
t0 = time.perf_counter()
pieces = []
with requests.post(BASE + "/tts", json=payload, stream=True, timeout=300) as r:
    r.raise_for_status()
    for piece in r.iter_content(chunk_size=8192):
        now = time.perf_counter() - t0
        dur = len(piece) / 2 / 32000
        pieces.append(piece)
        if dur > 0.01:
            print(f"  chunk #{len(pieces)}: t={now:6.2f}s  audio={dur:5.2f}s")
wall = time.perf_counter() - t0
total = sum(len(p) for p in pieces) / 2 / 32000
print(f"wall {wall:.2f}s | audio {total:.2f}s | RTFx {total / wall:.2f}x")
