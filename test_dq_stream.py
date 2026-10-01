# -*- coding: utf-8 -*-
# End-to-end test: load the converted v2Pro ONNX model, stream a mixed
# zh/en sentence, report per-chunk latency, play the result.
import asyncio
import time

import numpy as np
import sounddevice as sd

import genie_tts as genie

MODEL_DIR = r"R:\git\dsh_projects\cyber_persona\Genie-TTS\models_dq_v2pro"
REF_WAV = r"R:\git\dsh_projects\cyber_persona\GPT-SoVITS-v2pro-20250604\output\slicer_opt\dq1.wav.reformatted_vocals.flac_0001392000_0001570560.wav"
REF_TEXT = "期待明年，明年一定做，今年我也有在想。"
TEXT = "你好，我是基于 ONNX 的轻量后端，现在 testing streaming output，延迟应该比大运行时低不少。"

print("loading character (ONNX sessions) ...")
genie.load_character("dq", MODEL_DIR, "Hybrid-Chinese-English")
genie.set_reference_audio("dq", REF_WAV, REF_TEXT, "Chinese")
print("loaded.")


async def run():
    t0 = time.perf_counter()
    pieces = []
    total_audio = 0.0
    async for chunk in genie.tts_async("dq", TEXT):
        now = time.perf_counter() - t0
        dur = len(chunk) / 2 / 32000  # int16 mono
        total_audio += dur
        print(f"  chunk #{len(pieces)+1}: t={now:6.2f}s  audio={dur:5.2f}s")
        pieces.append(np.frombuffer(chunk, dtype=np.int16))
    wall = time.perf_counter() - t0
    print(f"\nwall time {wall:.2f}s, audio {total_audio:.2f}s, RTFx {total_audio/wall:.2f}x")
    return pieces

pieces = asyncio.run(run())
full = np.concatenate(pieces)
import soundfile as sf
sf.write("test_dq_stream.wav", full.astype(np.float32) / 32767, 32000)
print("saved test_dq_stream.wav")
sd.play(full, 32000)
sd.wait()
print("playback finished")
