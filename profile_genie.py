# -*- coding: utf-8 -*-
# Profile Genie inference cost split on GPU:
#   1. t2s stage_decoder ms/step (growing KV) - is IOBinding worth it?
#   2. vits decode cost vs token count (prefix re-decode amplification)
import os
import time

os.add_dll_directory(r'R:\git\dsh_projects\cyber_persona\CHARM\.venv\Lib\site-packages\torch\lib')
import numpy as np
import onnxruntime as ort

ort.preload_dlls()
DIR = r"models_dq_v2pro"
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

stage = ort.InferenceSession(f"{DIR}/t2s_stage_decoder_fp32.onnx", so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inames = [i.name for i in stage.get_inputs()]
print("stage inputs:", [(i.name, i.shape) for i in stage.get_inputs()][:4], "... layers:",
      sum(1 for i in stage.get_inputs() if i.name.startswith('past_k')))

NLAYERS = sum(1 for i in stage.get_inputs() if i.name.startswith('past_k'))
HID = 512
vocab = 2333

def run_steps(n_steps, seq0=24):
    """Measure per-step wall time as KV grows from seq0."""
    y = np.zeros((1, seq0), dtype=np.int64)
    y_emb = np.zeros((1, seq0, HID), dtype=np.float32)
    kv = [np.zeros((seq0, 1, HID), dtype=np.float32) for _ in range(2 * NLAYERS)]
    times = []
    for s in range(n_steps):
        feed = {}
        for name in inames:
            if name == 'iy':
                feed[name] = y
            elif name == 'iy_emb':
                feed[name] = y_emb
            else:
                idx = int(name.split('_layer_')[1])
                feed[name] = kv[0]  # placeholder, replaced below
        # build feed properly
        ki = 0
        for name in inames:
            if name.startswith('past_k'):
                feed[name] = kv[ki]; ki += 1
            elif name.startswith('past_v'):
                feed[name] = kv[ki]; ki += 1
        t0 = time.perf_counter()
        outputs = stage.run(None, feed)
        dt = time.perf_counter() - t0
        times.append(dt)
        # grow by one step (shapes must stay consistent for next call)
        y = np.concatenate([y, np.zeros((1, 1), dtype=np.int64)], axis=1)
        y_emb = np.concatenate([y_emb, np.zeros((1, 1, HID), dtype=np.float32)], axis=1)
        kv = [np.concatenate([k, np.zeros((1, 1, HID), dtype=np.float32)], axis=0) for k in kv]
    return times

ts = run_steps(60)
warm = ts[10:]
print(f"stage_decoder: avg {np.mean(warm)*1000:.2f} ms/step (steps 10-60), "
      f"first {ts[0]*1000:.2f} ms, last {ts[-1]*1000:.2f} ms")

# vits decode cost vs token count
vits = ort.InferenceSession(f"{DIR}/vits_fp32.onnx", so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
vin = {i.name: i for i in vits.get_inputs()}
print("vits inputs:", {k: v.shape for k, v in vin.items()})
ref = np.zeros((1, 32000), dtype=np.float32)
ge = np.zeros((1, 512), dtype=np.float32)
gea = np.zeros((1, 512), dtype=np.float32)
for n_tok in (16, 48, 96, 192, 288):
    feed = {"text_seq": np.zeros((1, 50), dtype=np.int64), "pred_semantic": np.zeros((1, 1, n_tok), dtype=np.int64),
            "ge": ge, "ge_advanced": gea}
    if "ref_audio" in vin:
        feed = {"text_seq": feed["text_seq"], "pred_semantic": feed["pred_semantic"], "ref_audio": ref}
    t0 = time.perf_counter()
    for _ in range(3):
        vits.run(None, feed)
    dt = (time.perf_counter() - t0) / 3
    print(f"vits decode {n_tok:4d} tokens ({n_tok/25:5.1f}s audio): {dt*1000:8.1f} ms")
