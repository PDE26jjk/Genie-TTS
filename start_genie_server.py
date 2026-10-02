# -*- coding: utf-8 -*-
# Genie-TTS API server launcher.
#
# Reads genie_server_config.json next to this script:
# {
#   "host": "127.0.0.1", "port": 9882,
#   "device": "gpu",            // gpu | dml | cpu (default cpu)
#   "dll_dirs": ["..."],        // extra Windows DLL dirs for CUDA (optional)
#   "characters": {             // loaded at startup
#     "dq": {
#       "onnx_model_dir": "...", "language": "Hybrid-Chinese-English",
#       "ref_audio": "...", "ref_text": "...", "ref_lang": "Chinese"
#     }
#   }
# }
#
# Run:  .venv\Scripts\python.exe start_genie_server.py
import argparse
import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "genie_server_config.json")

with open(CONFIG_FILE, encoding="utf-8") as f:
    cfg = json.load(f)

# resolve the final device first: GENIE_DEVICE (env) wins over the config,
# and the static CUDA-graph decode is GPU-only - on CPU the fp32 2048-token
# graph runs at RTFx ~0.3 while the dynamic graph is faster
device = os.environ.get("GENIE_DEVICE") or cfg.get("device", "cpu")
os.environ["GENIE_DEVICE"] = device
if cfg.get("static_decode") and device != "cpu":
    os.environ.setdefault("GENIE_STATIC_DECODE", "1")
if cfg.get("emotion_carry"):
    os.environ.setdefault("GENIE_EMOTION_CARRY", "1")
if cfg.get("carry_characters"):
    # whitelist of voices allowed to use the token-carry history; some
    # pretrained models degrade with a long history (re-voicing / EOS)
    os.environ.setdefault(
        "GENIE_CARRY_CHARACTERS",
        ",".join(cfg["carry_characters"]))
# every model (G2PW tokenizer included) is on local disk - never let a HF
# hub probe hang the startup or silently demote G2PW to the G2pM fallback
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
print("env:", os.environ.get("GENIE_EMOTION_CARRY"), "| static:", os.environ.get("GENIE_STATIC_DECODE"))


# DLL directories must be registered before the first CUDA session is created
for d in cfg.get("dll_dirs", []):
    if os.path.isdir(d):
        os.add_dll_directory(d)

import onnxruntime as ort  # noqa: E402

try:
    ort.preload_dlls()
except Exception as e:
    print("preload_dlls:", e)

import genie_tts as genie  # noqa: E402

for name, c in cfg.get("characters", {}).items():
    genie.load_character(name, c["onnx_model_dir"], c["language"])
    genie.set_reference_audio(name, c["ref_audio"], c["ref_text"], c.get("ref_lang", "Chinese"))
    print(f"character ready: {name}")

port = int(os.environ.get("GENIE_PORT", cfg.get("port", 9882)))
genie.start_server(cfg.get("host", "127.0.0.1"), port)
