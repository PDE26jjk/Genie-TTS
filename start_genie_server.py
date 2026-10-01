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

os.environ.setdefault("GENIE_DEVICE", cfg.get("device", "cpu"))
os.environ.setdefault("GENIE_STATIC_DECODE", "1" if cfg.get("static_decode") else "0")


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

genie.start_server(cfg.get("host", "127.0.0.1"), int(cfg.get("port", 9882)))
