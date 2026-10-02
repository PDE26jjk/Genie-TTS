<div align="center">
<pre>
██████╗  ███████╗███╗   ██╗██╗███████╗
██╔════╝ ██╔════╝████╗  ██║██║██╔════╝
██║  ███╗█████╗  ██╔██╗ ██║██║█████╗  
██║   ██║██╔══╝  ██║╚██╗██║██║██╔══╝  
╚██████╔╝███████╗██║ ╚████║██║███████╗
 ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝╚══════╝
</pre>
</div>

<div align="center">

# 🔮 GENIE: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) Lightweight Inference Engine

**Experience near-instantaneous speech synthesis on your CPU**

[简体中文](./README_zh.md) | [English](./README.md)

</div>

---

**GENIE** is a lightweight inference engine built on the open-source TTS
project [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). It integrates TTS inference, ONNX model conversion, API
server, and other core features, aiming to provide ultimate performance and convenience.

* **✅ Supported Model Version:** GPT-SoVITS V2, V2Pro, V2ProPlus
* **✅ Supported Language:** Japanese, English, Chinese, Korean
* **✅ Supported Python Version:** >= 3.10

---

## 🎬 Demo Video

- **[➡️ Watch the demo video (Chinese)](https://www.bilibili.com/video/BV1d2hHzJEz9)**

---

## 🚀 Performance Advantages

GENIE optimizes the original model for outstanding CPU performance.

| Feature                     |  🔮 GENIE   | Official PyTorch Model | Official ONNX Model |
|:----------------------------|:-----------:|:----------------------:|:-------------------:|
| **First Inference Latency** |  **1.13s**  |         1.35s          |        3.57s        |
| **Runtime Size**            | **\~200MB** |      \~several GB      |  Similar to GENIE   |
| **Model Size**              | **\~230MB** |    Similar to GENIE    |       \~750MB       |

> 📝 **Latency Test Info:** All latency data is based on a test set of 100 Japanese sentences (\~20 characters each),
> averaged. Tested on CPU i7-13620H.

---

## 🏁 QuickStart

> **⚠️ Important:** It is recommended to run GENIE in **Administrator mode** to avoid potential performance degradation.

### 📦 Installation

Install via pip:

```bash
pip install genie-tts
```

## 📥 Pretrained Models

When running GENIE for the first time, it requires downloading resource files (**~391MB**). You can follow the library's
prompts to download them automatically.

> Alternatively, you can manually download the files
> from [HuggingFace](https://huggingface.co/High-Logic/Genie/tree/main/GenieData)
> and place them in a local folder. Then set the `GENIE_DATA_DIR` environment variable **before** importing the library:

```python
import os

# Set the path to your manually downloaded resource files
# Note: Do this BEFORE importing genie_tts
os.environ["GENIE_DATA_DIR"] = r"C:\path\to\your\GenieData"

import genie_tts as genie

# The library will now load resources from the specified directory
```

If you want the optional Chinese RoBERTa text features used only for **Chinese inference**
to improve Chinese prosody, you can also download them with:

```python
import genie_tts as genie

# Download only the optional Chinese RoBERTa assets
genie.download_roberta_data()

# Or use the built-in full resource download flow,
# which now also downloads the optional Chinese RoBERTa assets
genie.download_genie_data()
```

These RoBERTa features are intended only for the **Chinese** path to improve Chinese prosody.
They **should not be used** for non-Chinese inference (Japanese / English / Korean).

### ⚡️ Quick Tryout

No GPT-SoVITS model yet? No problem!
GENIE includes several predefined speaker characters you can use immediately —
for example:

* **Mika (聖園ミカ)** — *Blue Archive* (Japanese)
* **ThirtySeven (37)** — *Reverse: 1999* (English)
* **Feibi (菲比)** — *Wuthering Waves* (Chinese)

You can browse all available characters here:
**[https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels](
https://huggingface.co/High-Logic/Genie/tree/main/CharacterModels)**

Try it out with the example below:

```python
import genie_tts as genie
import time

# Automatically downloads required files on first run
genie.load_predefined_character('mika')

genie.tts(
    character_name='mika',
    text='どうしようかな……やっぱりやりたいかも……！',
    play=True,  # Play the generated audio directly
)

genie.wait_for_playback_done()  # Ensure audio playback completes
```

### 🎤 TTS Best Practices

A simple TTS inference example:

```python
import genie_tts as genie

# Step 1: Load character voice model
genie.load_character(
    character_name='<CHARACTER_NAME>',  # Replace with your character name
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",  # Folder containing ONNX model
    language='<LANGUAGE_CODE>',  # Replace with language code, e.g., 'en', 'zh', 'jp', 'kr'
)

# Step 2: Set reference audio (for emotion and intonation cloning)
genie.set_reference_audio(
    character_name='<CHARACTER_NAME>',  # Must match loaded character name
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",  # Path to reference audio
    audio_text="<REFERENCE_AUDIO_TEXT>",  # Corresponding text
)

# Step 3: Run TTS inference and generate audio
genie.tts(
    character_name='<CHARACTER_NAME>',  # Must match loaded character
    text="<TEXT_TO_SYNTHESIZE>",  # Text to synthesize
    play=True,  # Play audio directly
    save_path="<OUTPUT_AUDIO_PATH>",  # Output audio file path
)

genie.wait_for_playback_done()  # Ensure audio playback completes

print("🎉 Audio generation complete!")
```

---

## 🔧 Model Conversion

To convert original GPT-SoVITS models for GENIE, ensure `torch` is installed:

```bash
pip install torch
```

Use the built-in conversion tool:

> **Tip:** `convert_to_onnx` currently supports V2, V2Pro and V2ProPlus models (pass `version="v2Pro"` explicitly for v2Pro checkpoints; the auto-detection by file size cannot distinguish v2Pro from v2ProPlus, which is fine because they share the same architecture).

```python
import genie_tts as genie

genie.convert_to_onnx(
    torch_pth_path=r"<YOUR .PTH MODEL FILE>",  # Replace with your .pth file
    torch_ckpt_path=r"<YOUR .CKPT CHECKPOINT FILE>",  # Replace with your .ckpt file
    output_dir=r"<ONNX MODEL OUTPUT DIRECTORY>"  # Directory to save ONNX model
)
```

A command-line wrapper for the same call ships as
`convert_ckpt_to_onnx.py <ckpt> <pth> <out_dir> [version]`. See
**Voice Model Graphs** below for turning the components into the static
CUDA-graph decode.

---

## 🧩 Voice Model Graphs: dynamic vs static

Every character directory (`<model_dir>`, e.g.
`CharacterModels/v2ProPlus/feibi/tts_models`) holds the **dynamic** ONNX
components produced by `convert_ckpt_to_onnx.py` (see Model Conversion
above):

```
t2s_encoder_fp32.onnx + .bin      t2s_first_stage_decoder_fp32.onnx
t2s_stage_decoder_fp32.onnx + .bin (shared fp16 weights)
vits_fp32.onnx + .bin             prompt_encoder_fp32.onnx + .bin
```

On top of those you can optionally add the **static** CUDA-graph decode of
the t2s stage (fixed shapes, fixed memory, no allocation jitter):

```
t2s_static.onnx          fp32 graph
t2s_static_fp16.onnx     fp16 graph (preferred on CUDA)
```

**Which one runs where** (decided automatically at character load):

| Device | `static_decode` config | Graph actually used |
|---|---|---|
| GPU (CUDA) | `true` | `t2s_static_fp16.onnx` if present, else dynamic |
| GPU (CUDA) | `false` | dynamic |
| CPU | anything | **dynamic always** - the fp32 2048-token static graph is ~2x slower on CPU |

Static decode gives a fixed memory footprint and smooth streaming on GPU
but is meaningless on CPU, so the launcher (`start_genie_server.py`)
enables it only when the resolved device is not `cpu`.

**Conversion chain** (all steps in this repo):

```text
GPT-SoVITS checkpoint pair (.ckpt + .pth)
      |  convert_ckpt_to_onnx.py <ckpt> <pth> <out_dir> [v2|v2Pro|v2ProPlus]
      v
dynamic components  <model_dir>
      |  python export_static_t2s.py --components <model_dir>            --ckpt <any .ckpt, config source> --gptsovits <GPT-SoVITS package>            --out <model_dir>/t2s_static.onnx --max-len 2048
      v
t2s_static.onnx (fp32)
      |  python convert_static_fp16.py --model-dir <model_dir>
      v
t2s_static_fp16.onnx
```

- `--components` takes weights from the dynamic components (HF characters
  ship no checkpoint); `--ckpt <gpt_ckpt>` exports straight from a
  fine-tuned checkpoint instead.
- `--gptsovits` points at a GPT-SoVITS package root (contains `AR/`); the
  exporter needs `torch` (2.x with the legacy ONNX exporter), `onnx` and
  `torchmetrics` importable.
- The t2s architecture is identical across v2 / v2Pro / v2ProPlus / v3, so
  one exporter serves every version; only the s2 vocoder side differs.
- After exporting, reload the character (server restart or a tray model
  switch) to pick up the static graph.

## 🌐 Launch FastAPI Server

GENIE includes a lightweight FastAPI server:

```python
import genie_tts as genie

# Start server
genie.start_server(
    host="0.0.0.0",  # Host address
    port=8000,  # Port
    workers=1  # Number of workers
)
```

> For request formats and API details, see our [API Server Tutorial](./Tutorial/English/API%20Server%20Tutorial.py).


---

## 🎛️ TTS Relay (System Tray)

`relay/tts_relay_tray.py` is a standalone system-tray app (developed in this
fork) that owns a **single playback queue** and fronts **both** inference
backends - the full GPT-SoVITS runtime (`api_v2.py`, torch, real-time) and
this lightweight Genie ONNX server - so you can switch voices/engines from a
menu and push text into the queue from any agent or script.

```
any agent ──HTTP──▶ Relay (127.0.0.1:9881) ──▶ backend 9880 (GPT-SoVITS api_v2)
   tray ◀──menu/control── (queue + volume)  ──▶ backend 9882 (Genie, CUDA Graph)
                                     └──▶ speakers (pipelined playback)
```

Features:
- Tray menu: start / stop-and-clear the stream, **volume**, **backend switch**
  (GPT-SoVITS / Genie), model or checkpoint submenu per backend, live status.
- Pipelined playback: generation of the next utterance runs while the current
  one is still playing (no gap between queued sentences).
- Backend switching: for GPT-SoVITS it hot-swaps fine-tuned checkpoints via
  `/set_gpt_weights` + `/set_sovits_weights` (train → test immediately); for
  Genie it loads ONNX characters via `/load_character`.

### Setup

Extra dependency only for the tray UI: `pip install pystray pillow`
(requests / numpy / sounddevice are already Genie dependencies).

1. Create the runtime configs from the templates (personal configs are
   gitignored - a fresh clone only has the templates):

   ```powershell
   Copy-Item relay	ts_relay_config.example.json relay	ts_relay_config.json
   Copy-Item genie_server_config.example.json genie_server_config.json
   ```

   The templates ship the official **Feibi** character
   (`CharacterModels/v2ProPlus/feibi`, auto-downloaded from HuggingFace) as
   the default voice. Then edit `relay/tts_relay_config.json`; the common
   fields:
   - `backends`: URL + type of each backend (`gptsovits` or `genie`).
   - `checkpoints`: GPT-SoVITS fine-tuned checkpoints (paths are resolved by
     the api_v2 process, so keep them relative to that package).
   - `genie_characters`: ONNX character dirs for the Genie backend
     (`onnx_model_dir` is required - switching loads the character by it).
   - `ref_audio` / `prompt_text`: the voice-cloning reference (absolute path).
   - `test_text`, `volume`, `auto_play`, `seed` (-1 = random prosody).
2. Start the backends you want:

   ```powershell
   # (1) api_v2 from the GPT-SoVITS package (optional; port 9880 = the
   #     relay's GPT-SoVITS backend)
   cd <GPT-SoVITS package dir>
   runtime\python.exe api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS\configs	ts_infer.yaml

   # (2) Genie GPU mode (repo root; device comes from genie_server_config.json's
   #     "gpu", static CUDA-graph enabled automatically, port 9882)
   cd <Genie-TTS repo dir>
   .venv\Scripts\python.exe start_genie_server.py

   # (3) Genie CPU mode (static decode disabled automatically; port 9883 =
   #     the relay's Genie-CPU backend)
   $env:GENIE_DEVICE='cpu'; $env:GENIE_PORT='9883'
   .venv\Scripts\python.exe start_genie_server.py
   ```

   `GENIE_DEVICE` overrides the config file's `device` field. On GPU,
   `static_decode=true` plus a `t2s_static_fp16.onnx` in the character
   directory selects the static CUDA-graph decode (see **Voice Model
   Graphs** above).
3. Start the relay:

```powershell
cd relay
..\.venv\Scripts\pythonw.exe tts_relay_tray.py   # pythonw = no console
```

### HTTP API (default 127.0.0.1:9881)

| Endpoint | Body | Effect |
|---|---|---|
| `POST /say` | `{"text": "...", "play": true?}` | enqueue text (auto-plays by default) |
| `POST /start` / `POST /stop` | - | resume queue / abort current + clear |
| `POST /backend` | `{"name": "Genie(ONNX)"}` | switch backend |
| `POST /model` | `{"name": "feibi-ONNX"}` | switch checkpoint/character |
| `POST /volume` | `{"volume": 0.3}` | 0.0 – 1.5, persisted |
| `GET /status` | - | state, queue, backend, model, volume |

Example (PowerShell-friendly, UTF-8 safe):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:9881/say -Method Post ^
  -ContentType "application/json; charset=utf-8" -Body '{"text":"你好。"}'
```

---

## 📝 Roadmap

* [x] **🌐 Language Expansion**

    * [x] Add support for **Chinese** and **English**.

* [x] **🚀 Model Compatibility**

    * [x] Support for `V2Pro` and `V2ProPlus`.
    * [ ] Support for `V3`, `V4`, and more.

* [x] **📦 Easy Deployment**

    * [ ] Release **Official Docker images**.
    * [x] Provide out-of-the-box **Windows bundles**.

---
