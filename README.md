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

---

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

1. Edit `relay/tts_relay_config.json`:
   - `backends`: URL + type of each backend (`gptsovits` or `genie`).
   - `checkpoints`: GPT-SoVITS fine-tuned checkpoints (paths are resolved by
     the api_v2 process, so keep them relative to that package).
   - `genie_characters`: ONNX character dirs for the Genie backend.
   - `ref_audio` / `prompt_text`: the voice-cloning reference (absolute path).
   - `test_text`, `volume`, `auto_play`, `seed` (-1 = random prosody).
2. Start the backends you want (api_v2 from the GPT-SoVITS package,
   `start_genie_server.py` from this repo).
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
| `POST /model` | `{"name": "dq-v2Pro(微调)"}` | switch checkpoint/character |
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
