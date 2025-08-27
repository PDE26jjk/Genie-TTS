<div align="center">

# 🔮 GENIE: Lightweight Inference Engine for [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)

**A high-performance, lightweight inference engine specifically designed
for [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)**

[简体中文](./README.md) | [English](./README_en.md)

</div>

---

**GENIE** is a lightweight inference engine built on the open-source TTS
project [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS). It integrates core features such as TTS inference, ONNX
model conversion, and an API server, aiming to deliver maximum performance and a smooth user experience.

* **✅ Supported Model Version:** GPT-SoVITS V2
* **✅ Supported Language:** Japanese

## 🚀 Performance Advantages

GENIE is highly optimized compared to the original model, demonstrating excellent performance on CPU environments.

| Feature                     |  🔮 GENIE   | Official PyTorch Model | Official ONNX Model |
|:----------------------------|:-----------:|:----------------------:|:-------------------:|
| **First Inference Latency** |  **1.13s**  |         1.35s          |        3.57s        |
| **Runtime Size**            | **\~200MB** |      \~several GB      |  Similar to GENIE   |
| **Model Size**              | **\~230MB** |    Similar to GENIE    |       \~750MB       |

> 📝 **Note:** Since GPU inference does not show a significant latency improvement over CPU for first-inference, we
> currently only provide a CPU version to ensure the best out-of-the-box experience.
>
> 📝 **Latency Test Details:** All latency data is measured using a test set of 100 Japanese sentences (\~20 characters
> each), with averages calculated. Tests were conducted on a CPU i7-12620H.

---

## 🏁 Quick Start

> **⚠️ Important:** It is recommended to run GENIE in **Administrator mode** to avoid potential severe performance
> drops.

### 📦 Installation

Install via pip:

```bash
pip install genie-tts
```

### 🎤 Usage Example

Here’s a simple TTS inference example:

```python
import genie_tts as genie

# Step 1: Load the character voice model
genie.load_character(
    character_name='<CHARACTER_NAME>',  # Replace with your character name
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",  # Replace with folder path containing ONNX models
)

# Step 2: Set reference audio (for emotion and intonation cloning)
genie.set_reference_audio(
    character_name='<CHARACTER_NAME>',  # Must match the loaded character name
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",  # Replace with your reference audio file path
    audio_text="<REFERENCE_AUDIO_TEXT>",  # Replace with the text corresponding to the reference audio
)

# Step 3: Perform TTS inference and generate audio
genie.tts(
    character_name='<CHARACTER_NAME>',  # Must match the loaded character name
    text="<TEXT_TO_SYNTHESIZE>",  # Replace with the text you want to synthesize
    play=True,  # Set to True to play the generated audio directly
    save_path="<OUTPUT_AUDIO_PATH>",  # Replace with desired output path
)

print("🎉 Audio generation complete!")
```

## 🔧 Model Conversion

If you need to convert the original GPT-SoVITS model to a format compatible with GENIE, first ensure that `torch` is
installed:

```bash
pip install torch
```

Then, you can use the built-in conversion tool.

> **Tip:** The `convert_to_onnx` function currently only supports V2 models.

```python
import genie_tts as genie

genie.convert_to_onnx(
    torch_pth_path=r"<PATH_TO_YOUR_.PTH_MODEL>",  # Replace with your .pth model path
    torch_ckpt_path=r"<PATH_TO_YOUR_.CKPT_CHECKPOINT>",  # Replace with your .ckpt checkpoint path
    output_dir=r"<OUTPUT_ONNX_MODEL_DIR>"  # Specify directory to save ONNX model
)
```

## 🌐 Start FastAPI Server

GENIE comes with a simple built-in FastAPI server.

```python
import genie_tts as genie

# Start the server
genie.start_server(
    host="0.0.0.0",  # Host address to bind
    port=8000,  # Port to listen on
    workers=1  # Number of worker processes
)
```

> For server request formats, endpoint details, and usage instructions, please refer to
> our [API Server Tutorial](./Tutorial/English/API%20Server%20Tutorial.py).

---