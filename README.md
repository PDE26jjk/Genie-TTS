<div align="center">

# 🔮 GENIE: [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 轻量级推理引擎

**专为 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 设计的高性能、轻量级的推理引擎**

[简体中文](./README.md) | [English](./README_en.md)

</div>

---

**GENIE** 是基于开源 TTS 项目 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 打造的轻量级推理引擎，集成了
TTS 推理、ONNX 模型转换、API Server 等核心功能，旨在提供更极致的性能与更便捷的体验。

- **✅ 支持模型版本:** GPT-SoVITS V2
- **✅ 支持语言:** 日语 (Japanese)

## 🚀 性能优势

GENIE 对原版模型进行了高度优化，在 CPU 环境下展现了卓越的性能。

| 特性        |  🔮 GENIE  | 官方 Pytorch模型 | 官方 ONNX 模型 |
|:----------|:----------:|:------------:|:----------:|
| **首包延迟**  | **1.13s**  |    1.35s     |   3.57s    |
| **运行时大小** | **~200MB** |    ~数 GB     | 与 GENIE 类似 |
| **模型大小**  | **~230MB** |  与 GENIE 类似  |   ~750MB   |

> 📝 **备注:** 由于 GPU 推理的首包延迟与 CPU 相比未拉开显著差距，我们暂时仅发布 CPU 版本，以提供最佳的开箱即用体验。
>
> 📝 **延迟测试说明:** 所有延迟数据基于一个包含 100 个日语句子的测试集，每句约 20 个字符，取平均值计算。在 CPU i7-12620H
> 上进行推理测试。
---

## 🏁 快速开始 (QuickStart)

> **⚠️ 重要提示:** 建议在 **管理员模式 (Administrator)** 下运行 GENIE，以避免潜在的严重性能下降问题。

### 📦 安装 (Installation)

通过 pip 安装：

```bash
pip install genie-tts
```

### 🔗 依赖项下载

对于中国大陆用户，我们强烈建议您手动下载必要的依赖项，并将模型与字典文件放置在某个本地位置。

| 下载渠道         | 链接                                                                                           |
|:-------------|:---------------------------------------------------------------------------------------------|
| 腾讯微云         | [https://share.weiyun.com/0Jtg2dYT](https://share.weiyun.com/0Jtg2dYT)                       |
| Hugging Face | [https://huggingface.co/High-Logic/Genie/tree/main](https://huggingface.co/High-Logic/Genie) |

下载后，请通过环境变量 (os.environ) 指定文件路径。

### 🎤 使用示例 (Usage)

下面是一个简单的 TTS 推理示例：

```python
import os

# (可选) 设置 HuBERT 中文模型路径。若不设置，程序将尝试从 Hugging Face 自动下载。
os.environ['HUBERT_MODEL_PATH'] = r"C:\path\to\your\chinese-hubert-base.onnx"

# (可选) 设置 Open JTalk 字典文件夹路径。若不设置，程序将尝试从 Github 自动下载。
os.environ['OPEN_JTALK_DICT_DIR'] = r"C:\path\to\your\open_jtalk_dic_utf_8-1.11"

import genie_tts as genie

# 步骤 1: 加载角色声音模型
genie.load_character(
    character_name='<CHARACTER_NAME>',  # 替换为你的角色名称
    onnx_model_dir=r"<PATH_TO_CHARACTER_ONNX_MODEL_DIR>",  # 替换为包含 ONNX 模型的文件夹路径
)

# 步骤 2: 设置参考音频 (用于情感和语调克隆)
genie.set_reference_audio(
    character_name='<CHARACTER_NAME>',  # 确保与加载的角色名称一致
    audio_path=r"<PATH_TO_REFERENCE_AUDIO>",  # 替换为你的参考音频文件路径
    audio_text="<REFERENCE_AUDIO_TEXT>",  # 替换为参考音频对应的文本
)

# 步骤 3: 执行 TTS 推理并生成音频
genie.tts(
    character_name='<CHARACTER_NAME>',  # 确保与加载的角色名称一致
    text="<TEXT_TO_SYNTHESIZE>",  # 替换为你想要合成的文本
    play=True,  # 设置为 True 可直接播放生成的音频
    save_path="<OUTPUT_AUDIO_PATH>",  # 替换为期望的音频保存路径
)

print("🎉 音频生成完毕!")
```

## 🔧 模型转换 (Model Conversion)

如果您需要将原始的 GPT-SoVITS 模型转换为 GENIE 使用的格式，请先确保已安装 `torch`。

```bash
pip install torch
```

然后，您可以使用内置的转换工具。

> **提示:** 目前 `convert_to_onnx` 函数仅支持转换 V2 版本的模型。

```python
import genie_tts as genie

genie.convert_to_onnx(
    torch_pth_path=r"<你的 .pth 模型文件路径>",  # 替换为您的 .pth 模型文件路径
    torch_ckpt_path=r"<你的 .ckpt 检查点文件路径>",  # 替换为您的 .ckpt 检查点文件路径
    output_dir=r"<ONNX 模型输出文件夹路径>"  # 指定 ONNX 模型保存的目录
)
```

## 🌐 启动 FastAPI 服务器

GENIE 内置了一个简单的 FastAPI 服务器。

```python
import os

os.environ['HUBERT_MODEL_PATH'] = r"C:\path\to\your\chinese-hubert-base.onnx"
os.environ['OPEN_JTALK_DICT_DIR'] = r"C:\path\to\your\open_jtalk_dic_utf_8-1.11"

import genie_tts as genie

# 启动服务器
genie.start_server(
    host="0.0.0.0",  # 监听的主机地址
    port=8000,  # 监听的端口
    workers=1  # 工作进程数
)
```

> 关于服务器的请求格式、接口详情等信息，请参考我们的 [API 服务器使用教程](./Tutorial/English/API%20Server%20Tutorial.py)。

---