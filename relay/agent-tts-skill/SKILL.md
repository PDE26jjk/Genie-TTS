---
name: agent-tts
description: 让 Agent 通过本机 TTS 中转服务把文本合成语音并播放。当用户要求"朗读 / 念出来 / 播报 / 语音播放 / TTS / 试音 / 用声音说"，或 Agent 需要主动以语音输出一段内容时使用。直接调用本机 HTTP 接口（默认 127.0.0.1:9881），不解析任何界面或日志。
---

# Agent-TTS 语音播报

把文本推给本机 TTS relay（托盘程序 `tts_relay_tray.py`，默认监听 `127.0.0.1:9881`），由当前后端（Genie 或 GPT-SoVITS）和当前音色合成播放。队列串行：多段文本按顺序播完。

## 调用方式（首选脚本，标准库实现，任意 Python 可跑）

`<技能目录>` 即本 SKILL.md 所在目录（Agent 加载技能时会显示其路径）：

```
python "<技能目录>\tts_say.py" "要朗读的文本"
```

| 操作 | 命令 |
|---|---|
| 播报文本 | `tts_say.py "文本"` |
| 从 stdin 读长文本 | `... --stdin < file.txt` |
| 查看状态/队列 | `tts_say.py --status` |
| 停止并清空队列 | `tts_say.py --stop` |
| relay 地址不同 | 加 `--url http://127.0.0.1:端口` |

成功时 stdout 返回 JSON：`{"queued": <项数>, "status": {...}}`。

## 使用规则

1. **发给脚本的文本必须是"可读的口语"**：先去掉代码块、行内代码、URL、表格、文件路径和 markdown 符号；代码不要朗读，用自己的话概述。
2. Windows 命令行里**中文经 curl `-d` 会乱码**——一律用本脚本（或 `curl --data-binary @file`），不要用 curl 内联中文。
3. 长文本整体传入即可，relay 会自动拆句排队；播放途中可随时 `--stop` 打断。
4. relay 未运行时脚本报 `relay unreachable`。启动方式（后台托盘，无窗口，用 Genie-TTS 的 .venv 或已装 pystray/sounddevice 的 Python）：
   `pythonw tts_relay_tray.py`（该脚本位于 Genie-TTS 仓库的 `relay/` 目录）。
5. 音色/后端切换在托盘菜单操作，接口层无需关心。
