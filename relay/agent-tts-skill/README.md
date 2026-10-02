# agent-tts — a ZCode skill for speech output

`agent-tts` lets a ZCode agent speak text out loud on demand: the agent
posts text to the local TTS relay (`tts_relay_tray.py`, default
`http://127.0.0.1:9881`) and the current backend (Genie / GPT-SoVITS)
synthesizes and plays it with the active voice. No screen scraping, no
log watching — the agent calls the HTTP API explicitly.

Files:

- `SKILL.md` — the skill definition (loaded by ZCode, written in Chinese)
- `tts_say.py` — the helper script the skill tells the agent to run
  (standard library only, works with any Python 3.8+)

## Prerequisites

1. A working TTS relay: start the tray app from this directory with a
   Python that has the relay dependencies installed (Genie-TTS `.venv`):

   ```
   pythonw tts_relay_tray.py
   ```

   Leave it running in the tray. The port is set by `listen_port` in
   `tts_relay_config.json` (default 9881).

2. ZCode (desktop or CLI) with skill support — any recent version.

## Install (per user)

Copy this folder into the ZCode user-level skill directory, keeping the
folder name `agent-tts`:

**Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.zcode\skills" | Out-Null
Copy-Item -Recurse -Force "<repo>\relay\agent-tts-skill" "$env:USERPROFILE\.zcode\skills\agent-tts"
```

**macOS / Linux:**

```bash
mkdir -p ~/.zcode/skills
cp -r <repo>/relay/agent-tts-skill ~/.zcode/skills/agent-tts
```

`<repo>` is your Genie-TTS checkout. The skill is user-level, so it is
available in every workspace. Start (or restart) a ZCode session so the
skill list reloads, then check it appears in the available skills.

## Usage

Just ask the agent in natural language:

> 把这段话读出来：……
> 用语音播报一下刚才的结论

Behind the scenes the agent runs:

```
python "<skill-dir>\tts_say.py" "要朗读的文本"
python "<skill-dir>\tts_say.py" --status     # queue / playing state
python "<skill-dir>\tts_say.py" --stop       # abort current speech, clear queue
```

`<skill-dir>` is the folder the SKILL.md was loaded from. Long texts are
split into sentences by the relay and queued automatically.

## Configuration

- Relay address / port: `tts_relay_config.json` (`listen_host`,
  `listen_port`), or pass `--url http://127.0.0.1:<port>` per call.
- Voice / backend: switch in the tray menu; the skill never needs changes.

## Notes

- Text handed to the skill should already be speakable prose (no code
  blocks, URLs, tables); the relay strips leftovers but the agent is
  expected to summarize code instead of reading it.
- On Windows, never inline Chinese text into `curl -d` — the console
  mangles UTF-8. Use this script (or `curl --data-binary @file`).
