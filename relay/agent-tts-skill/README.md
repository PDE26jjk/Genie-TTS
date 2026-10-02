# agent-tts — speech output for coding agents

`agent-tts` lets a coding agent speak text out loud on demand: the agent
posts text to the local TTS relay (`tts_relay_tray.py`, default
`http://127.0.0.1:9881`) and the current backend (Genie / GPT-SoVITS)
synthesizes and plays it with the active voice. No screen scraping, no
log watching — the agent calls the HTTP API explicitly.

The skill is agent-agnostic. Anything that can run a shell command can
use it; the `SKILL.md` merely teaches skill-capable agents when and how.

Files:

- `SKILL.md` — the skill definition (written in Chinese)
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

2. Any coding agent that can execute shell commands (ZCode, Claude Code,
   Codex, ...). Skill-file support is optional but recommended.

## Install (per user, per agent)

Copy this folder, keeping the folder name `agent-tts`, into the skill
directory of whichever agent(s) you use:

| Agent          | Skill directory                          |
|----------------|------------------------------------------|
| ZCode          | `~/.zcode/skills/agent-tts`              |
| Claude Code    | `~/.claude/skills/agent-tts`             |
| Codex          | `$CODEX_HOME/skills/agent-tts` (default `~/.codex/skills`) |
| anything else  | no install needed — see below            |

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

`<repo>` is your Genie-TTS checkout. Restart the agent session so its
skill list reloads.

**No skill mechanism?** Nothing to install: give the agent the absolute
path to `tts_say.py` (in its memory/instructions file, or just tell it
in conversation) and it can run the same commands. The script is the
whole integration.

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

`<skill-dir>` is the folder the SKILL.md was loaded from (or wherever
`tts_say.py` lives). Long texts are split into sentences by the relay
and queued automatically.

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
