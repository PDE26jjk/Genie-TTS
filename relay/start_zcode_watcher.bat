@echo off
rem Start the ZCode speech watcher (reads ZCode rollout files, speaks
rem replies through the relay). Hidden window; logs go to zcode_watcher.log.
start "zcode-watcher" /min "R:\git\dsh_projects\cyber_persona\Genie-TTS\.venv\Scripts\python.exe" "R:\git\dsh_projects\cyber_persona\Genie-TTS\relay\zcode_speech_watcher.py"
