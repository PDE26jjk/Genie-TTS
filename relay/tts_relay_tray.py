# -*- coding: utf-8 -*-
# Single-script TTS relay for GPT-SoVITS: a system-tray app that owns the
# playback stream and exposes a small HTTP API so external agents (ZCode
# hooks, dsh, codex, ...) can push text into it.
#
# Tray menu:  start the stream / stop & clear / switch checkpoint / quit
# HTTP API (default 127.0.0.1:9881):
#   POST /say        {"text": "...", "play": true?}   enqueue text
#   POST /start      begin consuming the queue
#   POST /stop       abort current utterance and clear the queue
#   POST /checkpoint {"name": "<key in tts_relay_config.json>"}
#   GET  /status     {"state": "...", "queue": [...], "checkpoint": "..."}
#
# Run:  runtime\pythonw.exe tts_relay_tray.py   (pythonw hides the console)
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pystray
import requests
import sounddevice as sd
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "tts_relay_config.json")

DEFAULTS = {
    "api_url": "http://127.0.0.1:9880",
    "listen_host": "127.0.0.1",
    "listen_port": 9881,
    "volume": 1.0,
    "ref_audio": "R:\\git\\dsh_projects\\cyber_persona\\GPT-SoVITS-v2pro-20250604\\output\\slicer_opt\\dq1.wav.reformatted_vocals.flac_0001392000_0001570560.wav",
    "prompt_text": "期待明年，明年一定做，今年我也有在想。",
    "text_lang": "auto",
    "text_split": "cut5",
    "prompt_lang": "zh",
    "sr": 32000,
    "auto_play": True,
    "seed": -1,
    "test_text": "你好，这是一段测试文本，用来快速检查当前后端和音色是否正常。",
    "checkpoints": {
        "dq-v2Pro(微调)": {
            "gpt": "GPT_weights_v2Pro/dq-e15.ckpt",
            "sovits": "SoVITS_weights_v2Pro/dq_e8_s176.pth",
        },
        "v2Pro-预训练": {
            "gpt": "GPT_SoVITS/pretrained_models/s1v3.ckpt",
            "sovits": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth",
        },
    },
    "default_checkpoint": "dq-v2Pro(微调)",
}


def load_config():
    cfg = dict(DEFAULTS)
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception as e:
        print("config load failed, using defaults:", e)
    return cfg


class Relay:
    """Owns the playback stream: queue of texts + play/stop state."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.volume = max(0.0, min(1.5, float(cfg.get("volume", 1.0))))
        self.backends = cfg.get("backends") or {
            "GPT-SoVITS": {"type": "gptsovits", "api_url": cfg.get("api_url", "http://127.0.0.1:9880")}
        }
        self.backend_name = cfg.get("default_backend") if cfg.get("default_backend") in self.backends \
            else next(iter(self.backends))
        # active model entry name (checkpoint for gptsovits, character for genie)
        self.model_name = cfg.get("default_checkpoint")
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.texts = []  # pending texts (FIFO)
        self.playing = False  # stream switch (tray start/stop)
        self.abort = threading.Event()  # abort the current utterance
        self.current_stream = None  # sounddevice stream, for hard abort
        self.icon = None  # set after tray creation
        # audio pipeline: generation runs ahead of playback through this queue
        # (int16 PCM chunks; None marks the end of one utterance)
        self.audio_q = queue.Queue(maxsize=2048)
        self._play_quit = object()
        # emotion carry-over: (wav_path, transcript) of the last utterance
        self._last_carry = None
        self._carry_seq = 0
        self._utt_end = object()
        self._stream_restart = object()

    # ---- backend/model helpers ----
    def backend(self):
        return self.backends.get(self.backend_name, {"type": "gptsovits", "api_url": self.cfg.get("api_url")})

    def _model_entry(self):
        return self.models_for(self.backend_name).get(self.model_name) or {}

    def _ref_pair(self):
        """(ref_audio, prompt_text) for the active model, falling back to the
        global config pair."""
        e = self._model_entry()
        return (e.get("ref_audio") or self.cfg.get("ref_audio"),
                e.get("prompt_text") or self.cfg.get("prompt_text"))

    def models_for(self, backend_name):
        b = self.backends.get(backend_name, {})
        if b.get("type") == "genie":
            return self.cfg.get("genie_characters") or {}
        return self.cfg.get("checkpoints") or {}

    def set_backend(self, name):
        if name not in self.backends:
            return False
        self.stop()
        self.backend_name = name
        models = self.models_for(name)
        if self.model_name not in models:
            self.model_name = next(iter(models), None)
        self._refresh_menu()
        return True

    # ---- control surface (called from HTTP threads / tray actions) ----
    def say(self, text, play=None):
        text = (text or "").strip()
        if not text:
            return 0
        with self.cond:
            self.texts.append(text)
            n = len(self.texts)
            # play=True forces start, play=False leaves the stream paused,
            # None defers to the auto_play config
            if play if play is not None else self.cfg.get("auto_play"):
                self.playing = True
            self.cond.notify_all()
        self._refresh_menu()
        return n

    def start(self):
        with self.cond:
            self.playing = True
            self.abort.clear()
            self.cond.notify_all()
        self._refresh_menu()

    def stop(self):
        with self.cond:
            self.playing = False
            self.texts.clear()
        self.abort.set()
        # drain pending audio
        while True:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break
        stream = self.current_stream
        if stream is not None:
            # abort() kills sounddevice's blocking-write machinery for THIS
            # stream object; the play thread must reopen a fresh one.
            try:
                stream.abort()
            except Exception:
                pass
        try:
            self.audio_q.put_nowait(self._stream_restart)
        except queue.Full:
            pass
        self._refresh_menu()

    def switch_model(self, name):
        """Switch the active model on the current backend."""
        models = self.models_for(self.backend_name)
        if name not in models:
            return False, "unknown model"
        self.stop()  # avoid glitching audio while weights swap
        b = self.backend()
        try:
            if b.get("type") == "genie":
                entry = models[name]
                r1 = requests.post(b["api_url"] + "/load_character", json={
                    "character_name": entry["character_name"],
                    "onnx_model_dir": entry.get("onnx_model_dir", ""),
                    "language": entry.get("language", "Chinese"),
                }, timeout=120)
                r2 = requests.post(b["api_url"] + "/set_reference_audio", json={
                    "character_name": entry["character_name"],
                    "audio_path": entry.get("ref_audio") or self.cfg["ref_audio"],
                    "audio_text": entry.get("prompt_text") or self.cfg["prompt_text"],
                    "language": self.cfg.get("prompt_lang", "zh"),
                }, timeout=120)
                ok = r1.ok and r2.ok
                msg = f"{r1.status_code}/{r2.status_code}"
            else:
                r1 = requests.get(b["api_url"] + "/set_gpt_weights",
                                  params={"weights_path": models[name]["gpt"]}, timeout=120)
                r2 = requests.get(b["api_url"] + "/set_sovits_weights",
                                  params={"weights_path": models[name]["sovits"]}, timeout=120)
                ok = r1.ok and r2.ok
                msg = f"{r1.status_code}/{r2.status_code}"
            if ok:
                self.model_name = name
            self._refresh_menu()
            return ok, msg
        except Exception as e:
            return False, str(e)

    def status(self):
        with self.lock:
            return {
                "state": "playing" if self.playing else "paused",
                "queue": list(self.texts),
                "backend": self.backend_name,
                "model": self.model_name,
                "volume": round(self.volume, 2),
                "speaking": self.current_stream is not None,
            }

    def set_volume(self, value):
        """Clamp to [0, 1.5], apply immediately and persist to the config file."""
        try:
            value = max(0.0, min(1.5, float(value)))
        except (TypeError, ValueError):
            return False
        with self.lock:
            self.volume = value
        try:
            cfg = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    cfg = json.load(f)
            cfg["volume"] = round(value, 2)
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("failed to persist volume:", e)
        self._refresh_menu()
        return True

    # ---- pipeline workers ----
    def gen_loop(self):
        """Generation thread: pulls texts and streams PCM into audio_q ahead
        of playback, so the backend synthesizes the next utterance while the
        current one is still playing."""
        while True:
            with self.cond:
                while not (self.playing and self.texts):
                    self.cond.wait()
                text = self.texts.pop(0)
            try:
                self._generate(text)
            except Exception:
                import traceback
                print("generate failed:", traceback.format_exc(), flush=True)
            finally:
                try:
                    self.audio_q.put_nowait(self._utt_end)
                except queue.Full:
                    pass

    def _play_loop(self):
        """Playback thread: drains audio_q into the device continuously.

        Owns the stream lifecycle: after every abort (stop/switch) or write
        error the stream is closed and a fresh one opened, because
        OutputStream.abort() permanently kills sounddevice's blocking-write
        machinery for that stream object (writes would go into a dead ring
        buffer -> permanent silence)."""
        while True:
            try:
                stream = sd.OutputStream(samplerate=int(self.cfg["sr"]),
                                         channels=1, dtype="int16")
                stream.start()
            except Exception as e:
                print("audio device init failed:", e, flush=True)
                return
            self.current_stream = stream
            reopen = False
            try:
                while True:
                    item = self.audio_q.get()
                    if item is self._play_quit:
                        return
                    if item is self._stream_restart:
                        reopen = True
                        break
                    if item is self._utt_end or item is None:
                        continue
                    data = np.frombuffer(item, dtype="<i2")
                    if self.volume != 1.0:
                        data = np.clip(data.astype(np.float32) * self.volume,
                                       -32767, 32767).astype("<i2")
                    try:
                        stream.write(data)
                    except Exception:
                        # stream was aborted underneath us; reopen a fresh one
                        reopen = True
                        break
            finally:
                self.current_stream = None
                try:
                    stream.close()
                except Exception:
                    pass
            # loop: reopen a fresh stream and keep serving the queue

    def _generate(self, text):
        self.abort.clear()
        b = self.backend()
        if b.get("type") == "genie":
            entry = (self.cfg.get("genie_characters") or {}).get(self.model_name, {})
            payload = {
                "character_name": entry.get("character_name", self.model_name),
                "text": text,
            }
        capture = (b.get("type") == "gptsovits" and self.cfg.get("emotion_carry"))
        captured = bytearray() if capture else None
        if b.get("type") == "gptsovits":
            ref_audio, prompt_text = self._ref_pair()
            if self.cfg.get("emotion_carry") and self._last_carry is not None:
                carry_path, carry_text = self._last_carry
                ref_audio, prompt_text = carry_path, carry_text
                print(f"[relay] emotion carry: ref={carry_path}", flush=True)
            payload = {
                "text": text,
                "text_lang": self.cfg["text_lang"],
                "ref_audio_path": ref_audio,
                "prompt_text": prompt_text,
                "prompt_lang": self.cfg["prompt_lang"],
                "text_split_method": self.cfg["text_split"],
                "streaming_mode": 2,
                "media_type": "raw",
            }
            seed = int(self.cfg.get("seed", -1))
            if seed >= 0:
                payload["seed"] = seed  # fixed seed -> reproducible audio
        t_start = time.perf_counter()
        first_byte = None
        with requests.post(b["api_url"] + "/tts", json=payload,
                           stream=True, timeout=30) as r:
            r.raise_for_status()
            buf = bytearray()
            for piece in r.iter_content(chunk_size=4096):
                if first_byte is None:
                    first_byte = time.perf_counter()
                    print(f"[relay] first byte: {first_byte - t_start:.2f}s "
                          f"({b.get('type')} / {self.model_name})", flush=True)
                if self.abort.is_set():
                    break
                buf.extend(piece)
                # raw mode is s16le mono: only queue whole samples
                usable = len(buf) - (len(buf) % 2)
                if usable > 0:
                    chunk = bytes(buf[:usable])
                    self.audio_q.put(chunk)
                    if captured is not None:
                        captured.extend(chunk)
                    del buf[:usable]
        if first_byte is not None:
            print(f"[relay] generated: total {time.perf_counter() - t_start:.2f}s, "
                  f"first byte at {first_byte - t_start:.2f}s", flush=True)
        if captured is not None and not self.abort.is_set():
            self._store_carry(bytes(captured), text)
        self._refresh_menu()

    def _store_carry(self, pcm: bytes, text: str):
        """Keep the finished utterance as the next request's emotion-carry
        reference (3-10 s; outside that range fall back to the anchor ref).
        Each utterance gets a UNIQUE filename because both backends cache
        reference features by path string."""
        import wave
        dur = len(pcm) / 2 / int(self.cfg["sr"])
        if not (3.0 <= dur <= 10.0):
            self._last_carry = None
            print(f"[relay] carry skipped, duration {dur:.1f}s out of 3-10s", flush=True)
            return
        self._carry_seq += 1
        path = os.path.join(BASE_DIR, f"carry_{self._carry_seq % 50:03d}.wav")
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(int(self.cfg["sr"]))
                w.writeframes(pcm)
            old = self._last_carry[0] if self._last_carry else None
            self._last_carry = (path, text)
            if old and old != path and os.path.exists(old):
                try:
                    os.remove(old)
                except OSError:
                    pass
            print(f"[relay] carry stored: {path} ({dur:.1f}s)", flush=True)
        except Exception as e:
            print("carry store failed:", e, flush=True)

    def _refresh_menu(self):
        if self.icon is not None:
            try:
                self.icon.update_menu()
            except Exception:
                pass


# ---- HTTP API ----
class Handler(BaseHTTPRequestHandler):
    relay = None  # injected at startup

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def log_message(self, fmt, *args):
        pass  # keep the console clean

    def do_POST(self):
        relay, body = self.relay, self._body()
        if self.path == "/say":
            n = relay.say(body.get("text", ""), body.get("play"))
            self._json({"queued": n, "status": relay.status()})
        elif self.path == "/start":
            relay.start()
            self._json({"ok": True, "status": relay.status()})
        elif self.path == "/stop":
            relay.stop()
            self._json({"ok": True, "status": relay.status()})
        elif self.path == "/model":
            ok, msg = relay.switch_model(body.get("name", ""))
            self._json({"ok": ok, "message": msg,
                        "status": relay.status()}, 200 if ok else 400)
        elif self.path == "/checkpoint":  # alias of /model
            ok, msg = relay.switch_model(body.get("name", ""))
            self._json({"ok": ok, "message": msg,
                        "status": relay.status()}, 200 if ok else 400)
        elif self.path == "/backend":
            ok = relay.set_backend(body.get("name", ""))
            self._json({"ok": ok, "status": relay.status()}, 200 if ok else 400)
        elif self.path == "/volume":
            ok = relay.set_volume(body.get("volume"))
            self._json({"ok": ok, "status": relay.status()}, 200 if ok else 400)
        else:
            self._json({"error": "unknown endpoint"}, 404)

    def do_GET(self):
        if self.path == "/status":
            self._json(self.relay.status())
        else:
            self._json({"error": "unknown endpoint"}, 404)


# ---- tray ----
def make_icon_image():
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=(40, 110, 220, 255))
    # simple speaker glyph: box + triangle + two sound arcs
    d.polygon([(16, 26), (26, 26), (36, 16), (36, 48), (26, 38), (16, 38)], fill="white")
    d.arc((34, 22, 48, 42), start=-55, end=55, fill="white", width=3)
    d.arc((38, 16, 58, 48), start=-55, end=55, fill="white", width=3)
    return img


def main():
    cfg = load_config()
    relay = Relay(cfg)
    Handler.relay = relay

    httpd = ThreadingHTTPServer((cfg["listen_host"], int(cfg["listen_port"])), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    threading.Thread(target=relay.gen_loop, daemon=True).start()
    threading.Thread(target=relay._play_loop, daemon=True).start()
    print(f"relay ready: http://{cfg['listen_host']}:{cfg['listen_port']}")

    def start_stream(icon, item):
        relay.start()

    def stop_stream(icon, item):
        relay.stop()

    def play_test(icon, item):
        relay.say(relay.cfg.get("test_text") or "测试文本。")

    def quit_app(icon, item):
        icon.stop()
        httpd.shutdown()
        os._exit(0)

    def make_model(name):
        def action(icon, item):
            threading.Thread(target=relay.switch_model, args=(name,), daemon=True).start()
        return action

    def model_checked(name):
        return lambda item: relay.model_name == name

    def model_visible(name, backend_name):
        return lambda item: relay.backend_name == backend_name

    model_items = [
        pystray.MenuItem(name, make_model(name), radio=True,
                         checked=model_checked(name),
                         visible=model_visible(name, bk))
        for bk in relay.backends
        for name in relay.models_for(bk)
    ]

    def make_backend(name):
        def action(icon, item):
            relay.set_backend(name)
        return action

    def backend_checked(name):
        return lambda item: relay.backend_name == name

    backend_items = [
        pystray.MenuItem(name, make_backend(name), radio=True, checked=backend_checked(name))
        for name in relay.backends
    ]

    def make_volume(pct):
        def action(icon, item):
            relay.set_volume(pct / 100.0)
        return action

    def volume_checked(pct):
        return lambda item: abs(relay.volume * 100 - pct) < 5

    volume_items = [
        pystray.MenuItem(f"{pct}%", make_volume(pct), radio=True, checked=volume_checked(pct))
        for pct in range(0, 101, 10)
    ]

    def status_text(*args):
        with relay.lock:
            q = len(relay.texts)
            state = "播放中" if relay.playing else "已暂停"
            return f"状态: {state} | 队列 {q} | 音量 {int(relay.volume * 100)}% | {relay.backend_name}: {relay.model_name}"

    menu = pystray.Menu(
        pystray.MenuItem(status_text, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("开始播放流", start_stream),
        pystray.MenuItem("停止并清空流", stop_stream),
        pystray.MenuItem("播放测试文本", play_test),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("音量", pystray.Menu(*volume_items)),
        pystray.MenuItem("后端", pystray.Menu(*backend_items)),
        pystray.MenuItem("模型", pystray.Menu(*model_items)),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("退出", quit_app),
    )

    icon = pystray.Icon("tts_relay", make_icon_image(), "GPT-SoVITS 中转", menu)
    relay.icon = icon
    icon.run()


if __name__ == "__main__":
    main()
