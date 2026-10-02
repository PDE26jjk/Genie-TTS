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
import sys
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

HOOK_MARKER = os.path.join(os.path.expanduser("~"), ".zcode", "hooks", "tts_off")

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
        # emotion carry-over v3: sliding window of recent utterances
        # (pcm_bytes, text) that get CONCATENATED with the anchor reference
        # into one wav -> a single in-distribution reference pair
        self._carry_utts = []  # oldest first
        self._carry_seq = 0
        self._anchor_pcm = None
        self._anchor_sr = None
        self._utt_end = object()
        self._stream_restart = object()
        # playback prebuffer in seconds; overridden per backend entry
        self._play_prebuffer = float(cfg.get("prebuffer_s", 1.5))

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
        self._carry_utts = []
        self._anchor_pcm = None
        self.backend_name = name
        models = self.models_for(name)
        if self.model_name not in models:
            self.model_name = next(iter(models), None)
        b = self.backend()
        if b.get("type") == "genie":
            try:
                requests.post(b["api_url"] + "/carry",
                              json={"enabled": bool(self.cfg.get("emotion_carry"))},
                              timeout=5)
            except Exception:
                pass
        self._refresh_menu()
        return True

    def set_carry(self, enabled: bool):
        """Runtime emotion-carry toggle: persist the choice, and sync genie
        servers via /carry so no restart is needed (older servers without
        the endpoint are skipped silently)."""
        enabled = bool(enabled)
        self.cfg["emotion_carry"] = enabled
        try:
            cfg = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    cfg = json.load(f)
            cfg["emotion_carry"] = enabled
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("failed to persist emotion_carry:", e, flush=True)
        b = self.backend()
        if b.get("type") == "genie":
            try:
                requests.post(b["api_url"] + "/carry",
                              json={"enabled": enabled}, timeout=5)
                print(f"[relay] carry -> {'on' if enabled else 'off'} "
                      f"({b['api_url']})", flush=True)
            except Exception as e:
                print(f"[relay] carry sync failed ({b['api_url']}):", e, flush=True)
        self._refresh_menu()
        return enabled

    # ---- control surface (called from HTTP threads / tray actions) ----
    def say(self, text, play=None):
        text = (text or "").strip()
        if not text:
            return 0
        # enqueue each piece separately: the gen loop pipelines generation
        # ahead of playback per item, so long replies flow piece by piece
        pieces = self._split_text(text)
        with self.cond:
            for piece in pieces:
                self.texts.append(piece)
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
        self._carry_utts = []
        self._anchor_pcm = None  # anchor is per-voice
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
            # prebuffer: hold audio until this many seconds are queued before
            # the first write of each burst, so chunky backends (CPU mode)
            # don't starve the device between generation chunks
            prebuf_bytes = int(self._play_prebuffer *
                               int(self.cfg["sr"])) * 2
            pending = bytearray()

            def _flush():
                if pending:
                    data = np.frombuffer(bytes(pending), dtype="<i2")
                    if self.volume != 1.0:
                        data = np.clip(data.astype(np.float32) * self.volume,
                                       -32767, 32767).astype("<i2")
                    stream.write(data)
                    pending.clear()

            try:
                while True:
                    item = self.audio_q.get()
                    if item is self._play_quit:
                        return
                    if item is self._stream_restart:
                        reopen = True
                        break
                    if item is self._utt_end or item is None:
                        _flush()
                        continue
                    pending.extend(item)
                    if len(pending) >= prebuf_bytes:
                        _flush()
            except Exception:
                # stream was aborted underneath us; reopen a fresh one
                reopen = True
            finally:
                self.current_stream = None
                try:
                    stream.close()
                except Exception:
                    pass
            # loop: reopen a fresh stream and keep serving the queue

    def _split_text(self, text: str) -> list:
        limit = int(self.backend().get("max_chunk_chars", 90))
        """The t2s context fits roughly 90 Chinese chars per utterance
        (GPU static KV=800 minus reference/history; CPU dynamic loop is
        500 steps) - longer text gets force-truncated mid-sentence by the
        model, so pack sentences into chunks below the limit."""
        import re
        text = text.strip()
        if len(text) <= limit:
            return [text]
        pieces = []
        for sent in re.split(r"(?<=[。！？；!?…])", text):
            sent = sent.strip()
            if not sent:
                continue
            while len(sent) > limit:
                cut = sent.rfind("，", 0, limit + 1)
                if cut < limit // 2:
                    cut = limit
                pieces.append(sent[:cut])
                sent = sent[cut:].lstrip("，、 ")
            pieces.append(sent)
        chunks, cur = [], ""
        for piece in pieces:
            if cur and len(cur) + len(piece) > limit:
                chunks.append(cur)
                cur = piece
            else:
                cur = (cur + piece) if cur else piece
        if cur:
            chunks.append(cur)
        return chunks

    def _generate(self, text):
        # a freshly popped item always generates: the abort flag belongs to
        # stop/switch actions on IN-FLIGHT generation, and leaving it set
        # (backend switch -> stop()) would silently swallow new items
        self.abort.clear()
        for chunk in self._split_text(text):
            if self.abort.is_set():
                print("[relay] generate aborted mid-text", flush=True)
                return
            self._generate_one(chunk)

    def _generate_one(self, text):
        self.abort.clear()
        b = self.backend()
        # per-backend prebuffer: fast backends start almost immediately,
        # near-realtime ones (CPU) need a cushion against chunk jitter
        self._play_prebuffer = float(b.get("prebuffer_s",
                                           self.cfg.get("prebuffer_s", 1.5)))
        # emotion carry: gptsovits uses wav-concat references (v3); the genie
        # backend uses the server-side token carry, so only gptsovits turns
        # are captured into the relay's sliding window
        capture = (bool(self.cfg.get("emotion_carry"))
                   and b.get("type") == "gptsovits")
        captured = bytearray() if capture else None
        # emotion carry v3, shared by both backends: once the sliding window
        # has content, generate against concat(anchor, recent utterances)
        ref_override = None
        if capture:
            # generating a sentence that is already in the reference makes
            # the model echo it (repeats/mismatch) - fall back to anchor
            norm = text.strip()
            if any(u[1].strip() == norm for u in self._carry_utts):
                print("[relay] carry skipped: current text duplicates a window entry", flush=True)
            else:
                concat_path, concat_text = self._build_concat_ref()
                if concat_path is not None:
                    ref_override = (concat_path, concat_text)
                    print(f"[relay] concat ref: {concat_text[:40]}...", flush=True)
        if b.get("type") == "genie":
            entry = (self.cfg.get("genie_characters") or {}).get(self.model_name, {})
            # token carry: the server keeps the anchor reference and appends
            # recent utterances as semantic tokens itself (GENIE_EMOTION_CARRY=1)
            # - no per-utterance reference push, no HuBERT/SV re-extraction
            payload = {
                "character_name": entry.get("character_name", self.model_name),
                "text": text,
                # one prefill per utterance: with incremental vocoder decoding
                # there is no prefix blow-up left, and sentence boundaries
                # would only insert extra prefill stalls into playback
                "split_sentence": False,
            }
        if b.get("type") == "gptsovits":
            anchor_ref, _ = self._ref_pair()
            ref_audio, prompt_text = self._ref_pair()
            if ref_override:
                ref_audio, prompt_text = ref_override
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

    # ---- emotion carry v3: concat(anchor, recent utterances) as one ref ----

    CARRY_ANCHOR_MAX = 10.0   # cap the anchor segment
    CARRY_UTT_MAX = 10.0      # cap the summed recent utterances
    CARRY_TOTAL_MAX = 14.0    # cap the whole concatenated reference

    def _load_anchor_pcm(self):
        """Lazy-load the per-model anchor wav as (s16 mono pcm @ cfg sr, text)."""
        model_key = f"{self.backend_name}:{self.model_name}"
        if self._anchor_pcm is not None and getattr(self, "_anchor_key", None) == model_key:
            return self._anchor_pcm
        import soundfile as sf
        import soxr
        ref_path, ref_text = self._ref_pair()
        data, sr = sf.read(ref_path, dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        if sr != int(self.cfg["sr"]):
            data = soxr.resample(data, sr, int(self.cfg["sr"]))
        pcm = (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()
        dur = len(pcm) / 2 / int(self.cfg["sr"])
        if dur > self.CARRY_ANCHOR_MAX:  # keep the anchor's tail
            pcm = pcm[-int(self.CARRY_ANCHOR_MAX * int(self.cfg["sr"])) * 2:]
        self._anchor_pcm = pcm
        self._anchor_key = model_key
        self._anchor_text = ref_text
        return pcm

    def _build_concat_ref(self):
        """Concatenate anchor + recent utterances into one wav (unique name per
        turn: both backends cache reference features by path string). Returns
        (path, transcript) or (None, None) when there is nothing to carry."""
        import wave
        anchor = self._load_anchor_pcm()
        utts = list(self._carry_utts)
        # duration-based trim of the window (newest kept)
        def utt_dur(u):
            return len(u[0]) / 2 / int(self.cfg["sr"])
        while utts and sum(utt_dur(u) for u in utts) > self.CARRY_UTT_MAX:
            utts.pop(0)
        total = (len(anchor) / 2 / int(self.cfg["sr"])) + sum(utt_dur(u) for u in utts)
        while utts and total > self.CARRY_TOTAL_MAX:
            total -= utt_dur(utts[0])
            utts.pop(0)
        if not utts:
            return None, None
        self._carry_utts = utts

        pieces = [anchor] + [u[0] for u in utts]
        text_parts = []
        for t in [self._anchor_text] + [u[1] for u in utts]:
            t = (t or "").strip()
            if not t:
                continue
            if t[-1] not in "。！？!?…；;":
                t += "。"
            text_parts.append(t)
        transcript = "".join(text_parts)

        self._carry_seq += 1
        path = os.path.join(BASE_DIR, f"carry_{self._carry_seq % 50:03d}.wav")
        try:
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(int(self.cfg["sr"]))
                w.writeframes(b"".join(pieces))
            old = getattr(self, "_last_concat_path", None)
            self._last_concat_path = path
            if old and old != path and os.path.exists(old):
                try:
                    os.remove(old)
                except OSError:
                    pass
            dur = len(b"".join(pieces)) / 2 / int(self.cfg["sr"])
            print(f"[relay] concat ref: {len(utts)} utt(s), {dur:.1f}s total", flush=True)
            return path, transcript
        except Exception as e:
            print("concat ref failed:", e, flush=True)
            return None, None

    def _store_carry(self, pcm: bytes, text: str):
        """Append a completed utterance to the sliding carry window."""
        dur = len(pcm) / 2 / int(self.cfg["sr"])
        if not (1.0 <= dur <= 12.0):
            print(f"[relay] carry: utterance {dur:.1f}s out of 1-12s, ignored", flush=True)
            return
        # echo guard: audio far longer than the text implies means the model
        # repeated itself - never let that into the reference chain
        expected = max(len(text.strip()) / 5.0, 1.0)
        if dur > expected * 1.8 + 1.0:
            print(f"[relay] carry: suspected repetition ({dur:.1f}s vs ~{expected:.1f}s "
                  f"expected), not carried", flush=True)
            return
        # duplicate guard: identical text already in the window
        norm = text.strip()
        if any(u[1].strip() == norm for u in self._carry_utts):
            print("[relay] carry: duplicate text, not carried", flush=True)
            return
        self._carry_utts.append((pcm, text))
        total = sum(len(u[0]) / 2 / int(self.cfg["sr"]) for u in self._carry_utts)
        while self._carry_utts and total > self.CARRY_UTT_MAX:
            total -= len(self._carry_utts[0][0]) / 2 / int(self.cfg["sr"])
            self._carry_utts.pop(0)
        print(f"[relay] carry window: {len(self._carry_utts)} utt(s), {total:.1f}s", flush=True)

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

    # Exclusive bind: http.server sets SO_REUSEADDR, which on Windows
    # silently allows a second instance to double-bind the same port -
    # two relays then both play audio and split the queue. Fail loudly.
    class ExclusiveServer(ThreadingHTTPServer):
        allow_reuse_address = False

    try:
        httpd = ExclusiveServer((cfg["listen_host"], int(cfg["listen_port"])), Handler)
    except OSError as e:
        print(f"another relay instance is probably running on port "
              f"{cfg['listen_port']} ({e}); exiting.", flush=True)
        sys.exit(1)
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

    def toggle_carry(icon, item):
        relay.set_carry(not relay.cfg.get("emotion_carry"))

    def toggle_hook(icon, item):
        # the ZCode Stop-hook stays silent while the marker file exists
        if os.path.exists(HOOK_MARKER):
            os.remove(HOOK_MARKER)
        else:
            os.makedirs(os.path.dirname(HOOK_MARKER), exist_ok=True)
            open(HOOK_MARKER, "w").close()

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
        pystray.MenuItem("朗读ZCode回复", toggle_hook,
                         checked=lambda item: not os.path.exists(HOOK_MARKER)),
        pystray.MenuItem("拼接参考(情绪延续)", toggle_carry,
                         checked=lambda item: bool(relay.cfg.get("emotion_carry"))),
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
