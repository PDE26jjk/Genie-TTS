import queue
import asyncio
import threading
import time

import numpy as np
import wave
from typing import Optional, List
import pyaudio  # <-- 已替换
import logging

from ..Japanese.Split import split_japanese_text
from ..Core.Inference import tts_client
from ..ModelManager import model_manager
from ..Utils.Shared import context
from ..Utils.Utils import clear_queue

logger = logging.getLogger(__name__)

STREAM_END = 'STREAM_END'  # 这是一个特殊的标记，表示文本流结束


class TTSPlayer:
    def __init__(self, sample_rate: int = 32000):
        self.sample_rate: int = sample_rate
        self.channels: int = 1
        self.bytes_per_sample: int = 2  # 16-bit audio

        self._text_queue: queue.Queue = queue.Queue()
        self._audio_queue: queue.Queue = queue.Queue()

        self._stop_event: threading.Event = threading.Event()
        self._tts_done_event: threading.Event = threading.Event()
        self._api_lock: threading.Lock = threading.Lock()

        self._tts_worker: Optional[threading.Thread] = None
        self._playback_worker: Optional[threading.Thread] = None

        self._play: bool = False
        self._current_save_path: Optional[str] = None
        self._session_audio_chunks: List[np.ndarray] = []
        self._stream_queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    @staticmethod
    def _preprocess_for_playback(audio_float: np.ndarray) -> bytes:
        audio_int16 = (audio_float.squeeze() * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _tts_worker_loop(self):
        """从文本队列取句子，生成音频，放入音频队列。"""
        while not self._stop_event.is_set():
            try:
                sentence = self._text_queue.get(timeout=1)
                if sentence is None or self._stop_event.is_set():
                    break
            except queue.Empty:
                continue

            try:
                if sentence is STREAM_END:
                    if self._current_save_path and self._session_audio_chunks:
                        self._save_session_audio()
                    self._tts_done_event.set()
                    continue

                gsv_model = model_manager.get(context.current_speaker)
                if not gsv_model or not context.current_prompt_audio:
                    logger.error("Missing model or reference audio.")
                    continue

                tts_client.stop_event.clear()
                audio_chunk = tts_client.tts(
                    text=sentence,
                    prompt_audio=context.current_prompt_audio,
                    encoder=gsv_model.T2S_ENCODER,
                    first_stage_decoder=gsv_model.T2S_FIRST_STAGE_DECODER,
                    stage_decoder=gsv_model.T2S_STAGE_DECODER,
                    vocoder=gsv_model.VITS,
                )

                if audio_chunk is not None:
                    if self._end_time is None:
                        self._end_time = time.time()
                        duration: float = self._end_time - self._start_time
                        logger.info(f"First packet latency: {duration:.3f} seconds.")
                    if self._play:
                        self._audio_queue.put(audio_chunk)
                    if self._current_save_path:
                        self._session_audio_chunks.append(audio_chunk)
                    if self._stream_queue and self._loop:
                        audio_data = self._preprocess_for_playback(audio_chunk)
                        self._loop.call_soon_threadsafe(self._stream_queue.put_nowait, audio_data)
            except Exception as e:
                logger.error(f"A critical error occurred while processing the TTS task: {e}", exc_info=True)
                self._tts_done_event.set()

    def _playback_worker_loop(self):
        p = None
        stream = None
        try:
            p = pyaudio.PyAudio()

            while not self._stop_event.is_set():
                try:
                    # 阻塞式等待音频块。
                    audio_chunk = self._audio_queue.get(timeout=1)
                    if audio_chunk is None:  # 收到“毒丸”，准备退出线程
                        break

                    # 如果流尚未打开，则在收到第一个音频块时创建它。
                    if stream is None:
                        stream = p.open(format=p.get_format_from_width(self.bytes_per_sample),
                                        channels=self.channels,
                                        rate=self.sample_rate,
                                        output=True)

                    # 将处理好的音频字节写入流。这是一个阻塞操作。
                    audio_data = self._preprocess_for_playback(audio_chunk)
                    stream.write(audio_data)

                except queue.Empty:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                        stream = None
                    continue
                except Exception as e:
                    logger.error(f"A critical error occurred while playing audio: {e}", exc_info=True)
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                        stream = None
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            if p is not None:
                p.terminate()

    def _save_session_audio(self):
        try:
            full_audio = np.concatenate(self._session_audio_chunks, axis=0)
            with wave.open(self._current_save_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.bytes_per_sample)
                wf.setframerate(self.sample_rate)
                wf.writeframes(self._preprocess_for_playback(full_audio))
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
        finally:
            self._session_audio_chunks = []
            self._current_save_path = None

    def start_session(self,
                      play: bool = False,
                      save_path: Optional[str] = None,
                      stream_queue: Optional[asyncio.Queue] = None
                      ):
        with self._api_lock:
            self._tts_done_event.clear()

            if stream_queue:
                try:
                    self._loop = asyncio.get_running_loop()
                    self._stream_queue = stream_queue
                except RuntimeError:
                    logger.warning(
                        "The `start_session` contains `stream_queue` but is not called within an asyncio event loop. Streaming will be ignored."
                    )
                    self._loop = None

            self._stop_event.clear()

            if self._tts_worker is None or not self._tts_worker.is_alive():
                self._tts_worker = threading.Thread(target=self._tts_worker_loop, daemon=True)
                self._tts_worker.start()

            if self._playback_worker is None or not self._playback_worker.is_alive():
                self._playback_worker = threading.Thread(target=self._playback_worker_loop, daemon=True)
                self._playback_worker.start()

            clear_queue(self._text_queue)
            clear_queue(self._audio_queue)

            self._play = play
            self._current_save_path = save_path
            self._session_audio_chunks = []
            self._start_time = None
            self._end_time = None

    def feed(self, text_chunk: str):
        with self._api_lock:
            if not text_chunk:
                return
            if self._start_time is None:
                self._start_time = time.time()

            sentences = split_japanese_text(text_chunk.strip())
            for sentence in sentences:
                self._text_queue.put(sentence)

    def end_session(self):
        with self._api_lock:
            self._text_queue.put(STREAM_END)

            if self._stream_queue and self._loop:
                self._loop.call_soon_threadsafe(self._stream_queue.put_nowait, None)

    def stop(self):
        with self._api_lock:
            if self._tts_worker is None and self._playback_worker is None:
                return
            if self._stop_event.is_set():
                return

            tts_client.stop_event.set()
            self._stop_event.set()
            self._tts_done_event.set()

            # 发送毒丸唤醒可能在 get() 中阻塞的线程
            self._text_queue.put(None)
            self._audio_queue.put(None)

            if self._tts_worker and self._tts_worker.is_alive():
                self._tts_worker.join()
            if self._playback_worker and self._playback_worker.is_alive():
                self._playback_worker.join()

            # 线程结束后，重置它们的状态，以便下次可以重新启动
            self._tts_worker = None
            self._playback_worker = None

    def wait_for_tts_completion(self):
        if self._tts_done_event.is_set():
            return
        self._tts_done_event.wait()


tts_player: TTSPlayer = TTSPlayer()
