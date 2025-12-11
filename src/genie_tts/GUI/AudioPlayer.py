import sounddevice as sd
import threading
import queue
from typing import Union, Optional, Callable
import numpy as np


def run_in_sub_thread(func) -> Callable[..., threading.Thread]:
    def wrapper(*args, **kwargs) -> threading.Thread:
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread

    return wrapper


class AudioPlayer:
    CHUNK_SIZE: int = 1024

    def __init__(self):
        self._task_queue: queue.Queue[bytes | str] = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._start_worker()

    def _start_worker(self):
        """启动工作线程（如果它尚未运行或已关闭）。"""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = self._playback_worker()

    @run_in_sub_thread
    def _playback_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                task: bytes = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            stream: Optional[sd.OutputStream] = None
            try:
                stream = sd.OutputStream(
                    samplerate=32000,
                    channels=1,
                    dtype='float32'  # 我们将处理好的浮点数据送入流
                )
                stream.start()
                offset: int = 0
                while offset < len(task) and not self._stop_event.is_set():
                    chunk = task[offset:offset + self.CHUNK_SIZE]
                    stream.write(chunk)
                    offset += self.CHUNK_SIZE

            except Exception as e:
                print(f"播放时发生错误: {e}")
            finally:
                if stream:
                    stream.stop()
                    stream.close()
                self._task_queue.task_done()

    def play(self, source: Union[str, np.ndarray]):
        """将音频源加入播放队列。"""
        self._start_worker()
        self._task_queue.put(source)

    def stop(self):
        """停止播放并清空播放队列。"""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join()
        self._stop_event.clear()

        with self._task_queue.mutex:
            self._task_queue.queue.clear()
            while self._task_queue.unfinished_tasks > 0:
                self._task_queue.task_done()

    def wait(self):
        """阻塞，直到队列中所有任务都播放完成。"""
        self._task_queue.join()

    def close(self):
        """永久关闭播放器并释放资源。"""
        self.stop()
