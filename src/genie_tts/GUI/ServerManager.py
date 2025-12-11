import sys
import os
import time
import subprocess
import requests
from typing import Optional
import threading
import numpy as np

from PySide6.QtCore import Signal, QThread
from .Utils import find_free_port

PORT: int = find_free_port()
API_BASE_URL: str = f"http://127.0.0.1:{PORT}"

current_dir = os.path.dirname(os.path.abspath(__file__))
SERVER_SCRIPT_PATH: str = os.path.join(current_dir, "API Server.py")


class ServerManager(QThread):
    """管理 API Server 进程"""
    server_ready = Signal()

    def __init__(self):
        super().__init__()
        self._process: Optional[subprocess.Popen] = None
        self._is_running: bool = True
        self._output_thread: Optional[threading.Thread] = None

    def run(self):
        print("[SYSTEM] 正在启动 TTS 后端服务...")
        try:
            # 1. 启动子进程 (保持之前的 subprocess.PIPE 修改不变)
            self._process = subprocess.Popen(
                [sys.executable, SERVER_SCRIPT_PATH, "--port", str(PORT)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )

            # 2. 启动输出读取线程 (保持不变)
            self._output_thread = threading.Thread(target=self._read_process_output, daemon=True)
            self._output_thread.start()

            # 3. 循环检查逻辑
            start_time = time.time()
            connected = False  # 标记是否已经连上过

            while self._is_running:
                if not connected:
                    try:
                        requests.get(f"{API_BASE_URL}/docs", timeout=1)
                        self.server_ready.emit()
                        connected = True  # 更新状态，下次循环就不会再发请求了
                    except requests.exceptions.ConnectionError:
                        # 还没连上，忽略，等待下一次循环
                        pass
                    except Exception as e:
                        print(f"[WARN] 连接检查出错: {e}")

                    # 超时提醒逻辑也放在这里
                    if not connected and time.time() - start_time > 30:
                        print("[WARN] 等待服务启动时间较长...")
                        start_time = time.time()

                # 4. 无论是否连接，都要检查进程是否活着
                if self._process.poll() is not None:
                    print("[ERROR] 后端服务进程意外退出！")
                    self._is_running = False
                    break

                time.sleep(1)

        except Exception as e:
            print(f"[ERROR] 启动服务器失败: {e}")

    def _read_process_output(self):
        """实时读取子进程管道并打印，从而触发 LogRedirector"""
        if not self._process:
            return
        # 持续读取直到进程结束
        try:
            for line in iter(self._process.stdout.readline, ''):
                if line:
                    print(line.strip())
        except Exception:
            pass

    def stop(self):
        print("[SYSTEM] 正在停止后台服务线程...")
        self._is_running = False

        if self._process:
            print("[SYSTEM] 正在关闭服务器进程...")
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            print("[SYSTEM] 服务器进程已关闭。")
            self._process = None

        # 安全退出线程
        self.quit()
        self.wait()
        print("[SYSTEM] 后台服务线程已安全退出。")


class InferenceWorker(QThread):
    """执行推理任务的 Worker"""
    finished = Signal(bool, str, object)  # success, message, data

    def __init__(self, request_data: dict, mode: str):
        super().__init__()
        self.req: dict = request_data
        self.mode: str = mode

    def run(self) -> None:
        try:
            if self.mode == 'load_character':
                resp = requests.post(f"{API_BASE_URL}/load_character", json=self.req)
                if resp.status_code == 200:
                    self.finished.emit(True, "角色模型加载成功\n", None)
                else:
                    self.finished.emit(False, f"角色模型加载失败: {resp.text}\n", None)

            elif self.mode == 'set_reference_audio':
                resp = requests.post(f"{API_BASE_URL}/set_reference_audio", json=self.req)
                if resp.status_code == 200:
                    self.finished.emit(True, "参考音频设置成功\n", None)
                else:
                    self.finished.emit(False, f"参考音频设置失败: {resp.text}\n", None)

            elif self.mode == 'tts':
                resp = requests.post(f"{API_BASE_URL}/tts", json=self.req)
                if resp.status_code == 200:
                    try:
                        chunks = []
                        for chunk in resp.iter_content(chunk_size=1024):
                            if chunk:
                                chunks.append(chunk)
                        pcm = b''.join(chunks)
                        return_data = {
                            "sample_rate": 32000,
                            "audio_list": [np.frombuffer(pcm, dtype=np.int16)],
                        }
                        self.finished.emit(True, "推理完成", return_data)
                    except Exception as e:
                        self.finished.emit(False, f"数据解析失败: {e}", None)
                else:
                    self.finished.emit(False, f"推理失败: {resp.text}", None)


        except Exception as e:
            self.finished.emit(False, f"网络请求异常: {str(e)}", None)
