"""
使用 genie.start_server(host=SERVER_HOST, port=SERVER_PORT, workers=1) 开启 API Server。


Genie TTS 服务器 API 快速参考

1. 加载角色模型
接口: POST /load_character
功能: 将一个角色模型加载到服务器。
请求参数 (JSON):
    - character_name (string): 角色的唯一名称。
    - onnx_model_dir (string): 服务器上模型文件夹的路径。

2. 设置参考音频
接口: POST /set_reference_audio
功能: 为已加载的角色指定声音克隆所需的音频。
请求参数 (JSON):
    - character_name (string): 要设置的角色名称。
    - audio_path (string): 服务器上参考音频文件的路径。
    - audio_text (string): 参考音频对应的文本。

3. 文本转语音 (TTS)
接口: POST /tts
功能: 生成语音，并以 audio/wav 音频流返回。
请求参数 (JSON):
    - character_name (string): 使用的角色名称。
    - text (string): 要转换为语音的文本。
    - split_sentence (boolean, 可选): 是否自动切分句子，默认为 false。
    - save_path (string, 可选): 在服务器上保存音频的完整路径。

4. 卸载角色模型
接口: POST /unload_character
功能: 从服务器内存中移除一个角色以释放资源。
请求参数 (JSON):
    - character_name (string): 要卸载的角色名称。

5. 停止所有TTS任务
接口: POST /stop
功能: 立即停止当前所有正在进行的语音合成任务。
请求参数: 无。

6. 清除参考音频缓存
接口: POST /clear_reference_audio_cache
功能: 清除服务器中已加载的参考音频缓存。
请求参数: 无。


"""

import os

os.environ['HUBERT_MODEL_PATH'] = r"C:\Users\Haruka\Desktop\Midori\Data\common_resource\tts\chinese-hubert-base.onnx"
os.environ['Max_Cached_Character_Models'] = '3'
os.environ['Max_Cached_Reference_Audio'] = '10'

import time
import requests
import pyaudio
import multiprocessing

import genie_tts as genie

# --- 配置区 ---
# 服务器地址
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

BYTES_PER_SAMPLE = 2
CHANNELS = 1
SAMPLE_RATE = 32000


def run_server():
    genie.start_server(host=SERVER_HOST, port=SERVER_PORT, workers=1)


def main_client():
    # 1. 加载角色
    print("\n[客户端] 步骤 1: 发送加载角色请求...")
    load_payload = {
        "character_name": "irene",
        "onnx_model_dir": r"C:\Users\Haruka\Desktop\GENIE\GENIE_CPU_RUNTIME_OLD\Docs\Output"
    }
    try:
        response = requests.post(f"{BASE_URL}/load_character", json=load_payload)
        response.raise_for_status()
        print(f"[客户端] 加载角色成功: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        print(f"[客户端] 加载角色失败: {e}")
        return

    # 2. 设置参考音频
    print("\n[客户端] 步骤 2: 发送设置参考音频请求...")
    ref_audio_payload = {
        "character_name": "irene",
        "audio_path": r"C:\Users\Haruka\Desktop\Midori\Data\character_resource\irene\tts\prompt_wav\6924128.wav",
        "audio_text": "みんながもっといい生活を送れるなら 原力の存在も我慢できますね"
    }
    try:
        response = requests.post(f"{BASE_URL}/set_reference_audio", json=ref_audio_payload)
        response.raise_for_status()
        print(f"[客户端] 设置参考音频成功: {response.json()['message']}")
    except requests.exceptions.RequestException as e:
        print(f"[客户端] 设置参考音频失败: {e}")
        return

    # 3. 请求TTS并流式播放
    print("\n[客户端] 步骤 3: 请求TTS并准备流式播放...")
    tts_payload = {
        "character_name": "irene",
        "text": "公園に行って、散歩をしながら花を見ました。",
        "split_sentence": True
    }

    p = pyaudio.PyAudio()
    stream = None

    try:
        with requests.post(f"{BASE_URL}/tts", json=tts_payload, stream=True) as response:
            response.raise_for_status()
            print("[客户端] 已连接到音频流，开始播放...")

            # 迭代接收到的音频数据块
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    if stream is None:
                        stream = p.open(format=p.get_format_from_width(BYTES_PER_SAMPLE),
                                        channels=CHANNELS,
                                        rate=SAMPLE_RATE,
                                        output=True)
                    stream.write(chunk)

            print("[客户端] 音频流接收完毕。")

    except requests.exceptions.RequestException as e:
        print(f"[客户端] TTS请求失败: {e}")
    except Exception as e:
        print(f"[客户端] 播放时发生错误: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        p.terminate()


if __name__ == "__main__":
    # 创建并启动服务器进程
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()

    # 给予服务器一点启动时间
    time.sleep(3)

    # 运行客户端逻辑
    try:
        main_client()
    finally:
        print("\n[主进程] 测试完成，正在关闭服务器...")
        server_process.terminate()
        server_process.join()
        print("[主进程] 服务器已关闭。")
