import os

# （可选）我们推荐给 Genie 手动指定 Hubert 路径，您可以在 https://huggingface.co/High-Logic/Genie 下载模型。
# 注：如果不写这行，Genie 会自动从 Huggingface 下载模型。
os.environ['HUBERT_MODEL_PATH'] = r"C:\Users\Haruka\Desktop\Midori\Data\common_resource\tts\chinese-hubert-base.onnx"

# （可选）您可以设置缓存模型与参考音频的数量。
os.environ['Max_Cached_Character_Models'] = '3'
os.environ['Max_Cached_Reference_Audio'] = '10'

# 一定要先编辑环境变量，再导入 Genie。
import genie_tts as genie
import time

genie.load_character(
    character_name='irene',
    # ↑ 指定一个角色名作为标识。
    onnx_model_dir=r"C:\Users\Haruka\Desktop\GENIE\GENIE_CPU_RUNTIME_OLD\Docs\Output",
)
genie.set_reference_audio(
    character_name='irene',
    # ↑ 使用刚刚指定的角色名。
    audio_path=r"C:\Users\Haruka\Desktop\Midori\Data\character_resource\irene\tts\prompt_wav\6924128.wav",
    # ↑ 参考音频路径，为减少运行时体积，目前不支持 .mp3。
    audio_text='みんながもっといい生活を送れるなら 原力の存在も我慢できますね',
    # ↑ 参考音频的文本。
)
genie.tts(
    character_name='irene',  # 使用刚刚指定的角色名。
    text='公園に行って、散歩をしながら花を見ました。',  # 要 TTS 的内容。
    play=True,  # 是否直接播放音频。
    split_sentence=True,  # 是否切分句子后进行 TTS。
    save_path='tmp.wav',  # 保存路径。
)

time.sleep(10)  # 设置了 play=True，所以要等待播放完毕。
