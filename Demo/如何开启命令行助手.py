import os

os.environ['HUBERT_MODEL_PATH'] = r"C:\Users\Haruka\Desktop\GENIE\GENIE_CPU_RUNTIME_OLD\Data\chinese-hubert-base.onnx"
os.environ['Max_Cached_Character_Models'] = '3'
os.environ['Max_Cached_Reference_Audio'] = '10'
import genie_tts as genie

genie.launch_command_line_client()
