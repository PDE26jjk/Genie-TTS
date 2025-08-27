import os

# (Optional) We recommend manually specifying the Hubert path for Genie.
# Download from Huggingface: https://huggingface.co/High-Logic/Genie
# Note: If this line is not set, Genie will automatically download the model from Huggingface.
os.environ['HUBERT_MODEL_PATH'] = r"C:\path\to\chinese-hubert-base.onnx"

# (Optional) We recommend manually specifying the dictionary path for pyopenjtalk.
# Download from Huggingface: https://huggingface.co/High-Logic/Genie
# Note: If this line is not set, pyopenjtalk will automatically download the dictionary.
os.environ['OPEN_JTALK_DICT_DIR'] = r"C:\path\to\open_jtalk_dic_utf_8-1.11"

import genie_tts as genie

# Launch the Genie command-line client
genie.launch_command_line_client()
