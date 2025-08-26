import os

os.environ['HUBERT_MODEL_PATH'] = r"<PATH_TO_HUBERT_ONNX_MODEL>"

import genie_tts as genie

# Launch the Genie command-line client
genie.launch_command_line_client()
