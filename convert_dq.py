# -*- coding: utf-8 -*-
# Convert the user's fine-tuned GPT-SoVITS v2Pro checkpoints to ONNX.
import genie_tts as genie

genie.convert_to_onnx(
    torch_ckpt_path=r"R:\git\dsh_projects\cyber_persona\GPT-SoVITS-v2pro-20250604\GPT_weights_v2Pro\dq-e15.ckpt",
    torch_pth_path=r"R:\git\dsh_projects\cyber_persona\GPT-SoVITS-v2pro-20250604\SoVITS_weights_v2Pro\dq_e8_s176.pth",
    output_dir=r"R:\git\dsh_projects\cyber_persona\Genie-TTS\models_dq_v2pro",
    version="v2Pro",
)
