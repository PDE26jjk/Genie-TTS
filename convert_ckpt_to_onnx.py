# -*- coding: utf-8 -*-
# Convert a fine-tuned GPT-SoVITS checkpoint pair (GPT .ckpt + SoVITS .pth)
# into the ONNX component layout Genie loads at runtime:
#   t2s_encoder / t2s_first_stage_decoder / t2s_stage_decoder (+ fp16 weight
#   bins) / vits / prompt_encoder - per architecture version (v2, v2Pro,
#   v2ProPlus).
#
# Usage:
#   python convert_ckpt_to_onnx.py <gpt_ckpt> <sovits_pth> <output_dir> [version]
#
# These are the DYNAMIC components. For the GPU static CUDA-graph decode,
# run export_static_t2s.py and convert_static_fp16.py afterwards (see
# README, "Voice model graphs").
import sys

import genie_tts as genie

if len(sys.argv) not in (4, 5):
    print(__doc__)
    sys.exit(2)

ckpt, pth, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
version = sys.argv[4] if len(sys.argv) == 5 else "v2Pro"

genie.convert_to_onnx(
    torch_ckpt_path=ckpt,
    torch_pth_path=pth,
    output_dir=out_dir,
    version=version,
)
