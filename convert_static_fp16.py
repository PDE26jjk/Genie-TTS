# Convert the static t2s decode graph and the first-stage decoder to fp16.
#
# Both fp32 weights and all captured activations halve, which is the main
# lever for the server's VRAM footprint (the CUDA-graph capture pool keeps
# every intermediate alive). Run with the CHARM python (needs
# onnxconverter-common):
#
#   CHARM\.venv\Scripts\python.exe convert_static_fp16.py --model-dir models_dq_v2pro
#
# Outputs <name>_fp16.onnx next to the originals; the originals stay
# untouched so a bad conversion can be rolled back by deleting the files.
import argparse
import os

import onnx
from onnx.external_data_helper import convert_model_from_external_data
from onnxconverter_common import float16


def convert(src: str, dst: str) -> None:
    print(f"loading {src} ...")
    model = onnx.load(src)  # pulls external data into memory
    # embed any external weights so the fp16 file is self-contained
    try:
        convert_model_from_external_data(model)
    except Exception as e:
        print(f"  (no external data to embed: {e})")
    print("converting to fp16 ...")
    fp16_model = float16.convert_float_to_float16(model)
    onnx.save(fp16_model, dst)
    print(f"saved {dst} ({os.path.getsize(dst) / 1048576:.0f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()

    jobs = [
        ("t2s_static.onnx", "t2s_static_fp16.onnx"),
        ("t2s_first_stage_decoder_fp32.onnx", "t2s_first_stage_decoder_fp16.onnx"),
    ]
    for src_name, dst_name in jobs:
        src = os.path.join(args.model_dir, src_name)
        dst = os.path.join(args.model_dir, dst_name)
        if not os.path.exists(src):
            print(f"skip (missing): {src}")
            continue
        if os.path.exists(dst):
            print(f"skip (exists): {dst}")
            continue
        convert(src, dst)


if __name__ == "__main__":
    main()
