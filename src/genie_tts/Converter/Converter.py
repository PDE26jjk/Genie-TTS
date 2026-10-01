from .v2.Converter import convert as convert_v2
from .v2Pro.Converter import convert as convert_v2pro
from .v2ProPlus.Converter import convert as convert_v2pp

import os

SUPPORTED_VERSIONS = ("auto", "v2", "v2Pro", "v2ProPlus")


def convert(torch_ckpt_path: str, torch_pth_path: str, output_dir: str, version: str = "auto") -> None:
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported version: {version!r}. Supported: {SUPPORTED_VERSIONS}")

    if version == "v2Pro":
        convert_v2pro(torch_ckpt_path, torch_pth_path, output_dir)
    elif version == "v2ProPlus":
        convert_v2pp(torch_ckpt_path, torch_pth_path, output_dir)
    elif version == "v2":
        convert_v2(torch_ckpt_path, torch_pth_path, output_dir)
    elif os.path.getsize(torch_pth_path) > 150 * 1024 * 1024:  # 大于 150 MB
        convert_v2pp(torch_ckpt_path, torch_pth_path, output_dir)
    else:
        convert_v2(torch_ckpt_path, torch_pth_path, output_dir)
