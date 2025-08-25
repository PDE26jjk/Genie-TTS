from .VITSConverter import VITSConverter
from .T2SConverter import T2SModelConverter
from .EncoderConverter import EncoderConverter

import logging
from typing import Optional, Tuple
import os
import shutil
import traceback

logger = logging.getLogger()

ENCODER_ONNX_PATH = "./Data/v2/Models/t2s_encoder_fp32.onnx"
STAGE_DECODER_ONNX_PATH = "./Data/v2/Models/t2s_stage_decoder_fp32.onnx"
FIRST_STAGE_DECODER_ONNX_PATH = "./Data/v2/Models/t2s_first_stage_decoder_fp32.onnx"
VITS_ONNX_PATH = r"./Data/v2/Models/vits_fp32.onnx"
T2S_KEYS_PATH = "./Data/v2/Keys/t2s_onnx_keys.txt"
VITS_KEYS_PATH = "./Data/v2/Keys/vits_onnx_keys.txt"
CACHE_DIR = f"./Cache"
OUTPUT_DIR = f"./Output"


def find_ckpt_and_pth(directory: str) -> Tuple[Optional[str], Optional[str]]:
    ckpt_path: Optional[str] = None
    pth_path: Optional[str] = None

    # 遍历文件夹（不递归子目录）
    for filename in os.listdir(directory):
        full_path: str = os.path.join(directory, filename)
        if filename.endswith(".ckpt") and ckpt_path is None:
            ckpt_path = full_path
        elif filename.endswith(".pth") and pth_path is None:
            pth_path = full_path

        if ckpt_path and pth_path:
            break

    return ckpt_path, pth_path


def remove_folder(folder: str) -> None:
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    except Exception as e:
        print(f'❌ 清理文件夹失败: {e}')


def convert(torch_model_path: str):
    character_name: str = os.path.basename(torch_model_path)
    output_dir: str = os.path.join(OUTPUT_DIR, character_name)

    if os.path.exists(output_dir):
        logger.warning(f'输出文件夹 {output_dir} 已存在，将覆盖内容。')

    torch_ckpt_path, torch_pth_path = find_ckpt_and_pth(torch_model_path)

    if not torch_ckpt_path or not torch_pth_path:
        logger.error(f'无法处理文件夹 {torch_model_path} 。请保证文件夹内有 GPT—SOVITS V2 导出的 .pth 和 .ckpt 模型。')
        return

    logger.info(f'正在处理 {torch_model_path} 。')

    converter_1 = T2SModelConverter(
        torch_ckpt_path=torch_ckpt_path,
        stage_decoder_onnx_path=STAGE_DECODER_ONNX_PATH,
        first_stage_decoder_onnx_path=FIRST_STAGE_DECODER_ONNX_PATH,
        key_list_file=T2S_KEYS_PATH,
        output_dir=output_dir,
        cache_dir=CACHE_DIR,
    )
    converter_2 = VITSConverter(
        torch_pth_path=torch_pth_path,
        vits_onnx_path=VITS_ONNX_PATH,
        key_list_file=VITS_KEYS_PATH,
        output_dir=output_dir,
        cache_dir=CACHE_DIR,
    )
    converter_3 = EncoderConverter(
        ckpt_path=torch_ckpt_path,
        pth_path=torch_pth_path,
        onnx_input_path=ENCODER_ONNX_PATH,
        output_dir=output_dir,
    )

    try:
        converter_1.run_full_process()
    except Exception as e:
        logger.error(f"❌ 转换 .ckpt 文件时失败: {e}")
        logger.error(traceback.format_exc())
        remove_folder(output_dir)
        logger.info(f"🧹 已清理文件夹: {output_dir}\n")
        return

    try:
        converter_2.run_full_process()
    except Exception as e:
        logger.error(f"❌ 转换 .pth 文件时失败: {e}")
        logger.error(traceback.format_exc())
        remove_folder(output_dir)
        logger.info(f"🧹 已清理文件夹: {output_dir}\n")
        return

    try:
        converter_3.convert()
    except Exception as e:
        logger.error(f"❌ 抽取 Encoder 权重时失败: {e}")
        logger.error(traceback.format_exc())
        remove_folder(output_dir)
        logger.info(f"🧹 已清理文件夹: {output_dir}\n")
        return

    logger.info(f"🎉 转换成功，已保存至: {output_dir}\n")
