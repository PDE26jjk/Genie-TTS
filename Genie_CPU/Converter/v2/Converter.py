from .VITSConverter import VITSConverter
from .T2SConverter import T2SModelConverter
from .EncoderConverter import EncoderConverter

import logging
from typing import Optional, Tuple
import os
import shutil
import traceback
import importlib.resources
import contextlib

logger = logging.getLogger()

PACKAGE_NAME = "Genie_CPU"
CACHE_DIR = os.path.join(os.getcwd(), "Cache")
DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), "Output")
_ENCODER_RESOURCE_PATH = "Data/v2/Models/t2s_encoder_fp32.onnx"
_STAGE_DECODER_RESOURCE_PATH = "Data/v2/Models/t2s_stage_decoder_fp32.onnx"
_FIRST_STAGE_DECODER_RESOURCE_PATH = "Data/v2/Models/t2s_first_stage_decoder_fp32.onnx"
_VITS_RESOURCE_PATH = "Data/v2/Models/vits_fp32.onnx"
_T2S_KEYS_RESOURCE_PATH = "Data/v2/Keys/t2s_onnx_keys.txt"
_VITS_KEYS_RESOURCE_PATH = "Data/v2/Keys/vits_onnx_keys.txt"


def find_ckpt_and_pth(directory: str) -> Tuple[Optional[str], Optional[str]]:
    ckpt_path: Optional[str] = None
    pth_path: Optional[str] = None
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
            logger.info(f"🧹 已清理文件夹: {folder}")
    except Exception as e:
        logger.error(f'❌ 清理文件夹 {folder} 失败: {e}')


def convert(torch_model_path: str, output_base_dir: Optional[str] = None):
    """
    转换模型。

    Args:
        torch_model_path (str): 包含 .ckpt 和 .pth 文件的源模型文件夹路径。
        output_base_dir (str, optional): 用于存放所有转换结果的根目录。
                                         如果为 None, 默认输出到当前工作目录下的 'Output' 文件夹。
    """
    # 如果用户没有提供输出目录，则使用默认值
    if output_base_dir is None:
        output_base_dir = DEFAULT_OUTPUT_DIR

    character_name: str = os.path.basename(torch_model_path)
    output_dir: str = os.path.join(output_base_dir, character_name)

    # 确保缓存和输出目录存在
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if len(os.listdir(output_dir)) > 0:
        logger.warning(f'输出文件夹 {output_dir} 非空，将覆盖内容。')

    torch_ckpt_path, torch_pth_path = find_ckpt_and_pth(torch_model_path)

    if not torch_ckpt_path or not torch_pth_path:
        logger.error(f'无法处理文件夹 {torch_model_path} 。请保证文件夹内有 GPT—SOVITS V2 导出的 .pth 和 .ckpt 模型。')
        return

    logger.info(f'正在处理 {torch_model_path} 。')

    try:
        with contextlib.ExitStack() as stack:
            files = importlib.resources.files(PACKAGE_NAME)

            encoder_onnx_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_ENCODER_RESOURCE_PATH)))
            stage_decoder_path = stack.enter_context(
                importlib.resources.as_file(files.joinpath(_STAGE_DECODER_RESOURCE_PATH)))
            first_stage_decoder_path = stack.enter_context(
                importlib.resources.as_file(files.joinpath(_FIRST_STAGE_DECODER_RESOURCE_PATH)))
            vits_onnx_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_VITS_RESOURCE_PATH)))
            t2s_keys_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_T2S_KEYS_RESOURCE_PATH)))
            vits_keys_path = stack.enter_context(importlib.resources.as_file(files.joinpath(_VITS_KEYS_RESOURCE_PATH)))

            converter_1 = T2SModelConverter(
                torch_ckpt_path=torch_ckpt_path,
                stage_decoder_onnx_path=str(stage_decoder_path),
                first_stage_decoder_onnx_path=str(first_stage_decoder_path),
                key_list_file=str(t2s_keys_path),
                output_dir=output_dir,
                cache_dir=CACHE_DIR,
            )
            converter_2 = VITSConverter(
                torch_pth_path=torch_pth_path,
                vits_onnx_path=str(vits_onnx_path),
                key_list_file=str(vits_keys_path),
                output_dir=output_dir,
                cache_dir=CACHE_DIR,
            )
            converter_3 = EncoderConverter(
                ckpt_path=torch_ckpt_path,
                pth_path=torch_pth_path,
                onnx_input_path=str(encoder_onnx_path),
                output_dir=output_dir,
            )

            try:
                converter_1.run_full_process()
                converter_2.run_full_process()
                converter_3.convert()
                logger.info(f"🎉 转换成功，已保存至: {output_dir}\n")
            except Exception:
                logger.error(f"❌ 转换过程中发生严重错误")
                logger.error(traceback.format_exc())
                remove_folder(output_dir)  # 只在失败时清理输出目录

    finally:
        # 无论成功还是失败，都尝试清理缓存目录
        remove_folder(CACHE_DIR)
