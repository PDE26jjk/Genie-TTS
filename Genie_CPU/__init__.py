import logging
from rich.logging import RichHandler
import onnxruntime

onnxruntime.set_default_logger_severity(3)

# 以防万一写在导入库之前。
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

import os
from os import PathLike

from .ModelManager import model_manager


def load_onnx(
        character_name: str,
        model_path: str | PathLike,
):
    model_path: str = os.fspath(model_path)
    model_manager.load_character(
        character_name=character_name,
        model_dir=model_path,
    )


def convert_to_onnx(
        character_name: str,
        model_path: str | PathLike,
):
    try:
        import torch
    except ImportError as e:
        logger.error("❌ PyTorch 未安装，请先执行 `pip install torch`")
        return
