# 请严格遵循导入顺序。
# 1、环境变量。
import os
from os import PathLike

os.environ["HF_HUB_ENABLE_PROGRESS_BAR"] = "1"

# 2、Logging。
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]"
)
logger = logging.getLogger(__name__)

# 3、ONNX。
import onnxruntime

onnxruntime.set_default_logger_severity(3)

# 导入剩余库。

import asyncio
from typing import AsyncIterator, Optional

from .Audio.ReferenceAudio import ReferenceAudio
from .Core.TTSPlayer import tts_player
from .ModelManager import model_manager
from .Utils.Shared import context

# A module-level private dictionary to store reference audio configurations.
_reference_audios: dict[str, dict] = {}
SUPPORTED_AUDIO_EXTS = {'.wav', '.flac', '.ogg', '.aiff', '.aif'}


def load_character(
        character_name: str,
        onnx_model_dir: str | PathLike,
) -> None:
    """
    Loads a character model from an ONNX model directory.

    Args:
        character_name (str): The name to assign to the loaded character.
        onnx_model_dir (str | PathLike): The directory path containing the ONNX model files.
    """
    model_path: str = os.fspath(onnx_model_dir)
    model_manager.load_character(
        character_name=character_name,
        model_dir=model_path,
    )


def unload_character(
        character_name: str,
) -> None:
    """
    Unloads a previously loaded character model to free up resources.

    Args:
        character_name (str): The name of the character to unload.
    """
    model_manager.remove_character(
        character_name=character_name,
    )


def set_reference_audio(
        character_name: str,
        audio_path: str,
        audio_text: str,
) -> None:
    """
    Sets the reference audio for a character to be used for voice cloning.

    This must be called for a character before using 'tts' or 'tts_async'.

    Args:
        character_name (str): The name of the character.
        audio_path (str): The file path to the reference audio (e.g., a WAV file).
        audio_text (str): The transcript of the reference audio.
    """
    # 检查文件后缀是否支持
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_AUDIO_EXTS:
        logger.error(
            f"Audio format '{ext}' is not supported. Only the following formats are supported: {SUPPORTED_AUDIO_EXTS}"
        )
        return

    _reference_audios[character_name] = {
        'audio_path': audio_path,
        'audio_text': audio_text,
    }
    context.current_prompt_audio = ReferenceAudio(
        prompt_wav=audio_path,
        prompt_text=audio_text,
    )


async def tts_async(
        character_name: str,
        text: str,
        play: bool = False,
        split_sentence: bool = False,
        save_path: str | PathLike | None = None,
) -> AsyncIterator[bytes]:
    """
    Asynchronously generates speech from text and yields audio chunks.

    This function returns an async iterator that provides the audio data in
    real-time as it's being generated.

    Args:
        character_name (str): The name of the character to use for synthesis.
        text (str): The text to be synthesized into speech.
        play (bool, optional): If True, plays the audio as it's generated. Defaults to False.
        split_sentence (bool, optional): If True, splits the text into sentences for synthesis. Defaults to False.
        save_path (str | PathLike | None, optional): If provided, saves the generated audio to this file path. Defaults to None.

    Yields:
        bytes: A chunk of the generated audio data.

    Raises:
        ValueError: If 'set_reference_audio' has not been called for the character.
    """
    if character_name not in _reference_audios:
        raise ValueError("Please call 'set_reference_audio' first to set the reference audio.")

    if save_path:
        save_path = os.fspath(save_path)
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    # Create a queue for asynchronous audio chunk transfer
    stream_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    # Set the current TTS context
    context.current_speaker = character_name
    context.current_prompt_audio = ReferenceAudio(
        prompt_wav=_reference_audios[character_name]['audio_path'],
        prompt_text=_reference_audios[character_name]['audio_text'],
    )

    # Start the TTS session
    tts_player.start_session(
        play=play,
        split=split_sentence,
        save_path=save_path,
        stream_queue=stream_queue,
    )

    # Feed text to the TTS engine
    tts_player.feed(text)
    tts_player.end_session()

    # Async iterator: continuously get data from the queue
    while True:
        chunk = await stream_queue.get()
        if chunk is None:
            # None indicates the end of the TTS stream
            break
        yield chunk


def tts(
        character_name: str,
        text: str,
        play: bool = False,
        split_sentence: bool = False,
        save_path: str | PathLike | None = None,
        stream_queue: Optional[asyncio.Queue] = None
) -> None:
    """
    Synchronously generates speech from text.

    This is a blocking function that will not return until the entire TTS
    process is complete.

    Args:
        character_name (str): The name of the character to use for synthesis.
        text (str): The text to be synthesized into speech.
        play (bool, optional): If True, plays the audio. Defaults to False.
        split_sentence (bool, optional): If True, splits the text into sentences for synthesis. Defaults to False.
        save_path (str | PathLike | None, optional): If provided, saves the generated audio to this file path. Defaults to None.
        stream_queue (Optional[asyncio.Queue], optional): An optional asyncio queue to stream audio chunks to. Defaults to None.
    """
    if character_name not in _reference_audios:
        logger.error("Please call 'set_reference_audio' first to set the reference audio.")
        return

    if save_path:
        save_path = os.fspath(save_path)
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    context.current_speaker = character_name
    context.current_prompt_audio = ReferenceAudio(
        prompt_wav=_reference_audios[character_name]['audio_path'],
        prompt_text=_reference_audios[character_name]['audio_text'],
    )

    tts_player.start_session(
        play=play,
        split=split_sentence,
        save_path=save_path,
        stream_queue=stream_queue,
    )
    tts_player.feed(text)
    tts_player.end_session()
    tts_player.wait_for_tts_completion()


def stop() -> None:
    """
    Stops the currently playing text-to-speech audio.
    """
    tts_player.stop()


def convert_to_onnx(
        torch_ckpt_path: str | PathLike,
        torch_pth_path: str | PathLike,
        output_dir: str | PathLike
) -> None:
    """
    Converts PyTorch model checkpoints to the ONNX format.

    This function requires PyTorch to be installed.

    Args:
        torch_ckpt_path (str | PathLike): The path to the T2S model (.ckpt) file.
        torch_pth_path (str | PathLike): The path to the VITS model (.pth) file.
        output_dir (str | PathLike): The directory where the ONNX models will be saved.
    """
    try:
        import torch
    except ImportError:
        logger.error("❌ PyTorch is not installed. Please run `pip install torch` first.")
        return

    from .Converter.v2.Converter import convert

    torch_ckpt_path = os.fspath(torch_ckpt_path)
    torch_pth_path = os.fspath(torch_pth_path)
    output_dir = os.fspath(output_dir)

    convert(
        torch_pth_path=torch_pth_path,
        torch_ckpt_path=torch_ckpt_path,
        output_dir=output_dir,
    )


def clear_reference_audio_cache() -> None:
    """
    Clears the cache of reference audio data.
    """
    ReferenceAudio.clear_cache()
