import contextlib
import importlib.resources
import logging
import os
import traceback

import numpy as np
import onnx

from ...Utils.Constants import PACKAGE_NAME
from ..load_state_dict import load_sovits_model
from ..v2.Converter import (ENCODER_RESOURCE_PATH, STAGE_DECODER_RESOURCE_PATH,
                            FIRST_STAGE_DECODER_RESOURCE_PATH, T2S_KEYS_RESOURCE_PATH,
                            CACHE_DIR, remove_folder)
from ..v2.T2SConverter import T2SModelConverter
from ..v2.VITSConverter import VITSConverter
from ..v2.EncoderConverter import EncoderConverter
from ..v2ProPlus.PromptEncoderConverter import PromptEncoderConverter

logger = logging.getLogger()

# The v2Pro s2 model shares the v2ProPlus graph topology (same layer counts
# and key names) but differs in decoder hyper-parameters (e.g. hidden
# channels 512 vs 768, different upsample kernels). ONNX Conv kernels are
# taken from the weights themselves, so the shipped v2ProPlus template only
# needs its initializer dims adapted to the checkpoint before weight
# relinking.
VPP_VITS_TEMPLATE_PATH = "Data/v2ProPlus/Models/vits_fp32.onnx"
VPP_PROMPT_ENCODER_PATH = "Data/v2ProPlus/Models/prompt_encoder_fp32.onnx"
VPP_VITS_KEYS_RESOURCE_PATH = "./Data/v2ProPlus/Keys/vits_weights.txt"
VPP_PROMPT_ENCODER_KEYS_RESOURCE_PATH = "./Data/v2ProPlus/Keys/prompt_encoder_weights.txt"

ADAPTED_VITS_TEMPLATE = os.path.join(CACHE_DIR, "v2Pro_vits_template_fp32.onnx")


def _build_adapted_vits_template(torch_pth_path: str, out_path: str) -> None:
    """Patch the v2ProPlus vits template initializer dims to match the checkpoint."""
    model = onnx.load(VPP_VITS_TEMPLATE_PATH_RESOLVED, load_external_data=False)
    d = load_sovits_model(torch_pth_path)
    state = d['weight']

    patched = 0
    for tensor in model.graph.initializer:
        key = tensor.name[len("vq_model."):] if tensor.name.startswith("vq_model.") else tensor.name
        if key not in state:
            continue
        shape = list(np.asarray(state[key]).shape)
        if list(tensor.dims) == shape:
            continue
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            # Weight relinking only rewrites external-data tensors; an
            # embedded (raw_data) initializer with a different shape would
            # mean a real graph difference we cannot patch safely.
            raise ValueError(
                f"Embedded initializer '{tensor.name}' shape {list(tensor.dims)} "
                f"differs from checkpoint shape {shape}; v2Pro graph diverges "
                f"beyond initializer shapes and needs a dedicated template."
            )
        del tensor.dims[:]
        tensor.dims.extend(shape)
        patched += 1

    _patch_conv_transpose_attrs(model, state, d)
    onnx.save(model, out_path)
    logger.info(f"v2Pro template adapted: {patched} initializer shapes updated")


def _patch_conv_transpose_attrs(model, state, checkpoint_dict) -> None:
    """Rewrite dec.ups ConvTranspose attributes from the checkpoint's hps config.

    The template hardcodes the v2ProPlus upsample rates/kernel sizes as node
    attributes; v2Pro uses different ones (e.g. kernels [16,16,8,2,2] with
    rates [10,8,2,2,2] vs Plus [20,16,8,2,2]).
    """
    cfg = checkpoint_dict.get('config')
    model_cfg = None
    if cfg is not None:
        # config may be a plain dict or an HParams-style attribute object
        model_cfg = cfg.get('model') if isinstance(cfg, dict) else getattr(cfg, 'model', None)
    if not model_cfg:
        raise ValueError("Checkpoint config does not contain model hps; cannot patch ConvTranspose attributes")

    def _get(obj, name):
        if isinstance(obj, dict):
            return obj[name]
        return getattr(obj, name)

    kernels = _get(model_cfg, 'upsample_kernel_sizes')
    rates = _get(model_cfg, 'upsample_rates')
    initial_channel = _get(model_cfg, 'upsample_initial_channel')

    by_name = {t.name: t for t in model.graph.initializer}
    patched = 0
    for node in model.graph.node:
        if node.op_type != 'ConvTranspose' or len(node.input) < 3:
            continue
        # weight_norm fuses g*v into a Mul node, so the weight input is not an
        # initializer; the bias input reliably carries the layer name.
        import re
        m = re.match(r'vq_model\.dec\.ups\.(\d+)\.bias$', node.input[2])
        if not m:
            continue
        i = int(m.group(1))
        k, u = kernels[i], rates[i]
        p = (k - u) // 2
        # sanity: channel widths implied by the config must match the weights
        expect_w = [initial_channel // (2 ** i), initial_channel // (2 ** (i + 1)), k]
        w_dims = list(by_name[f'vq_model.dec.ups.{i}.weight_v'].dims)
        if w_dims != expect_w:
            raise ValueError(f"ups.{i} weight dims {w_dims} contradict config {expect_w}")
        attrs = {a.name: a for a in node.attribute}
        def _set(name, ints):
            if name in attrs:
                attrs[name].ClearField('ints')
                attrs[name].ints.extend(ints)
        _set('kernel_shape', [k])
        _set('strides', [u])
        _set('pads', [p, p])
        _set('output_padding', [0])
        _set('dilations', [1])
        patched += 1
    logger.info(f"v2Pro ConvTranspose attributes patched on {patched} nodes "
                f"(kernels={kernels}, rates={rates})")


# resolved at convert() time inside the ExitStack
VPP_VITS_TEMPLATE_PATH_RESOLVED = None


def convert(torch_ckpt_path: str, torch_pth_path: str, output_dir: str) -> None:
    global VPP_VITS_TEMPLATE_PATH_RESOLVED
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if len(os.listdir(output_dir)) > 0:
        logger.warning(f"The output directory {output_dir} is not empty!")

    with contextlib.ExitStack() as stack:
        files = importlib.resources.files(PACKAGE_NAME)

        def enter(p: str) -> str:
            return str(stack.enter_context(importlib.resources.as_file(files.joinpath(p))))

        encoder_onnx_path = enter(ENCODER_RESOURCE_PATH)
        stage_decoder_path = enter(STAGE_DECODER_RESOURCE_PATH)
        first_stage_decoder_path = enter(FIRST_STAGE_DECODER_RESOURCE_PATH)
        vpp_vits_template_path = enter(VPP_VITS_TEMPLATE_PATH)
        prompt_encoder_path = enter(VPP_PROMPT_ENCODER_PATH)
        t2s_keys_path = enter(T2S_KEYS_RESOURCE_PATH)
        vits_keys_path = enter(VPP_VITS_KEYS_RESOURCE_PATH)
        prompt_encoder_keys_path = enter(VPP_PROMPT_ENCODER_KEYS_RESOURCE_PATH)

        VPP_VITS_TEMPLATE_PATH_RESOLVED = vpp_vits_template_path
        _build_adapted_vits_template(torch_pth_path, ADAPTED_VITS_TEMPLATE)

        converter_1 = T2SModelConverter(
            torch_ckpt_path=torch_ckpt_path,
            stage_decoder_onnx_path=stage_decoder_path,
            first_stage_decoder_onnx_path=first_stage_decoder_path,
            key_list_file=t2s_keys_path,
            output_dir=output_dir,
            cache_dir=CACHE_DIR,
        )
        converter_2 = VITSConverter(
            torch_pth_path=torch_pth_path,
            vits_onnx_path=ADAPTED_VITS_TEMPLATE,
            key_list_file=vits_keys_path,
            output_dir=output_dir,
            cache_dir=CACHE_DIR,
        )
        converter_3 = EncoderConverter(
            ckpt_path=torch_ckpt_path,
            pth_path=torch_pth_path,
            onnx_input_path=encoder_onnx_path,
            output_dir=output_dir,
        )
        converter_4 = PromptEncoderConverter(
            torch_pth_path=torch_pth_path,
            prompt_encoder_onnx_path=prompt_encoder_path,
            key_list_file=prompt_encoder_keys_path,
            output_dir=output_dir,
            cache_dir=CACHE_DIR,
        )

        try:
            converter_1.run_full_process()
            converter_2.run_full_process()
            converter_3.run_full_process()
            converter_4.run_full_process()
            logger.info(f"🎉 Conversion successful! Saved to: {os.path.abspath(output_dir)}\n"
                        f"- Model Type: V2Pro")
        except Exception:
            logger.error(f"❌ A critical error occurred during the conversion process")
            logger.error(traceback.format_exc())
            remove_folder(output_dir)
        finally:
            remove_folder(CACHE_DIR)
