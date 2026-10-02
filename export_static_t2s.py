# -*- coding: utf-8 -*-
# Export a static-shape, CUDA-graph-friendly single decode step of the
# GPT-SoVITS v2Pro s1 (t2s) model to ONNX.
#
# Graph contract (all shapes static):
#   inputs:
#     token_id  [1, 1]      int64   - semantic token to process this step
#     input_pos [1]         int64   - 0-based position of this token in the
#                                     semantic sequence (= tokens before it)
#     kv_k_in   [L,1,MAX,H,D] fp32  - per-layer key cache (zero padded)
#     kv_v_in   [L,1,MAX,H,D] fp32
#   outputs:
#     logits    [1, V]      fp32    - sampling happens OUTSIDE the graph
#     kv_k_out / kv_v_out [L,1,MAX,H,D] - caches with this step's K/V written
#
# Sampling (top-k/top-p/repetition-penalty) and the EOS check run in Python,
# which also removes the CPU-only Random/TopK ops that made the original
# template fall back to CPU segments.
#
# Run with the GPT-SoVITS integrated-package runtime (needs torch + onnx).
#
# NOTE: the s1/t2s architecture is identical across GPT-SoVITS v2 / v2Pro /
# v2ProPlus / v3 (they only differ in the s2 vocoder side), so this exporter
# is universal: pass a different --ckpt to export another variant's weights.
import argparse
import math
import os
import sys

# AR/ (Text2SemanticDecoder) lives in the GPT-SoVITS package; point
# --gptsovits at its root (the folder that CONTAINS the GPT_SoVITS subdir)
GPT_SOVITS = os.environ.get("GPT_SOVITS_DIR", "")
if GPT_SOVITS:
    sys.path.insert(0, GPT_SOVITS)
    sys.path.insert(0, os.path.join(GPT_SOVITS, "GPT_SoVITS"))

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-5
MAX_LEN = 2048


def load_decoder(ckpt_path):
    from AR.models.t2s_model import Text2SemanticDecoder
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = d["config"]
    w = d["weight"]
    stripped = {k[len("model."):] if k.startswith("model.") else k: v for k, v in w.items()}
    dec = Text2SemanticDecoder(cfg, norm_first=False)
    missing, unexpected = dec.load_state_dict(stripped, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    dec.eval()
    m = cfg["model"]
    return dec, m


def load_decoder_from_components(comp_dir, cfg, key_list_path):
    """Rebuild the t2s decoder weights from Genie's ONNX component layout
    (t2s_stage_decoder_fp32.onnx + t2s_shared_fp16.bin + the t2s key list),
    for characters published without a .ckpt (e.g. feibi).  The component
    initializers reference the fp16 bin through an fp32-byte offset/length
    table - the same convention as ModelManager.load_session_with_fp16_conversion."""
    import onnx

    model = onnx.load(os.path.join(comp_dir, "t2s_stage_decoder_fp32.onnx"),
                      load_external_data=False)
    ext = {}
    for t in model.graph.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            continue
        off = int(next(e.value for e in t.external_data if e.key == "offset"))
        ln = int(next(e.value for e in t.external_data if e.key == "length"))
        ext[t.name] = (off, ln, list(t.dims))

    fp16 = np.fromfile(os.path.join(comp_dir, "t2s_shared_fp16.bin"), dtype=np.float16)
    fp32 = fp16.astype(np.float32).tobytes()  # offsets below are BYTE offsets

    with open(key_list_path) as f:
        onnx_keys = [l.strip() for l in f if l.strip()]
    assert set(onnx_keys) == set(ext), (
        f"key list / component mismatch: {len(onnx_keys)} keys vs {len(ext)} initializers")

    sd = {}
    for k in onnx_keys:
        off, ln, dims = ext[k]
        sd[k.replace("transformer_encoder", "h")] = \
            torch.from_numpy(np.frombuffer(fp32[off:off + ln], dtype=np.float32)
                             .reshape(dims).copy())

    from AR.models.t2s_model import Text2SemanticDecoder
    dec = Text2SemanticDecoder(cfg, norm_first=False)
    missing, unexpected = dec.load_state_dict(sd, strict=False)
    # positional tables are structural buffers rebuilt by the wrapper, not
    # shipped weights - anything else missing means the layout drifted
    # text-side modules (bert proj / text embedding) exist on the decoder
    # class but are never exercised by the audio path and are not shipped
    # in the component bin
    TEXT_SIDE = ("bert_proj", "ar_text_embedding", "ar_text_position")
    real_missing = [k for k in missing
                    if "pe" not in k and not any(t in k for t in TEXT_SIDE)]
    assert not real_missing, f"missing keys: {real_missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    dec.eval()
    return dec, cfg["model"]


class StaticDecodeStep(nn.Module):
    """One autoregressive decode step with static KV buffers and an explicit
    padding mask. Attention math mirrors T2SBlock.decode_next_token exactly
    (scale 1/sqrt(head_dim), post-norm, relu MLP)."""

    def __init__(self, dec: Text2SemanticDecoder, max_len: int):
        super().__init__()
        m = None
        self.max_len = max_len
        self.audio_emb = dec.ar_audio_embedding
        self.alpha = dec.ar_audio_position.alpha  # [1]
        self.predict = dec.ar_predict_layer

        # positional table baked as a persistent buffer (exported as initializer)
        div_term = torch.exp(
            torch.arange(0, 512, 2, dtype=torch.float32) * -(math.log(10000.0) / 512)
        )
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        pe = torch.zeros(1, max_len, 512)
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe, persistent=True)
        self.register_buffer("arange_c", torch.arange(max_len, dtype=torch.int64), persistent=True)

        self.layers = nn.ModuleList()
        for layer in dec.h.layers:
            blk = nn.Module()
            blk.in_proj_w = layer.self_attn.in_proj_weight
            blk.in_proj_b = layer.self_attn.in_proj_bias
            blk.out_w = layer.self_attn.out_proj.weight
            blk.out_b = layer.self_attn.out_proj.bias
            blk.l1w = layer.linear1.weight
            blk.l1b = layer.linear1.bias
            blk.l2w = layer.linear2.weight
            blk.l2b = layer.linear2.bias
            blk.n1w = layer.norm1.weight
            blk.n1b = layer.norm1.bias
            blk.n2w = layer.norm2.weight
            blk.n2b = layer.norm2.bias
            self.layers.append(blk)
        self.n_layer = len(self.layers)
        self.n_head = dec.num_head
        self.head_dim = 512 // dec.num_head
        self.m = m

    def forward(self, token_id, input_pos, kv_pos, kv_k_in, kv_v_in):
        emb = F.embedding(token_id, self.audio_emb.weight)  # [1,1,512]
        pe_v = self.pe.index_select(1, input_pos)  # [1,1,512]
        x = emb + self.alpha * pe_v

        H, D = self.n_head, self.head_dim
        idx = kv_pos.view(1, 1, 1, 1).expand(1, 1, H, D)  # int64
        mask = torch.where(
            self.arange_c <= kv_pos,
            torch.zeros(1, dtype=torch.float32),
            torch.full((1,), float("-1e9"), dtype=torch.float32),
        ).view(1, 1, 1, self.max_len)

        k_outs, v_outs = [], []
        for i, blk in enumerate(self.layers):
            k_in = kv_k_in[i]  # [1,MAX,H,D]
            v_in = kv_v_in[i]
            qkv = F.linear(x, blk.in_proj_w, blk.in_proj_b)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(1, 1, H, D).transpose(1, 2)  # [1,H,1,D]
            idx = input_pos.view(1, 1, 1, 1).expand(1, 1, H, D)  # int64
            k_cat = torch.scatter(k_in, 1, idx, k.view(1, 1, H, D))  # [1,MAX,H,D]
            v_cat = torch.scatter(v_in, 1, idx, v.view(1, 1, H, D))
            scores = torch.matmul(q, k_cat.permute(0, 2, 3, 1)) * (1.0 / math.sqrt(D))
            scores = scores + mask
            p = torch.softmax(scores, dim=-1)
            attn = torch.matmul(p, v_cat.permute(0, 2, 1, 3))  # [1,H,1,D]
            attn = attn.transpose(1, 2).reshape(1, 1, 512)
            attn = F.linear(attn, blk.out_w, blk.out_b)
            x = F.layer_norm(x + attn, (512,), blk.n1w, blk.n1b, EPS)
            x = x + F.linear(F.relu(F.linear(x, blk.l1w, blk.l1b)), blk.l2w, blk.l2b)
            x = F.layer_norm(x, (512,), blk.n2w, blk.n2b, EPS)
            k_outs.append(k_cat)
            v_outs.append(v_cat)

        logits = F.linear(x, self.predict.weight)  # [1,1,V]
        return logits.view(1, -1), torch.stack(k_outs), torch.stack(v_outs)


def reference_logits(dec, tokens, kv_k_list, kv_v_list, pos, P):
    """One decode step through the original module path (grows cache by cat)."""
    tok = torch.tensor([[tokens[pos - P] if isinstance(P, int) else tokens[pos]]], dtype=torch.int64)
    emb = dec.ar_audio_embedding(tok)
    x = emb * 1.0 + dec.ar_audio_position.alpha * dec.ar_audio_position.pe[:, pos]
    for i, layer in enumerate(dec.h.layers):
        qkv = F.linear(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        kv_k_list[i] = torch.cat([kv_k_list[i], k], dim=1)
        kv_v_list[i] = torch.cat([kv_v_list[i], v], dim=1)
        qh = q.view(1, 1, dec.num_head, -1).transpose(1, 2)
        kh = kv_k_list[i].view(1, -1, dec.num_head, 32).transpose(1, 2)
        vh = kv_v_list[i].view(1, -1, dec.num_head, 32).transpose(1, 2)
        a = F.scaled_dot_product_attention(qh, kh, vh)
        a = a.transpose(1, 2).reshape(1, 1, 512)
        a = F.linear(a, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)
        x = F.layer_norm(x + a, (512,), layer.norm1.weight, layer.norm1.bias, EPS)
        x = x + F.linear(F.relu(F.linear(x, layer.linear1.weight, layer.linear1.bias)),
                         layer.linear2.weight, layer.linear2.bias)
        x = F.layer_norm(x, (512,), layer.norm2.weight, layer.norm2.bias, EPS)
    return F.linear(x, dec.ar_predict_layer.weight).view(1, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="GPT-SoVITS checkpoint; also used as the config source "
                         "when --components is given")
    ap.add_argument("--gptsovits", default=os.environ.get("GPT_SOVITS_DIR", ""),
                    help="GPT-SoVITS package root (contains AR/ and GPT_SoVITS/); "
                         "or set GPT_SOVITS_DIR")
    ap.add_argument("--components", default=None,
                    help="Genie component dir (t2s_stage_decoder_fp32.onnx + "
                         "t2s_shared_fp16.bin); weights come from here instead "
                         "of the checkpoint")
    ap.add_argument("--key-list", default=os.path.join(
        r"R:\git\dsh_projects\cyber_persona\Genie-TTS",
        "src", "genie_tts", "Data", "v2", "Keys", "t2s_onnx_keys.txt"))
    ap.add_argument("--out", default="t2s_static.onnx",
                    help="output path for the fp32 static graph")
    ap.add_argument("--max-len", type=int, default=MAX_LEN,
                    help="KV cache length in tokens; smaller shrinks the "
                         "attention scratch and cache (800 ≈ 32 s of audio)")
    args = ap.parse_args()
    max_len = args.max_len

    dec, m = load_decoder(args.ckpt) if args.components is None else \
        load_decoder_from_components(args.components,
                                     torch.load(args.ckpt, map_location="cpu",
                                                weights_only=False)["config"],
                                     args.key_list)
    print("config:", {k: m[k] for k in ("hidden_dim", "head", "n_layer", "vocab_size", "EOS")})

    wrapper = StaticDecodeStep(dec, max_len)
    wrapper.eval()

    # ---- numerical verification vs the original module path ----
    torch.manual_seed(0)
    N_STEPS = 24
    P = 8  # pretend an 8-token prompt exists; start positions at P
    kv_k_list = [torch.zeros(1, 0, 512) for _ in range(m["n_layer"])]
    kv_v_list = [torch.zeros(1, 0, 512) for _ in range(m["n_layer"])]

    kvk_in = torch.zeros(m["n_layer"], 1, max_len, m["head"], 512 // m["head"])
    kvv_in = torch.zeros_like(kvk_in)
    # pre-fill the prompt region of the static buffers with the prompt KV
    # (in the real pipeline this comes from the first-stage graph); for the
    # verification we synthesize a random prompt KV into BOTH paths.
    prompt_k = torch.randn(m["n_layer"], 1, P, m["head"], 512 // m["head"])
    prompt_v = torch.randn(m["n_layer"], 1, P, m["head"], 512 // m["head"])
    kvk_in[:, :, :P] = prompt_k
    kvv_in[:, :, :P] = prompt_v
    kv_k_list = [prompt_k[i, 0].reshape(P, 512).unsqueeze(0) for i in range(m["n_layer"])]
    kv_v_list = [prompt_v[i, 0].reshape(P, 512).unsqueeze(0) for i in range(m["n_layer"])]

    tokens = torch.randint(0, m["EOS"], (N_STEPS,)).tolist()
    max_err = 0.0
    with torch.no_grad():
        for step in range(N_STEPS):
            pos = P + step
            ref = reference_logits(dec, tokens, kv_k_list, kv_v_list, pos, P)
            static_logits, kvk_out, kvv_out = wrapper(
                torch.tensor([[tokens[step]]], dtype=torch.int64),
                torch.tensor([pos], dtype=torch.int64),
                torch.tensor([pos], dtype=torch.int64),
                kvk_in, kvv_in,
            )
            err = (ref - static_logits).abs().max().item()
            if step == 0:
                ref_first = ref.clone()
            max_err = max(max_err, err)
            # advance both paths with the same next token
            kvk_in, kvv_in = kvk_out, kvv_out
            kv_k_list = [kvk_in[i, 0, : pos + 1].reshape(pos + 1, 512).unsqueeze(0)
                         for i in range(m["n_layer"])]
            kv_v_list = [kvv_in[i, 0, : pos + 1].reshape(pos + 1, 512).unsqueeze(0)
                         for i in range(m["n_layer"])]
    print(f"max |ref - static| logits diff over {N_STEPS} steps: {max_err:.2e}")

    # ---- export ----
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_path = args.out
    tok = torch.tensor([[P]], dtype=torch.int64)
    pos = torch.tensor([P], dtype=torch.int64)
    kv_pos = torch.tensor([P], dtype=torch.int64)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (tok, pos, kv_pos, kvk_in, kvv_in),
            out_path,
            opset_version=17,
            do_constant_folding=True,
            input_names=["token_id", "input_pos", "kv_pos", "kv_k_in", "kv_v_in"],
            output_names=["logits", "kv_k_out", "kv_v_out"],
            # the dynamo exporter emits Split(num_outputs=...) which ORT
            # rejects as an illegal attribute; the legacy exporter produces
            # a graph ORT loads cleanly
            dynamo=False,
        )
    print("exported:", out_path, f"{os.path.getsize(out_path)/1e6:.1f} MB")

    # save reference tensors for the ORT-side check
    np.savez(
        os.path.join(os.path.dirname(args.out), "t2s_static_check.npz"),
        tokens=np.array(tokens), P=P,
        prompt_k=prompt_k.numpy(), prompt_v=prompt_v.numpy(),
        ref_first=ref_first.numpy(),
    )
    print("saved check tensors")


if __name__ == "__main__":
    main()
