import onnxruntime as ort
import onnx
from onnx import helper, TensorProto
import numpy as np
from typing import Iterator, List, Optional
import threading

from ..Audio.ReferenceAudio import ReferenceAudio
from ..GetPhonesAndBert import get_phones_and_bert

MAX_T2S_LEN = 1000

# Streaming chunking defaults, tuned after GPT-SoVITS streaming_mode=2
# (min_chunk_length=16). The vocoder re-decodes the whole semantic prefix and
# only the tail beyond already-yielded audio is emitted, blended with a short
# crossfade to hide the re-decode boundary.
STREAM_MIN_CHUNK_TOKENS = 16  # semantic tokens (25 Hz) per emitted chunk (~0.64 s)
STREAM_FADE_SAMPLES = 640     # 20 ms crossfade at 32 kHz

# ─── Static CUDA-graph T2S engine ────────────────────────────────────────────


class StaticT2SEngine:
    """Wraps t2s_static.onnx (single decode step, static shapes) with an ORT
    session configured for CUDA graph capture + replay. Pure ORT: all buffer
    traffic (H2D of token/positions, D2D of the KV cache, D2H of logits) goes
    through a small Identity-graph "update" session, so no torch is needed in
    the server process."""

    def __init__(self, onnx_path: str):
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("StaticT2SEngine requires the CUDA execution provider")
        so = ort.SessionOptions()
        so.add_session_config_entry("enable_cuda_graph", "1")
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CUDAExecutionProvider"])
        ins = {i.name: i.shape for i in self.sess.get_inputs()}
        outs = {o.name: o.shape for o in self.sess.get_outputs()}
        self.L = ins["kv_k_in"][0]
        self.max_len = ins["kv_k_in"][2]
        self.H = ins["kv_k_in"][3]
        self.D = ins["kv_k_in"][4]
        self.vocab = outs["logits"][1] if isinstance(outs["logits"][1], int) else 1025

        # GPU-resident buffers (created once; addresses never change)
        mk = lambda arr: ort.OrtValue.ortvalue_from_numpy(arr, device_type="cuda", device_id=0)
        self.token_b = mk(np.zeros((1, 1), dtype=np.int64))
        self.pe_pos_b = mk(np.zeros((1,), dtype=np.int64))
        self.kv_pos_b = mk(np.zeros((1,), dtype=np.int64))
        self.kvk_in = mk(np.zeros((self.L, 1, self.max_len, self.H, self.D), dtype=np.float32))
        self.kvv_in = mk(np.zeros((self.L, 1, self.max_len, self.H, self.D), dtype=np.float32))
        self.logits_b = mk(np.zeros((1, self.vocab), dtype=np.float32))
        self.kvk_out = mk(np.zeros((self.L, 1, self.max_len, self.H, self.D), dtype=np.float32))
        self.kvv_out = mk(np.zeros((self.L, 1, self.max_len, self.H, self.D), dtype=np.float32))

        io = self.sess.io_binding()
        io.bind_ortvalue_input("token_id", self.token_b)
        io.bind_ortvalue_input("input_pos", self.pe_pos_b)
        io.bind_ortvalue_input("kv_pos", self.kv_pos_b)
        io.bind_ortvalue_input("kv_k_in", self.kvk_in)
        io.bind_ortvalue_input("kv_v_in", self.kvv_in)
        io.bind_ortvalue_output("logits", self.logits_b)
        io.bind_ortvalue_output("kv_k_out", self.kvk_in)  # alias: write in place
        io.bind_ortvalue_output("kv_v_out", self.kvv_in)  # alias
        self.io = io

        # Identity-graph update session:
        #   inputs  (CPU): token, pe_pos, kv_pos      (small, new values per step)
        #   inputs  (GPU): kvk_src, kvv_src           (main-session outputs, or CPU for reset)
        #   outputs (GPU): token_b, pe_pos_b, kv_pos_b, kvk_in, kvv_in  (in-place)
        s_dims = [str(self.L), "b", str(self.max_len), str(self.H), str(self.D)]
        s_in = [
            ("i_token", TensorProto.INT64, [1, 1]),
            ("i_pe_pos", TensorProto.INT64, [1]),
            ("i_kv_pos", TensorProto.INT64, [1]),
            ("i_kvk_src", TensorProto.FLOAT, s_dims),
            ("i_kvv_src", TensorProto.FLOAT, s_dims),
        ]
        s_out = [
            ("o_token", TensorProto.INT64, [1, 1]),
            ("o_pe_pos", TensorProto.INT64, [1]),
            ("o_kv_pos", TensorProto.INT64, [1]),
            ("o_kvk", TensorProto.FLOAT, s_dims),
            ("o_kvv", TensorProto.FLOAT, s_dims),
        ]
        inputs = [helper.make_tensor_value_info(n, t, sh) for n, t, sh in s_in]
        outputs = [helper.make_tensor_value_info(n, t, sh) for n, t, sh in s_out]
        nodes = [helper.make_node("Identity", [i[0]], [o[0]]) for i, o in zip(s_in, s_out)]
        graph = helper.make_graph(nodes, "update", inputs, outputs)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        self.update_sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.io_update = self.update_sess.io_binding()
        self.io_update.bind_ortvalue_output("o_token", self.token_b)
        self.io_update.bind_ortvalue_output("o_pe_pos", self.pe_pos_b)
        self.io_update.bind_ortvalue_output("o_kv_pos", self.kv_pos_b)
        self.io_update.bind_ortvalue_output("o_kvk", self.kvk_in)
        self.io_update.bind_ortvalue_output("o_kvv", self.kvv_in)
        # step-time sources: main-session outputs (GPU OrtValues)
        self.io_update.bind_ortvalue_input("i_kvk_src", self.kvk_out)
        self.io_update.bind_ortvalue_input("i_kvv_src", self.kvv_out)

    def reset(self, present: List[np.ndarray], past_len: int):
        """Load the prefill KV (text + prompt) from the first-stage graph into
        the static buffers. present is interleaved [pk0, pv0, pk1, pv1, ...]."""
        full_k = np.zeros((self.L, 1, self.max_len, self.H, self.D), dtype=np.float32)
        full_v = np.zeros_like(full_k)
        for l in range(self.L):
            full_k[l, 0, :past_len] = present[2 * l].reshape(past_len, self.H, self.D)
            full_v[l, 0, :past_len] = present[2 * l + 1].reshape(past_len, self.H, self.D)
        # H2D through the update session: bind the CPU arrays as sources
        self.io_update.bind_input("i_kvk_src", "cpu", 0, np.float32, list(full_k.shape), full_k.ctypes.data)
        self.io_update.bind_input("i_kvv_src", "cpu", 0, np.float32, list(full_v.shape), full_v.ctypes.data)
        self.update_sess.run_with_iobinding(self.io_update)
        # restore the step-time GPU sources
        self.io_update.bind_ortvalue_input("i_kvk_src", self.kvk_out)
        self.io_update.bind_ortvalue_input("i_kvv_src", self.kvv_out)

    def step(self, token: int, pe_pos: int, kv_pos: int) -> np.ndarray:
        self.io_update.bind_input("i_token", "cpu", 0, np.int64, [1, 1],
                                  np.array([[token]], dtype=np.int64).ctypes.data)
        self.io_update.bind_input("i_pe_pos", "cpu", 0, np.int64, [1],
                                  np.array([pe_pos], dtype=np.int64).ctypes.data)
        self.io_update.bind_input("i_kv_pos", "cpu", 0, np.int64, [1],
                                  np.array([kv_pos], dtype=np.int64).ctypes.data)
        self.update_sess.run_with_iobinding(self.io_update)
        self.sess.run_with_iobinding(self.io)
        return self.logits_b.numpy()[0]


def _sample_numpy(logits: np.ndarray, previous_tokens, top_k: int = 15,
                  top_p: float = 1.0, temperature: float = 1.0,
                  repetition_penalty: float = 1.35) -> int:
    """Numpy port of AR/models/utils.py sample() (gumbel-exponential trick)."""
    x = logits.astype(np.float32).copy()
    if previous_tokens:
        pt = np.asarray(previous_tokens, dtype=np.int64)
        s = x[pt]
        x[pt] = np.where(s < 0, s * repetition_penalty, s / repetition_penalty)
    kth = np.sort(x)[-top_k]
    x[x < kth] = -np.inf
    if top_p < 1.0:
        sd = np.sort(x)[::-1]
        cum = np.cumsum(np.exp(sd - sd[0]) / np.sum(np.exp(sd - sd[0])))
        kill = cum > top_p
        kill[0] = False
        x[sd[kill]] = -np.inf
    x = x / max(temperature, 1e-5)
    finite_max = np.max(np.where(np.isfinite(x), x, -np.inf))
    e = np.exp(x - finite_max)
    e[~np.isfinite(x)] = 0.0
    probs = e / e.sum()
    q = -np.log(np.random.random(probs.shape) + 1e-12)  # Exp(1)
    return int(np.argmax(probs / q))
class GENIE:
    def __init__(self):
        self.stop_event: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Non-streaming API (unchanged signature), now built on tts_stream.
    # ------------------------------------------------------------------
    def tts(
            self,
            text: str,
            prompt_audio: ReferenceAudio,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            vocoder: ort.InferenceSession,
            prompt_encoder: Optional[ort.InferenceSession],
            language: str = 'japanese',
    ) -> Optional[np.ndarray]:
        pieces = list(self.tts_stream(
            text=text,
            prompt_audio=prompt_audio,
            encoder=encoder,
            first_stage_decoder=first_stage_decoder,
            stage_decoder=stage_decoder,
            vocoder=vocoder,
            prompt_encoder=prompt_encoder,
            language=language,
        ))
        if not pieces:
            return None
        return np.concatenate(pieces, axis=-1)[None, :]

    # ------------------------------------------------------------------
    # Streaming API: yields float32 audio chunks while the T2S decoder is
    # still generating semantic tokens (generate-while-play).
    # ------------------------------------------------------------------
    def tts_stream(
            self,
            text: str,
            prompt_audio: ReferenceAudio,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            vocoder: ort.InferenceSession,
            prompt_encoder: Optional[ort.InferenceSession],
            language: str = 'japanese',
            min_chunk_tokens: int = STREAM_MIN_CHUNK_TOKENS,
            fade_samples: int = STREAM_FADE_SAMPLES,
            static_engine=None,
    ) -> Iterator[np.ndarray]:
        text = '。' + text  # 防止漏第一句。
        text_seq, text_bert = get_phones_and_bert(text, language=language)

        # V2Pro/V2ProPlus: speaker embeddings are shared across all chunks.
        if prompt_encoder is not None:
            prompt_audio.update_global_emb(prompt_encoder=prompt_encoder)
        is_pro = prompt_encoder is not None

        tok_iter: Iterator[tuple[np.ndarray, bool]]
        if static_engine is not None:
            tok_iter = self._t2s_stream_static(
                static_engine, ref_seq=prompt_audio.phonemes_seq,
                ref_bert=prompt_audio.text_bert, text_seq=text_seq,
                text_bert=text_bert, ssl_content=prompt_audio.ssl_content,
                encoder=encoder, first_stage_decoder=first_stage_decoder,
                chunk_tokens=min_chunk_tokens)
        else:
            tok_iter = self._t2s_stream(
                ref_seq=prompt_audio.phonemes_seq,
                ref_bert=prompt_audio.text_bert,
                text_seq=text_seq,
                text_bert=text_bert,
                ssl_content=prompt_audio.ssl_content,
                encoder=encoder,
                first_stage_decoder=first_stage_decoder,
                stage_decoder=stage_decoder,
                chunk_tokens=min_chunk_tokens)

        def decode(tokens: np.ndarray) -> Optional[np.ndarray]:
            if tokens is None or tokens.shape[-1] == 0:
                return None
            feed = {"text_seq": text_seq, "pred_semantic": tokens}
            if is_pro:
                feed["ge"] = prompt_audio.global_emb
                feed["ge_advanced"] = prompt_audio.global_emb_advanced
            else:
                feed["ref_audio"] = prompt_audio.audio_32k
            return vocoder.run(None, feed)[0]

        sent = 0      # samples of the ideal signal already emitted
        tail = None   # held-back samples (fade window) not yet emitted
        ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        for tokens, is_final in tok_iter:
            if self.stop_event.is_set():
                return
            full = decode(tokens)
            if full is None:
                continue
            if full.ndim == 2:
                full = full[0]
            length = full.shape[-1]
            if is_final:
                if tail is None:
                    if length > sent:
                        yield full[sent:]
                elif length > sent + fade_samples:
                    blended = tail * (1.0 - ramp) + full[sent:sent + fade_samples] * ramp
                    yield np.concatenate([blended, full[sent + fade_samples:]], axis=-1)
                else:
                    # Degenerate final chunk: flush without blending.
                    yield np.concatenate([tail, full[sent:]], axis=-1)
                return

            new_end = length - fade_samples
            if new_end <= sent:
                continue  # not enough new audio to emit yet
            if tail is None:
                yield full[:new_end]
            else:
                blended = tail * (1.0 - ramp) + full[sent:sent + fade_samples] * ramp
                yield np.concatenate([blended, full[sent + fade_samples:new_end]], axis=-1)
            sent = new_end
            tail = full[new_end:]

    def _t2s_stream_static(
            self,
            engine,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            chunk_tokens: int,
    ) -> Iterator[tuple[np.ndarray, bool]]:
        """Static CUDA-graph decode: prefill via the dynamic first-stage graph,
        then per-step replay of t2s_static.onnx; sampling in numpy."""
        eos = engine.vocab - 1
        x, prompts = encoder.run(
            None,
            {
                "ref_seq": ref_seq,
                "text_seq": text_seq,
                "ref_bert": ref_bert,
                "text_bert": text_bert,
                "ssl_content": ssl_content,
            },
        )
        first_out = first_stage_decoder.run(None, {"x": x, "prompts": prompts})
        y, y_emb, *present = first_out
        y_tokens = y[0].tolist()
        P = y.shape[1] - 1  # last element of y is the first sampled token
        past_len = present[0].shape[0]
        engine.reset(present, past_len)

        pe_pos = P
        kv_pos = past_len
        token = y_tokens[-1]
        gen_idx = 1
        if token == eos:
            yield np.asarray(y_tokens[P:-1], dtype=np.int64)[None, None, :], True
            return

        for _ in range(engine.max_len - kv_pos - 2):
            if self.stop_event.is_set():
                return
            logits = engine.step(token, pe_pos, kv_pos)
            nxt = _sample_numpy(logits, y_tokens)
            gen_idx += 1
            if nxt == eos:
                yield np.asarray(y_tokens[P:], dtype=np.int64)[None, None, :], True
                return
            y_tokens.append(nxt)
            if gen_idx % chunk_tokens == 0:
                yield np.asarray(y_tokens[P:], dtype=np.int64)[None, None, :], False
            token = nxt
            pe_pos += 1
            kv_pos += 1
        yield np.asarray(y_tokens[P:], dtype=np.int64)[None, None, :], True

    def _t2s_stream(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            chunk_tokens: int,
    ) -> Iterator[tuple[np.ndarray, bool]]:
        """Incremental T2S decoding; yields (generated token prefix, is_final)."""
        if 'CUDAExecutionProvider' in (stage_decoder.get_providers() or []):
            yield from self._t2s_stream_cuda(
                ref_seq, ref_bert, text_seq, text_bert, ssl_content,
                encoder, first_stage_decoder, stage_decoder, chunk_tokens)
        else:
            yield from self._t2s_stream_cpu(
                ref_seq, ref_bert, text_seq, text_bert, ssl_content,
                encoder, first_stage_decoder, stage_decoder, chunk_tokens)

    def _t2s_stream_cuda(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            chunk_tokens: int,
    ) -> Iterator[tuple[np.ndarray, bool]]:
        """CUDA IOBinding variant: the KV cache and embeddings stay resident in
        GPU memory across decode steps (run_with_ort_values), removing the
        per-step H2D/D2H round-trip of ~10-15 MB of KV tensors that otherwise
        dominates step latency. Only the scalar stop condition is copied back
        each step (required for the EOS check), plus a tiny token-prefix copy
        on streaming chunk boundaries."""
        def gpu(arr: np.ndarray) -> ort.OrtValue:
            return ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(arr),
                                                    device_type='cuda', device_id=0)

        # Encoder (one-shot): run on device, keep outputs on GPU.
        x, prompts = encoder.run(
            None,
            {
                "ref_seq": ref_seq,
                "text_seq": text_seq,
                "ref_bert": ref_bert,
                "text_bert": text_bert,
                "ssl_content": ssl_content,
            },
        )
        first_out = first_stage_decoder.run_with_ort_values(
            None, {"x": gpu(x), "prompts": gpu(prompts)}
        )
        y, y_emb, *present = first_out

        stage_input_names = [i.name for i in stage_decoder.get_inputs()]
        noise_input = next((i for i in stage_decoder.get_inputs() if i.name == 'rand_noise'), None)
        vocab = noise_input.shape[0] if (noise_input and len(noise_input.shape) > 0 and isinstance(noise_input.shape[0], int)) else None
        for idx in range(0, 500):
            if self.stop_event.is_set():
                return
            feed = dict(zip(stage_input_names, [y, y_emb, *present]))
            if noise_input is not None and vocab:
                feed['rand_noise'] = gpu(np.random.randn(vocab).astype(np.float32))
            outputs = stage_decoder.run_with_ort_values(None, feed)
            y, y_emb, stop_ort, *present = outputs

            # EOS check: the only mandatory per-step device sync (scalar).
            stop_condition_tensor = np.asarray(stop_ort.numpy()).ravel()[0]
            if stop_condition_tensor:
                break

            # Emit the token prefix on chunk boundaries for streaming decode.
            # y is int64 [1, seq]: a few KB, negligible to copy.
            if (idx + 1) % chunk_tokens == 0:
                y_cpu = y.numpy()
                yield np.expand_dims(y_cpu[:, -idx:], axis=0), False

        if self.stop_event.is_set():
            return
        y_cpu = y.numpy()
        y_cpu[0, -1] = 0
        tokens = np.expand_dims(y_cpu[:, -idx:], axis=0)
        # Trim anything at/after EOS (same rule as the non-streaming path).
        eos_indices = np.where(tokens >= 1024)
        if len(eos_indices[0]) > 0:
            tokens = tokens[..., :eos_indices[-1][0]]
        yield tokens, True

    def _t2s_stream_cpu(
            self,
            ref_seq: np.ndarray,
            ref_bert: np.ndarray,
            text_seq: np.ndarray,
            text_bert: np.ndarray,
            ssl_content: np.ndarray,
            encoder: ort.InferenceSession,
            first_stage_decoder: ort.InferenceSession,
            stage_decoder: ort.InferenceSession,
            chunk_tokens: int,
    ) -> Iterator[tuple[np.ndarray, bool]]:
        """Incremental T2S decoding; yields (generated token prefix, is_final)."""
        # Encoder
        x, prompts = encoder.run(
            None,
            {
                "ref_seq": ref_seq,
                "text_seq": text_seq,
                "ref_bert": ref_bert,
                "text_bert": text_bert,
                "ssl_content": ssl_content,
            },
        )

        # First Stage Decoder
        y, y_emb, *present_key_values = first_stage_decoder.run(
            None, {"x": x, "prompts": prompts}
        )

        # Stage Decoder
        input_names: List[str] = [inp.name for inp in stage_decoder.get_inputs()]
        noise_input = next((inp for inp in stage_decoder.get_inputs() if inp.name == 'rand_noise'), None)
        vocab = noise_input.shape[0] if (noise_input and len(noise_input.shape) > 0 and isinstance(noise_input.shape[0], int)) else None
        idx: int = 0
        for idx in range(0, 500):
            if self.stop_event.is_set():
                return
            input_feed = {
                name: data
                for name, data in zip(input_names, [y, y_emb, *present_key_values])
            }
            if noise_input is not None and vocab:
                input_feed['rand_noise'] = np.random.randn(vocab).astype(np.float32)
            outputs = stage_decoder.run(None, input_feed)
            y, y_emb, stop_condition_tensor, *present_key_values = outputs

            if stop_condition_tensor:
                break

            # Emit the token prefix on chunk boundaries for streaming decode.
            if (idx + 1) % chunk_tokens == 0:
                yield np.expand_dims(y[:, -idx:], axis=0), False

        if self.stop_event.is_set():
            return
        y[0, -1] = 0
        tokens = np.expand_dims(y[:, -idx:], axis=0)
        # Trim anything at/after EOS (same rule as the non-streaming path).
        eos_indices = np.where(tokens >= 1024)
        if len(eos_indices[0]) > 0:
            tokens = tokens[..., :eos_indices[-1][0]]
        yield tokens, True


tts_client: GENIE = GENIE()
