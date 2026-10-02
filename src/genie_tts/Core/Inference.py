import os

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

# Emotion carry-over (token level): keep the ANCHOR reference tokens at the
# front and append a bounded sliding window of previous generated tokens, so
# prosody chains across sentences while the anchor suppresses drift.
CARRY_ENV = "GENIE_EMOTION_CARRY"
HISTORY_MAX_TOKENS = 150  # ~6 s of semantic tokens at 25 Hz


def _carry_history(prompt_audio):
    """Per-character history list, stored on the cached ReferenceAudio."""
    if os.environ.get(CARRY_ENV) != "1":  # "0" is truthy as a string!
        return None
    hist = getattr(prompt_audio, "carry_history", None)
    if hist is None:
        hist = []
        prompt_audio.carry_history = hist
    return hist


def _trim_history(hist):
    total = sum(len(e["tokens"]) for e in hist)
    while hist and total > HISTORY_MAX_TOKENS:
        total -= len(hist[0]["tokens"])
        hist.pop(0)


def _extend_inputs(text_seq, text_bert, hist):
    """Prepend history utterances' text/bert to the encoder inputs.
    Returns (text_seq_ext, text_bert_ext)."""
    seqs = [np.asarray(e["phones"], dtype=text_seq.dtype).reshape(1, -1) for e in hist]
    berts = [np.asarray(e["bert"], dtype=text_bert.dtype).reshape(-1, text_bert.shape[-1])
             for e in hist]
    seqs.append(text_seq.reshape(1, -1))
    berts.append(text_bert.reshape(-1, text_bert.shape[-1]))
    return np.concatenate(seqs, axis=1), np.concatenate(berts, axis=0)


def _append_history(hist, text_seq, text_bert, tokens):
    if not tokens:
        return
    hist.append({"phones": np.asarray(text_seq).copy(),
                 "bert": np.asarray(text_bert).copy(),
                 "tokens": [int(t) for t in tokens]})
    _trim_history(hist)

# ─── Static CUDA-graph T2S engine ────────────────────────────────────────────


class StaticT2SEngine:
    """Single-session CUDA-graph decode of t2s_static.onnx.

    Design (validated against the ORT 1.30 crash we hit earlier):
      - ONE session with enable_cuda_graph; scalars (token / pe_pos / kv_pos)
        are re-bound each step via bind_cpu_input, which the replay honors.
      - The KV outputs are bound ALIASED onto the input OrtValues: the
        in-graph ScatterElements writes the new K/V after every reader, so
        the cache updates in place with no copy session (the previous
        mixed-device copy session was what segfaulted next to the captured
        memory pool).
      - No torch anywhere; reset() H2D-loads the prefill cache via
        update_inplace with a full-size padded array."""

    def __init__(self, onnx_path: str):
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("StaticT2SEngine requires the CUDA execution provider")
        so = ort.SessionOptions()
        so.add_session_config_entry("enable_cuda_graph", "1")
        self.sess = ort.InferenceSession(
            onnx_path, so,
            providers=[("CUDAExecutionProvider",
                        {"arena_extend_strategy": "kSameAsRequested"})])
        outs = {o.name: o.shape for o in self.sess.get_outputs()}
        ins_meta = {i.name: i for i in self.sess.get_inputs()}
        ins = {name: m.shape for name, m in ins_meta.items()}
        self.L = ins["kv_k_in"][0]
        self.max_len = ins["kv_k_in"][2]
        self.H = ins["kv_k_in"][3]
        self.D = ins["kv_k_in"][4]
        self.vocab = outs["logits"][1] if isinstance(outs["logits"][1], int) else 1025
        # fp16 graphs (converted by convert_static_fp16.py) keep everything -
        # weights, KV cache, captured activations - at half the VRAM
        self.kv_dtype = np.float16 if 'float16' in ins_meta["kv_k_in"].type else np.float32
        self.logits_dtype = np.float16 if 'float16' in self.sess.get_outputs()[0].type else np.float32

        self.kvk = ort.OrtValue.ortvalue_from_numpy(
            np.zeros((self.L, 1, self.max_len, self.H, self.D), dtype=self.kv_dtype), "cuda", 0)
        self.kvv = ort.OrtValue.ortvalue_from_numpy(
            np.zeros((self.L, 1, self.max_len, self.H, self.D), dtype=self.kv_dtype), "cuda", 0)
        self.logits_b = ort.OrtValue.ortvalue_from_numpy(
            np.zeros((1, self.vocab), dtype=self.logits_dtype), "cuda", 0)

        self.io = self.sess.io_binding()
        self.io.bind_ortvalue_input("kv_k_in", self.kvk)
        self.io.bind_ortvalue_input("kv_v_in", self.kvv)
        self.io.bind_ortvalue_output("logits", self.logits_b)
        self.io.bind_ortvalue_output("kv_k_out", self.kvk)  # alias: in-place
        self.io.bind_ortvalue_output("kv_v_out", self.kvv)  # alias: in-place
        self._warm = False

    def reset(self, present: List[np.ndarray], past_len: int):
        """H2D-load the prefill KV (interleaved [pk0, pv0, pk1, pv1, ...])
        produced by the first-stage graph. Full-size padded arrays so
        update_inplace sizes match; values are cast to the cache dtype
        (fp32 first-stage output -> fp16 cache when the static graph is
        converted)."""
        full_k = np.zeros((self.L, 1, self.max_len, self.H, self.D), dtype=self.kv_dtype)
        full_v = np.zeros_like(full_k)
        for l in range(self.L):
            full_k[l, 0, :past_len] = present[2 * l].reshape(past_len, self.H, self.D)
            full_v[l, 0, :past_len] = present[2 * l + 1].reshape(past_len, self.H, self.D)
        self.kvk.update_inplace(full_k)
        self.kvv.update_inplace(full_v)

    def _bind_scalars(self, token: int, pe_pos: int, kv_pos: int):
        self.io.bind_cpu_input("token_id", np.array([[token]], dtype=np.int64))
        self.io.bind_cpu_input("input_pos", np.array([pe_pos], dtype=np.int64))
        self.io.bind_cpu_input("kv_pos", np.array([kv_pos], dtype=np.int64))

    def step(self, token: int, pe_pos: int, kv_pos: int) -> np.ndarray:
        self._bind_scalars(token, pe_pos, kv_pos)
        self.sess.run_with_iobinding(self.io)
        self._warm = True
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
            vocoder_overlap_tokens: int = 0,
            static_engine=None,
    ) -> Iterator[np.ndarray]:
        text = '。' + text  # 防止漏第一句。
        text_seq, text_bert = get_phones_and_bert(text, language=language)

        # V2Pro/V2ProPlus: speaker embeddings are shared across all chunks.
        if prompt_encoder is not None:
            prompt_audio.update_global_emb(prompt_encoder=prompt_encoder)
        is_pro = prompt_encoder is not None

        hist = _carry_history(prompt_audio)
        if hist:
            _trim_history(hist)
        tok_iter: Iterator[tuple[np.ndarray, bool]]
        if static_engine is not None:
            tok_iter = self._t2s_stream_static(
                static_engine, ref_seq=prompt_audio.phonemes_seq,
                ref_bert=prompt_audio.text_bert, text_seq=text_seq,
                text_bert=text_bert, ssl_content=prompt_audio.ssl_content,
                encoder=encoder, first_stage_decoder=first_stage_decoder,
                chunk_tokens=min_chunk_tokens, hist=hist)
        elif stage_decoder is not None:
            tok_iter = self._t2s_stream(
                ref_seq=prompt_audio.phonemes_seq,
                ref_bert=prompt_audio.text_bert,
                text_seq=text_seq,
                text_bert=text_bert,
                ssl_content=prompt_audio.ssl_content,
                encoder=encoder,
                first_stage_decoder=first_stage_decoder,
                stage_decoder=stage_decoder,
                chunk_tokens=min_chunk_tokens, hist=hist)
        else:
            # static mode skips loading the stage decoder; if the static
            # engine also failed to build there is no t2s path at all
            raise RuntimeError(
                "no t2s decoder available: static engine failed to build "
                "and the stage decoder session was not loaded")

        # pass-through wrapper: remember the utterance's tokens when the
        # generation ran to completion (not aborted), then store them as
        # carry context for the next sentence
        def _recording(base_iter):
            generated = None
            try:
                for tokens, is_final in base_iter:
                    if tokens is not None:
                        generated = tokens
                    yield tokens, is_final
            finally:
                if hist is not None and generated is not None and not self.stop_event.is_set():
                    print(f"[genie] carry: +{len(generated.flatten())} tokens "
                          f"(history {sum(len(e['tokens']) for e in hist)}/{HISTORY_MAX_TOKENS})", flush=True)
                    _append_history(hist, text_seq, text_bert, generated.flatten().tolist())
                elif hist is not None:
                    print(f"[genie] carry store skipped: generated={'None' if generated is None else len(generated.flatten())} tokens, "
                          f"stopped={self.stop_event.is_set()}", flush=True)
                    print(f"[genie] carry history: {sum(len(e['tokens']) for e in hist)} tokens "
                          f"from {len(hist)} utterance(s)", flush=True)

        tok_iter = _recording(tok_iter)

        # incremental vocoder: when overlap is enabled, each chunk is decoded
        # with only the last N tokens of context instead of the whole prefix,
        # making per-chunk cost O(overlap + chunk) instead of O(prefix)
        ctx_n = max(0, int(vocoder_overlap_tokens))
        prev_tokens: Optional[np.ndarray] = None
        live_pieces: List[np.ndarray] = []  # dbg: what we actually emitted

        def decode(tokens: np.ndarray) -> Optional[np.ndarray]:
            nonlocal prev_tokens
            if tokens is None or tokens.shape[-1] == 0:
                return None
            # the t2s stream yields CUMULATIVE prefixes. Full-prefix mode
            # (ctx_n == 0) re-decodes the whole prefix so the emitted stream
            # stays sample-exact. Overlap mode decodes only a small context
            # tail plus the tokens past what was already decoded, keeping
            # per-chunk vocoder cost flat.
            if ctx_n and prev_tokens is not None:
                new_only = tokens[:, :, prev_tokens.shape[-1]:]
                feed = np.concatenate(
                    [prev_tokens[:, :, -ctx_n:], new_only], axis=-1)
            else:
                feed = tokens
            if dbg:
                print(f"[t2s-stream] voc feed={int(feed.shape[-1])}tok "
                      f"(ctx_n={ctx_n})", flush=True)
            feed_dict = {"text_seq": text_seq, "pred_semantic": feed}
            if is_pro:
                feed_dict["ge"] = prompt_audio.global_emb
                feed_dict["ge_advanced"] = prompt_audio.global_emb_advanced
            else:
                feed_dict["ref_audio"] = prompt_audio.audio_32k
            audio = vocoder.run(None, feed_dict)[0]
            prev_tokens = tokens  # cumulative: already contains everything
            return audio

        sent = 0      # samples of the ideal signal already emitted
        tail = None   # held-back samples (fade window) not yet emitted
        ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        dbg = os.environ.get('GENIE_T2S_DEBUG', '0') == '1'
        import time as _time
        t_prev = _time.perf_counter()

        for tokens, is_final in tok_iter:
            if self.stop_event.is_set():
                return
            # with overlap decode the context segment (always exactly
            # ctx_n * 1280 samples - the vocoder adds only tail padding) is
            # re-decoded context that must never be emitted again
            if ctx_n and prev_tokens is not None:
                sent = ctx_n * 1280
            t_tok = _time.perf_counter()
            full = decode(tokens)
            t_dec = _time.perf_counter()
            if dbg:
                n_tok = 0 if tokens is None else int(tokens.shape[-1])
                print(f"[t2s-stream] chunk +{n_tok}tok "
                      f"t2s_gap={t_tok - t_prev:.2f}s voc={t_dec - t_tok:.2f}s", flush=True)
            t_prev = t_dec
            if full is None:
                continue
            if full.ndim == 2:
                full = full[0]
            length = full.shape[-1]
            if is_final:
                out = None
                if tail is None:
                    if length > sent:
                        out = full[sent:]
                elif length > sent + fade_samples:
                    blended = tail * (1.0 - ramp) + full[sent:sent + fade_samples] * ramp
                    out = np.concatenate([blended, full[sent + fade_samples:]], axis=-1)
                else:
                    # Degenerate final chunk: flush without blending.
                    out = np.concatenate([tail, full[sent:]], axis=-1)
                if out is not None:
                    if dbg and ctx_n:
                        live_pieces.append(out)
                        # decode the WHOLE utterance prefix once for an offline
                        # alignment/residual comparison against the live output
                        feed_dict = {"text_seq": text_seq, "pred_semantic": prev_tokens}
                        if is_pro:
                            feed_dict["ge"] = prompt_audio.global_emb
                            feed_dict["ge_advanced"] = prompt_audio.global_emb_advanced
                        else:
                            feed_dict["ref_audio"] = prompt_audio.audio_32k
                        ref_full = vocoder.run(None, feed_dict)[0]
                        if ref_full.ndim == 2:
                            ref_full = ref_full[0]
                        np.savez('_dbg_overlap_check.npz',
                                 live=np.concatenate(live_pieces), full=ref_full,
                                 ctx_n=ctx_n)
                    yield out
                return

            new_end = length - fade_samples
            if new_end <= sent:
                continue  # not enough new audio to emit yet
            if tail is None:
                out = full[sent:new_end]
            else:
                blended = tail * (1.0 - ramp) + full[sent:sent + fade_samples] * ramp
                out = np.concatenate([blended, full[sent + fade_samples:new_end]], axis=-1)
            if dbg:
                live_pieces.append(out)
            yield out
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
            hist: Optional[list] = None,
    ) -> Iterator[tuple[np.ndarray, bool]]:
        """Static CUDA-graph decode: prefill via the dynamic first-stage graph,
        then per-step replay of t2s_static.onnx; sampling in numpy."""
        dbg = os.environ.get('GENIE_T2S_DEBUG', '0') == '1'
        eos = engine.vocab - 1
        if hist:
            text_seq, text_bert = _extend_inputs(text_seq=text_seq,
                                             text_bert=text_bert, hist=hist)
        if dbg:
            print(f"[t2s-static] hist={len(hist) if hist else 0} "
                  f"text_len={int(text_seq.shape[-1])} eos_id={eos}", flush=True)
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
        # token carry: append the history utterances' semantic tokens to the
        # prompt so the next sentence starts in the same prosodic context
        # (same treatment as the dynamic paths)
        if hist:
            hist_tok = np.asarray(
                [[t for e in hist for t in e["tokens"]]], dtype=np.int64)
            prompts = np.concatenate([prompts, hist_tok], axis=1)
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
        if dbg:
            print(f"[t2s-static] y_shape={y.shape} P={P} past_len={past_len} "
                  f"first_token={token} prompt_tokens={y_tokens[:min(P, 6)]}...",
                  flush=True)
        if token == eos:
            if dbg:
                print(f"[t2s-static] STOP eos-on-prefill: P={P} "
                      f"prefill_tokens={len(y_tokens) - 1}", flush=True)
            yield np.asarray(y_tokens[P:-1], dtype=np.int64)[None, None, :], True
            return

        # hard iteration cap only (same spirit as the original infer loop's
        # max-length bound); no content-based force-stop guards - termination
        # is the model's own EOS, exactly like the dynamic path
        token_budget = int(text_seq.shape[-1]) * 6 + 60
        for _ in range(min(engine.max_len - kv_pos - 2, token_budget)):
            if self.stop_event.is_set():
                return
            logits = engine.step(token, pe_pos, kv_pos)
            nxt = _sample_numpy(logits, y_tokens)
            gen_idx += 1
            if dbg and gen_idx <= 8:
                top5 = np.argsort(logits)[::-1][:5]
                print(f"[t2s-static] step{gen_idx} kv_pos={kv_pos} pe_pos={pe_pos} "
                      f"sampled={nxt} top5={top5.tolist()}", flush=True)
            if nxt == eos:
                if dbg:
                    print(f"[t2s-static] STOP eos at gen_idx={gen_idx} "
                          f"generated={len(y_tokens) - P} tokens", flush=True)
                yield np.asarray(y_tokens[P:], dtype=np.int64)[None, None, :], True
                return
            y_tokens.append(nxt)
            if gen_idx % chunk_tokens == 0:
                yield np.asarray(y_tokens[P:], dtype=np.int64)[None, None, :], False
            token = nxt
            pe_pos += 1
            kv_pos += 1
        if dbg:
            print(f"[t2s-static] STOP loop-exhausted: generated={len(y_tokens) - P} tokens", flush=True)
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
            hist: Optional[list] = None,
    ) -> Iterator[tuple[np.ndarray, bool]]:
        """Incremental T2S decoding; yields (generated token prefix, is_final)."""
        if hist:
            text_seq, text_bert = _extend_inputs(text_seq=text_seq,
                                                 text_bert=text_bert, hist=hist)
        if 'CUDAExecutionProvider' in (stage_decoder.get_providers() or []):
            yield from self._t2s_stream_cuda(
                ref_seq, ref_bert, text_seq, text_bert, ssl_content,
                encoder, first_stage_decoder, stage_decoder, chunk_tokens, hist)
        else:
            yield from self._t2s_stream_cpu(
                ref_seq, ref_bert, text_seq, text_bert, ssl_content,
                encoder, first_stage_decoder, stage_decoder, chunk_tokens, hist)

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
            hist: Optional[list] = None,
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

        if hist:
            text_seq, text_bert = _extend_inputs(text_seq=text_seq,
                                                 text_bert=text_bert, hist=hist)
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
        if hist:
            hist_tok = np.asarray(
                [[t for e in hist for t in e["tokens"]]], dtype=np.int64)
            prompts = np.concatenate([prompts, hist_tok], axis=1)
        if hist:
            print(f"[genie] static carry: prompts={prompts.shape[1]} "
                  f"(anchor {prompts.shape[1] - hist_tok.shape[1]} + hist {hist_tok.shape[1]}), "
                  f"text_len={text_seq.shape[-1]}", flush=True)
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
            hist: Optional[list] = None,
    ) -> Iterator[tuple[np.ndarray, bool]]:
        """Incremental T2S decoding; yields (generated token prefix, is_final)."""
        dbg = os.environ.get('GENIE_T2S_DEBUG', '0') == '1'
        if hist:
            text_seq, text_bert = _extend_inputs(text_seq=text_seq,
                                                 text_bert=text_bert, hist=hist)
        if dbg:
            print(f"[t2s-cpu] hist={len(hist) if hist else 0} "
                  f"hist_toks={[len(e['tokens']) for e in hist] if hist else []} "
                  f"text_len={int(text_seq.shape[-1])}", flush=True)
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

        if hist:
            hist_tok = np.asarray(
                [[t for e in hist for t in e["tokens"]]], dtype=np.int64)
            prompts = np.concatenate([prompts, hist_tok], axis=1)
        if dbg:
            print(f"[t2s-cpu] prompts={prompts.shape[1]} "
                  f"text_len={int(text_seq.shape[-1])}", flush=True)

        # First Stage Decoder
        y, y_emb, *present_key_values = first_stage_decoder.run(
            None, {"x": x, "prompts": prompts}
        )
        if dbg:
            print(f"[t2s-cpu] y_shape={y.shape} prefill "
                  f"(ref+hist={prompts.shape[1]})", flush=True)

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
        if dbg:
            print(f"[t2s-cpu] done: idx={idx} steps (graph stop signal)", flush=True)
        y[0, -1] = 0
        tokens = np.expand_dims(y[:, -idx:], axis=0)
        # Trim anything at/after EOS (same rule as the non-streaming path).
        eos_indices = np.where(tokens >= 1024)
        if len(eos_indices[0]) > 0:
            tokens = tokens[..., :eos_indices[-1][0]]
        yield tokens, True


tts_client: GENIE = GENIE()
