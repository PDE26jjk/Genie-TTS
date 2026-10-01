import onnxruntime as ort
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
    ) -> Iterator[np.ndarray]:
        text = '。' + text  # 防止漏第一句。
        text_seq, text_bert = get_phones_and_bert(text, language=language)

        # V2Pro/V2ProPlus: speaker embeddings are shared across all chunks.
        if prompt_encoder is not None:
            prompt_audio.update_global_emb(prompt_encoder=prompt_encoder)
        is_pro = prompt_encoder is not None

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

        for tokens, is_final in self._t2s_stream(
                ref_seq=prompt_audio.phonemes_seq,
                ref_bert=prompt_audio.text_bert,
                text_seq=text_seq,
                text_bert=text_bert,
                ssl_content=prompt_audio.ssl_content,
                encoder=encoder,
                first_stage_decoder=first_stage_decoder,
                stage_decoder=stage_decoder,
                chunk_tokens=min_chunk_tokens,
        ):
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
