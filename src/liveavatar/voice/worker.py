# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Character-specific TTS worker with fixed weights and async streaming.

An ``NvcWorker`` wraps a GPT-SoVITS ``TTS`` instance whose weights are loaded
once at construction and **never switched** afterwards. This is the core
guarantee that eliminates the P0 cross-talk risk: unlike the global
``tts_pipeline`` in ``api_v2.py`` which mutates weights via
``activate_character()``, each worker is immutable.

Key design
----------
1. **Immutability**: ``char_id``, T2S weights, VITS weights and reference
   audio are set at construction and never change.
2. **Exclusive inference**: An ``asyncio.Lock`` serialises ``synthesize_stream``
   calls — GPT-SoVITS is not thread-safe (it mutates ``prompt_cache`` and
   ``stop_flag`` during inference).
3. **Async streaming**: The synchronous ``TTS.run()`` generator is consumed
   in a thread via ``asyncio.to_thread`` so the event loop is never blocked.
   Chunks are yielded to the caller as canonical 16 kHz mono S16LE bytes.
4. **Cooperative cancellation**: A ``CancelToken`` is checked between chunks;
   when a confirmed interrupt bumps the epoch, the token is set and the
   generator stops after the current chunk (≤ one chunk of latency).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from .lease import CancelToken

logger = logging.getLogger("liveavatar.voice.worker")


@dataclass(slots=True)
class NvcWorkerStats:
    """Lifetime counters for a worker."""

    syntheses_started: int = 0
    syntheses_completed: int = 0
    syntheses_cancelled: int = 0
    syntheses_failed: int = 0
    chunks_emitted: int = 0
    bytes_emitted: int = 0
    total_inference_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "syntheses_started": self.syntheses_started,
            "syntheses_completed": self.syntheses_completed,
            "syntheses_cancelled": self.syntheses_cancelled,
            "syntheses_failed": self.syntheses_failed,
            "chunks_emitted": self.chunks_emitted,
            "bytes_emitted": self.bytes_emitted,
            "total_inference_s": round(self.total_inference_s, 3),
        }


@dataclass(slots=True)
class CharacterAssets:
    """Resolved asset paths for a character.

    Follows the project convention: each character has an independent folder
    containing two models (GPT ``.ckpt`` + SoVITS ``.pth``) and one reference
    audio (``.wav``). The reference text is the audio filename (without
    extension), eliminating the need for a separate JSON metadata file.
    """

    char_id: str
    gpt_path: str
    sovits_path: str
    ref_audio_path: str
    ref_text: str  # = ref_audio filename without extension
    language: str = "zh"

    def to_dict(self) -> dict:
        return {
            "char_id": self.char_id,
            "gpt_path": self.gpt_path,
            "sovits_path": self.sovits_path,
            "ref_audio_path": self.ref_audio_path,
            "ref_text": self.ref_text,
            "language": self.language,
        }


class NvcWorker:
    """Character-pinned TTS worker with async streaming synthesis.

    Parameters
    ----------
    assets : CharacterAssets
        Resolved character asset paths (used for identification and logging).
    tts : Any
        A GPT-SoVITS ``TTS`` instance (or test double) with weights already
        loaded and ``set_ref_audio`` called. Must implement::
            ``run(inputs: dict) -> Generator[tuple[int, np.ndarray]]``
    target_sample_rate : int
        Canonical output sample rate (16000 Hz for the realtime pipeline).
    """

    def __init__(
        self,
        assets: CharacterAssets,
        tts: Any,
        *,
        target_sample_rate: int = 16000,
    ) -> None:
        self.assets = assets
        self.char_id = assets.char_id
        self._tts = tts
        self._target_sr = target_sample_rate
        self._infer_lock: asyncio.Lock = asyncio.Lock()
        self._busy: bool = False
        self._created_at = time.monotonic()
        self.stats = NvcWorkerStats()

    # ------------------------------------------------------------------ props

    @property
    def busy(self) -> bool:
        """True while a ``synthesize_stream`` call is in progress."""
        return self._busy

    @property
    def idle(self) -> bool:
        """True when no synthesis is running (available for new work)."""
        return not self._busy

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self._created_at

    # ---------------------------------------------------------- synthesize

    async def synthesize_stream(
        self,
        text: str,
        *,
        cancel_token: CancelToken | None = None,
        text_lang: str = "zh",
        speed_factor: float = 1.0,
        top_k: int = 15,
        top_p: float = 1.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.35,
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesise ``text``, yielding canonical PCM S16LE chunks.

        Parameters
        ----------
        text : str
            Text to synthesise.
        cancel_token : CancelToken | None
            If provided, the generator checks ``cancelled`` before each chunk
            and stops early when set.
        text_lang : str
            Language of ``text`` (``"zh"``, ``"en"``, ``"ja"``, ...).
        speed_factor : float
            Speech speed multiplier (1.0 = normal).
        top_k, top_p, temperature, repetition_penalty
            Sampling parameters forwarded to the TTS engine.

        Yields
        ------
        bytes
            Canonical 16 kHz mono S16LE PCM (variable chunk size).
        """
        token = cancel_token or CancelToken()
        async with self._infer_lock:
            self._busy = True
            self.stats.syntheses_started += 1
            start = time.monotonic()
            try:
                req = self._build_request(
                    text=text,
                    text_lang=text_lang,
                    speed_factor=speed_factor,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                )
                # ``TTS.run`` is a synchronous generator that blocks on
                # torch inference. Run each ``next()`` in a thread so the
                # event loop stays responsive.
                gen = self._tts.run(req)

                def _next_chunk() -> tuple[int, np.ndarray] | None:
                    try:
                        return next(gen)
                    except StopIteration:
                        return None

                while True:
                    if token.cancelled:
                        self.stats.syntheses_cancelled += 1
                        logger.info(
                            "synthesize_cancelled",
                            extra={"char_id": self.char_id, "text_len": len(text)},
                        )
                        break
                    result = await asyncio.to_thread(_next_chunk)
                    if result is None:
                        self.stats.syntheses_completed += 1
                        break
                    if token.cancelled:
                        self.stats.syntheses_cancelled += 1
                        break
                    sr, audio_np = result
                    pcm = self._to_canonical_pcm(audio_np, sr)
                    self.stats.chunks_emitted += 1
                    self.stats.bytes_emitted += len(pcm)
                    yield pcm
            except Exception:
                self.stats.syntheses_failed += 1
                logger.exception(
                    "synthesize_failed",
                    extra={"char_id": self.char_id, "text_len": len(text)},
                )
                raise
            finally:
                self.stats.total_inference_s += time.monotonic() - start
                self._busy = False

    # -------------------------------------------------------- PCM convert

    def _to_canonical_pcm(self, audio_np: np.ndarray, source_sr: int) -> bytes:
        """Convert TTS output to canonical 16 kHz mono S16LE bytes.

        Parameters
        ----------
        audio_np : np.ndarray
            Float32 audio in range [-1.0, 1.0] (shape: ``(samples,)`` or
            ``(1, samples)``).
        source_sr : int
            Sample rate of ``audio_np`` (typically 32000 for v2Pro).

        Returns
        -------
        bytes
            Little-endian int16 PCM at ``target_sample_rate``, mono.
        """
        # Flatten to 1-D.
        audio = cast(np.ndarray, np.asarray(audio_np, dtype=np.float32).reshape(-1))
        if audio.size == 0:
            return b""

        # Resample if the source rate differs from target.
        if source_sr != self._target_sr:
            audio = self._resample(audio, source_sr, self._target_sr)

        # float32 [-1, 1] → int16 [-32768, 32767]
        # GPT-SoVITS 合成输出振幅可能远超 [-1,1]（实测可达 ±3 以上），直接
        # clip 会产生大量 ±32767 的削波方波：波形失去动态范围，Whisper 特征
        # 饱和，MuseTalk 口型不随语音变化。先按峰值归一化到 0.95 再限幅。
        peak = float(np.abs(audio).max())
        if peak > 1.0 and peak > 1e-8:
            audio = audio / peak * 0.95
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)
        return audio_int16.tobytes()  # S16LE on little-endian platforms

    @staticmethod
    def _resample(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int,
    ) -> np.ndarray:
        """Linear-interpolation resampling (no external dependencies).

        Good enough for TTS output (speech bandwidth << Nyquist). For
        production-grade quality, replace with ``librosa.resample`` or
        ``torchaudio.transforms.Resample``.
        """
        if orig_sr == target_sr or len(audio) == 0:
            return audio
        n_out = int(len(audio) * target_sr / orig_sr)
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)
        indices = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    # --------------------------------------------------------- request

    def _build_request(
        self,
        *,
        text: str,
        text_lang: str,
        speed_factor: float,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
    ) -> dict:
        """Build the input dict for ``TTS.run()`` with streaming enabled.

        Uses ``streaming_mode=True`` + ``fixed_length_chunk=True`` (mode 3)
        for the fastest first-chunk response. The reference audio and prompt
        text are pre-set on the TTS instance, so we pass them again for
        completeness but the TTS engine reuses the cached prompt.
        """
        a = self.assets
        return {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": a.ref_audio_path,
            "aux_ref_audio_paths": [],
            "prompt_text": a.ref_text,
            "prompt_lang": a.language,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": "auto",
            "batch_size": 1,
            "batch_threshold": 0.75,
            "split_bucket": True,
            "speed_factor": speed_factor,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "raw",
            "streaming_mode": True,
            "parallel_infer": True,
            "repetition_penalty": repetition_penalty,
            "return_fragment": False,
            "fixed_length_chunk": True,
            "overlap_length": 2,
            "min_chunk_length": 16,
        }

    # ----------------------------------------------------------- stats

    def to_dict(self) -> dict:
        return {
            "char_id": self.char_id,
            "busy": self._busy,
            "uptime_s": round(self.uptime_s, 1),
            "stats": self.stats.to_dict(),
            "assets": self.assets.to_dict(),
        }
