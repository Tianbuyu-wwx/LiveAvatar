# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""TextSource — LLM spoke interface (预留挂载点).

A TextSource turns an utterance (e.g. ASR final text) into a stream of
text chunks. The session runtime consumes the stream, segments it into
sentence-bounded pieces and dispatches each piece to the TTS spoke, so
first-audio latency overlaps LLM generation.

Implement the :class:`TextSource` protocol (structural — no inheritance
required) to plug in any backend. :class:`OpenAIChatTextSource` is a
ready-to-use client for OpenAI-compatible APIs (DeepSeek, Qwen/DashScope
compatible mode, vLLM, Ollama, ...).

Example::

    source = OpenAIChatTextSource(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-...",
        model="deepseek-chat",
        system_prompt="你是数字人助手，回答保持口语化、简短。",
    )
    async for piece in sentence_stream(source.stream_text("你好")):
        await worker.dispatch_text(piece, epoch)   # → TTS → Avatar
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable

_SENTENCE_END = re.compile(r"([。！？；!?;]|\.\s|\n)")


@runtime_checkable
class TextSource(Protocol):
    """Turn an utterance into a stream of incremental text chunks."""

    name: str

    def stream_text(
        self,
        utterance: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]: ...


def split_sentences(text: str) -> tuple[str | None, str]:
    """Split ``text`` at sentence boundaries.

    Returns ``(complete_piece, remainder)``. ``complete_piece`` is ``None``
    when no sentence boundary was found yet.
    """
    match: re.Match[str] | None = None
    for _m in _SENTENCE_END.finditer(text):
        match = _m
    if match is None:
        return None, text
    end = match.end()
    piece, remainder = text[:end], text[end:]
    return (piece if piece.strip() else None), remainder


async def sentence_stream(
    chunks: AsyncGenerator[str, None],
    *,
    min_piece_chars: int = 4,
) -> AsyncGenerator[str, None]:
    """Merge raw text chunks into sentence-bounded pieces for TTS dispatch.

    Flushes the remainder as a final piece when the stream ends, so short
    answers ("好") still reach TTS.
    """
    buf = ""
    async for chunk in chunks:
        buf += chunk
        while True:
            piece, buf = split_sentences(buf)
            if piece is None:
                break
            if len(piece) < min_piece_chars and not buf:
                break
            yield piece
    if buf.strip():
        yield buf


class OpenAIChatTextSource:
    """Streaming client for OpenAI-compatible chat completion APIs.

    Works with DeepSeek, Qwen (DashScope compatible mode), vLLM, Ollama and
    any server implementing ``POST {base_url}/chat/completions`` with
    ``stream=true`` SSE responses.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        request_timeout: float = 60.0,
        name: str = "openai-chat",
        transport: Any = None,
    ) -> None:
        self.name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._timeout = request_timeout
        self._transport = transport  # httpx transport seam (tests)

    async def stream_text(
        self,
        utterance: str,
        *,
        history: list[dict[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield incremental delta texts from the chat completion stream."""
        import httpx2 as httpx

        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.extend(history or [])
        messages.append({"role": "user", "content": utterance})

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "temperature": self._temperature,
        }

        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            async with client.stream(
                "POST", f"{self._base_url}/chat/completions", json=body, headers=headers
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta: dict[str, Any] = choices[0].get("delta") or {}
                    text = delta.get("content")
                    if text:
                        yield text
