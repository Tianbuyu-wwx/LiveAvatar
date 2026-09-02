# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 LiveAvatar Contributors
# Commercial use requires a separate written license; see ../LICENSE.

"""Tests for the TextSource protocol (LLM spoke).

Covers: sentence splitting, sentence_stream merging, and the RealtimeWorker
LLM turn (ASR final → LLM stream → per-sentence TTS dispatch), including
history recording, error surfacing and epoch interruption. CPU-only.
"""

from __future__ import annotations

import asyncio
import json
import unittest

from liveavatar.runtime.fake_tts import FakeTts
from liveavatar.runtime.worker import RealtimeWorker
from liveavatar.text_source import (
    OpenAIChatTextSource,
    sentence_stream,
    split_sentences,
)


class _StubTextSource:
    """Deterministic TextSource that replays canned chunks."""

    name = "stub"

    def __init__(self, chunks: list[str], *, delay: float = 0.0) -> None:
        self._chunks = chunks
        self._delay = delay
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def stream_text(self, utterance: str, *, history=None):
        self.calls.append((utterance, list(history or [])))
        for chunk in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield chunk


# ── sentence splitting ──


class TestSplitSentences(unittest.TestCase):
    def test_chinese_boundary(self):
        piece, rest = split_sentences("你好。请讲题")
        self.assertEqual(piece, "你好。")
        self.assertEqual(rest, "请讲题")

    def test_no_boundary(self):
        piece, rest = split_sentences("还没有结束")
        self.assertIsNone(piece)
        self.assertEqual(rest, "还没有结束")

    def test_whitespace_only_piece_is_none(self):
        piece, rest = split_sentences("。abc")
        self.assertEqual(piece, "。")
        self.assertEqual(rest, "abc")


class TestSentenceStream(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, chunks, **kwargs):
        async def _gen():
            for c in chunks:
                yield c

        return [p async for p in sentence_stream(_gen(), **kwargs)]

    async def test_merges_into_sentences(self):
        # Short leading sentences flush immediately (first-audio latency).
        pieces = await self._collect(["你好。请讲", "题。谢谢"])
        self.assertEqual(pieces, ["你好。", "请讲题。", "谢谢"])

    async def test_flushes_short_remainder(self):
        pieces = await self._collect(["嗯"])
        self.assertEqual(pieces, ["嗯"])

    async def test_min_piece_chars(self):
        # "好。" is shorter than min_piece_chars but followed by more text.
        pieces = await self._collect(["好。接着说。"], min_piece_chars=4)
        self.assertEqual(pieces, ["好。接着说。"])

    async def test_openai_client_construction(self):
        client = OpenAIChatTextSource(
            base_url="https://example.com/v1/", model="m", api_key="k"
        )
        self.assertEqual(client._base_url, "https://example.com/v1")
        self.assertEqual(client.name, "openai-chat")


# ── OpenAI-compatible SSE client ──


def _sse_response(lines: list[str]):
    import httpx2 as httpx

    body = "\n".join(lines) + "\n"
    return httpx.Response(
        200, content=body.encode(), headers={"content-type": "text/event-stream"}
    )


class TestOpenAIChatTextSource(unittest.IsolatedAsyncioTestCase):
    def _source(self, handler, **kwargs):
        import httpx2 as httpx

        return OpenAIChatTextSource(
            base_url="https://llm.example/v1",
            api_key="sk-test",
            model="test-model",
            system_prompt="你是助手",
            transport=httpx.MockTransport(handler),
            **kwargs,
        )

    async def test_stream_text_parses_sse_deltas(self):
        import httpx2 as httpx

        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return _sse_response(
                [
                    'data: {"choices":[{"delta":{"content":"你好"}}]}',
                    "data: not-json",  # malformed → skipped
                    'data: {"choices":[]}',  # empty choices → skipped
                    'data: {"choices":[{"delta":{"content":"！"}}]}',
                    "data: [DONE]",
                    'data: {"choices":[{"delta":{"content":"after"}}]}',  # post-DONE
                ]
            )

        src = self._source(handler)
        out = [
            t
            async for t in src.stream_text(
                "你好", history=[{"role": "user", "content": "prev"}]
            )
        ]
        self.assertEqual(out, ["你好", "！"])
        self.assertEqual(captured["auth"], "Bearer sk-test")
        self.assertEqual(captured["url"], "https://llm.example/v1/chat/completions")
        body = captured["body"]
        self.assertTrue(body["stream"])
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["messages"][0], {"role": "system", "content": "你是助手"})
        self.assertEqual(body["messages"][1], {"role": "user", "content": "prev"})
        self.assertEqual(body["messages"][-1], {"role": "user", "content": "你好"})

    async def test_stream_text_no_auth_without_api_key(self):
        import httpx2 as httpx

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertEqual(body["messages"][0]["role"], "user")
            self.assertNotIn("system", [m["role"] for m in body["messages"]])
            self.assertIsNone(request.headers.get("authorization"))
            return _sse_response(['data: {"choices":[{"delta":{}}]}', "data: [DONE]"])

        src = OpenAIChatTextSource(
            base_url="https://x.example/v1/", model="m",
            transport=httpx.MockTransport(handler),
        )
        out = [t async for t in src.stream_text("hi")]
        self.assertEqual(out, [])

    async def test_stream_text_raises_on_http_error(self):
        import httpx2 as httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")

        src = self._source(handler)
        with self.assertRaises(httpx.HTTPStatusError):
            async for _ in src.stream_text("hi"):
                pass


# ── worker LLM turn ──


def _loud_frame(session_id: str, seq: int, pts_us: int):
    from liveavatar.audio_in.frame import PCMFrame

    return PCMFrame(
        session_id=session_id,
        epoch=0,
        seq=seq,
        pts_us=pts_us,
        deadline_us=pts_us + 20000,
        pcm_s16le=b"\x00\x40" * 320,  # 20ms @ 16kHz, loud
    )


def _silent_frame(session_id: str, seq: int, pts_us: int):
    from liveavatar.audio_in.frame import PCMFrame

    return PCMFrame(
        session_id=session_id,
        epoch=0,
        seq=seq,
        pts_us=pts_us,
        deadline_us=pts_us + 20000,
        pcm_s16le=b"\x00\x00" * 320,
    )


class _OutputCollector:
    """Polls the worker output queue with a deadline."""

    def __init__(self, worker: RealtimeWorker) -> None:
        self.worker = worker

    async def collect(
        self, event_type: str | None, count: int, timeout: float = 5.0
    ) -> list[dict]:
        out: list[dict] = []
        deadline = asyncio.get_running_loop().time() + timeout
        while len(out) < count:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            event = self.worker.output_queue.try_dequeue()
            if event is None:
                await asyncio.sleep(0.01)
                continue
            et = event.get("event_type")
            if event_type is None or et == event_type:
                out.append(event)
        return out


class TestWorkerLlmTurn(unittest.IsolatedAsyncioTestCase):
    async def test_asr_final_streams_through_llm_to_tts(self):
        worker = RealtimeWorker(
            "s1",
            tts=FakeTts(),
            text_source=_StubTextSource(["你好。请讲", "题。谢谢"]),
        )
        await worker.start()
        try:
            collector = _OutputCollector(worker)
            # 10 loud frames → partial; 5+ silent frames → final.
            for i in range(10):
                await worker.push_frame(_loud_frame("s1", i, i * 20000))
            for i in range(10, 35):
                await worker.push_frame(_silent_frame("s1", i, i * 20000))

            tts_events = await collector.collect("tts_audio", 3)
            self.assertEqual(len(tts_events), 3)
            # One TTS call per sentence piece, in stream order.
            texts = [s.text for s in worker.tts.active_segments]
            self.assertEqual(texts, ["你好。", "请讲题。", "谢谢"])

            # The completed exchange is appended to the rolling history.
            self.assertEqual(len(worker._history), 2)
            self.assertEqual(worker._history[0]["role"], "user")
            self.assertEqual(worker._history[1]["role"], "assistant")
            self.assertEqual(worker._history[1]["content"], "你好。请讲题。谢谢")
        finally:
            await worker.stop()

    async def test_history_capped(self):
        worker = RealtimeWorker(
            "s1", tts=FakeTts(), text_source=_StubTextSource(["好。"]), history_limit=4
        )
        await worker.start()
        try:
            collector = _OutputCollector(worker)
            for turn in range(3):
                for i in range(10):
                    await worker.push_frame(_loud_frame("s1", i, i * 20000))
                for i in range(10, 35):
                    await worker.push_frame(_silent_frame("s1", i, i * 20000))
                await collector.collect("tts_audio", 1)
                # Wait for the LLM task to finish appending history.
                for _ in range(100):
                    if len(worker._history) >= 2 * (turn + 1):
                        break
                    await asyncio.sleep(0.01)
            self.assertLessEqual(len(worker._history), worker.history_limit)
            self.assertEqual(worker.history_limit, 4)
        finally:
            await worker.stop()

    async def test_llm_error_becomes_error_event(self):
        class _BoomSource(_StubTextSource):
            async def stream_text(self, utterance, *, history=None):
                raise RuntimeError("llm down")
                yield  # pragma: no cover

        worker = RealtimeWorker("s1", tts=FakeTts(), text_source=_BoomSource([]))
        await worker.start()
        try:
            collector = _OutputCollector(worker)
            for i in range(10):
                await worker.push_frame(_loud_frame("s1", i, i * 20000))
            for i in range(10, 35):
                await worker.push_frame(_silent_frame("s1", i, i * 20000))

            errors = await collector.collect("error", 1)
            payload = errors[0]["payload"]["error_event"]
            self.assertIn("llm down", payload["message"])
            # Nothing was synthesized and history stays empty.
            self.assertEqual(worker.tts.active_segments, [])
            self.assertEqual(worker._history, [])
        finally:
            await worker.stop()

    async def test_epoch_interrupt_stops_llm_stream(self):
        source = _StubTextSource(["第一句。", "第二句。", "第三句。"], delay=0.05)
        worker = RealtimeWorker("s1", tts=FakeTts(), text_source=source)
        await worker.start()
        try:
            collector = _OutputCollector(worker)
            for i in range(10):
                await worker.push_frame(_loud_frame("s1", i, i * 20000))
            for i in range(10, 35):
                await worker.push_frame(_silent_frame("s1", i, i * 20000))

            # Wait for the first TTS segment, then barge-in.
            await collector.collect("tts_audio", 1)
            new_epoch = worker.advance_epoch()
            self.assertEqual(new_epoch, 1)

            # No further tts_audio for the old epoch may arrive afterwards.
            await asyncio.sleep(0.2)
            stale = []
            while True:
                event = worker.output_queue.try_dequeue()
                if event is None:
                    break
                if event.get("event_type") == "tts_audio":
                    payload = event["payload"]["tts_audio_event"]
                    self.assertEqual(payload["epoch"], 1)
                    stale.append(payload)
            for payload in stale:
                self.assertGreaterEqual(payload["epoch"], 1)
        finally:
            await worker.stop()

    async def test_echo_mode_without_text_source(self):
        worker = RealtimeWorker("s1", tts=FakeTts())
        await worker.start()
        try:
            collector = _OutputCollector(worker)
            for i in range(10):
                await worker.push_frame(_loud_frame("s1", i, i * 20000))
            for i in range(10, 35):
                await worker.push_frame(_silent_frame("s1", i, i * 20000))

            tts_events = await collector.collect("tts_audio", 1)
            self.assertEqual(len(tts_events), 1)
            # Echo: the full ASR-final text reaches TTS as one call
            # (ScriptedAsr emits "请讲题" after 10 loud + silent frames).
            self.assertEqual(worker.tts.active_segments[0].text, "请讲题")
            self.assertEqual(worker._history, [])
        finally:
            await worker.stop()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
