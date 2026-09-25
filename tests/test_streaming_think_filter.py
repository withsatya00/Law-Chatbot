"""Streaming think-tag leak (Priority 1 follow-up QA session, 2026-09-11).

Live repro: the non-streaming `/chat` fix (`_strip_inline_thinking`, applied
once the full response is assembled) does NOT protect `/chat/stream`, because
`ChatService.answer_stream` forwards each `self.llm.stream(...)` chunk to the
client as a `"token"` SSE event the instant it arrives -- by the time a later
chunk contains `</think>`, the reasoning text before it has already been sent.
A post-hoc strip on the final assembled string cannot un-send tokens the
client already rendered. `_InlineThinkStreamFilter` fixes this by filtering
BEFORE a chunk is yielded, not after the stream ends.

Uses synthetic reasoning markers only (`<think>...</think>` around made-up
text) -- never reproduces the actual leaked system-prompt content from the
live incident.
"""

import asyncio
from collections.abc import AsyncIterator

import pytest

from app.llm.base import ChatMessage, LLMProvider, LLMResponse
from app.llm.resilient import ResilientLLMProvider, _InlineThinkStreamFilter


def _feed_all(pieces: list[str]) -> str:
    """Feeds `pieces` one at a time (as separate stream chunks) through a
    fresh filter and returns everything ever emitted, in order -- i.e. what
    a client watching the live stream would actually have seen."""
    filt = _InlineThinkStreamFilter()
    out = []
    for piece in pieces:
        emitted = filt.feed(piece)
        if emitted:
            out.append(emitted)
    tail = filt.flush()
    if tail:
        out.append(tail)
    return "".join(out)


# --- 1. Complete <think> blocks -------------------------------------------

def test_a_complete_think_block_in_one_chunk_is_never_emitted() -> None:
    result = _feed_all(["<think>synthetic internal reasoning</think>Here is the real answer."])
    assert result == "Here is the real answer."
    assert "synthetic internal reasoning" not in result


def test_a_complete_think_block_split_into_many_small_chunks() -> None:
    text = "<think>synthetic internal reasoning</think>Here is the real answer."
    result = _feed_all(list(text))  # one character per "chunk" -- worst case
    assert result == "Here is the real answer."


# --- 2. Tags split across streaming chunks ---------------------------------

def test_opening_tag_split_across_chunk_boundary() -> None:
    # "<think>" split right in the middle of the tag itself.
    result = _feed_all(["Intro. <thi", "nk>synthetic reasoning</think> Real answer."])
    assert result == "Intro.  Real answer."
    assert "synthetic reasoning" not in result


def test_closing_tag_split_across_chunk_boundary() -> None:
    result = _feed_all(["<think>synthetic reasoning</thi", "nk> Real answer."])
    assert result == " Real answer."
    assert "synthetic reasoning" not in result


# --- 3. Reasoning content split across chunks ------------------------------

def test_reasoning_content_itself_split_across_many_chunks_never_leaks() -> None:
    pieces = ["<think>", "step one, ", "step two, ", "step three", "</think>", "Final answer only."]
    result = _feed_all(pieces)
    assert result == "Final answer only."
    for leaked in ("step one", "step two", "step three"):
        assert leaked not in result


# --- 4. Incomplete / unclosed tags ------------------------------------------

def test_unclosed_think_tag_at_end_of_stream_emits_nothing_after_it() -> None:
    # e.g. truncated at a provider length cap -- the tag never closes.
    result = _feed_all(["Before. <think>reasoning that never closes because the stream just ends"])
    assert result == "Before. "


def test_unclosed_think_tag_is_the_entire_stream() -> None:
    result = _feed_all(["<think>only ever reasoning, cut off mid-thought"])
    assert result == ""


# --- 5. Multiple reasoning blocks ------------------------------------------

def test_multiple_separate_think_blocks_are_all_stripped() -> None:
    text = (
        "<think>first synthetic reasoning block</think>"
        "Part one of the answer. "
        "<think>second synthetic reasoning block</think>"
        "Part two of the answer."
    )
    result = _feed_all([text])
    assert result == "Part one of the answer. Part two of the answer."
    assert "reasoning block" not in result


def test_multiple_think_blocks_split_arbitrarily_across_chunks() -> None:
    text = (
        "<think>first block</think>Answer A. <think>second block</think>Answer B."
    )
    # Split into 7-character chunks -- deliberately not aligned to tag boundaries.
    pieces = [text[i : i + 7] for i in range(0, len(text), 7)]
    result = _feed_all(pieces)
    assert result == "Answer A. Answer B."


# --- 6. Reasoning-only response --------------------------------------------

def test_response_that_is_pure_reasoning_yields_nothing_to_the_client() -> None:
    result = _feed_all(["<think>this is the entire model output, no real answer at all</think>"])
    assert result == ""


def test_pure_reasoning_response_over_chat_stream_falls_back_instead_of_an_empty_success() -> None:
    """Integration-level: `ResilientLLMProvider.stream()` (what `ChatService.
    answer_stream` actually consumes) must yield NOTHING for a pure-reasoning
    response, so the caller's existing `if stream_failed or not chunks:
    -> fallback_answer` branch in `chat_service.py` naturally treats it as a
    failure -- never an empty "successful" stream."""

    class _PureReasoningProvider(LLMProvider):
        provider_name = "fake"
        model = "fake-model"

        async def chat(self, messages, temperature: float = 0.1) -> LLMResponse:
            raise NotImplementedError

        async def stream(self, messages, temperature: float = 0.1) -> AsyncIterator[str]:
            for piece in ["<think>", "only reasoning", "</think>"]:
                yield piece

        async def health(self) -> bool:
            return True

    provider = ResilientLLMProvider(_PureReasoningProvider(), [])

    async def collect() -> list[str]:
        return [chunk async for chunk in provider.stream([ChatMessage(role="user", content="hi")])]

    chunks = asyncio.run(collect())
    assert chunks == []  # nothing at all -- caller must treat this as "no answer", not "empty answer"


# --- 7. Normal answer containing legitimate angle-bracket text -------------

@pytest.mark.parametrize(
    "text",
    [
        "Under Section 34, if X < Y then the claim fails.",
        "The notice period is 15 days (see <Annexure A>).",
        "Compare a < b < c for the three thresholds.",
        "<b>Important:</b> file within the limitation period.",
    ],
)
def test_legitimate_angle_bracket_text_passes_through_untouched(text: str) -> None:
    assert _feed_all([text]) == text
    # Also unaffected when delivered in arbitrary small chunks.
    assert _feed_all([text[i : i + 3] for i in range(0, len(text), 3)]) == text


def test_text_that_merely_starts_with_think_but_is_not_the_tag_is_untouched() -> None:
    text = "The <thinking-cap> emoji is not a reasoning tag."
    assert _feed_all([text]) == text


# --- 8. Provider error / retry after partial streaming ---------------------

def test_error_text_after_partial_clean_streaming_passes_through() -> None:
    """A provider that streams real answer text, then (mid-stream) starts
    yielding a plain-text error sentence instead of raising (this codebase's
    documented provider contract, see `app/llm/resilient.py`'s module
    docstring) must not have that error text treated as think-markup or
    swallowed -- it has no `<think` in it, so it must reach the caller
    exactly as the provider yielded it."""
    pieces = ["Here is part of a real answer. ", "Gemini API is unreachable. Check your network and try again."]
    result = _feed_all(pieces)
    assert result == "".join(pieces)


def test_retry_after_partial_stream_starts_a_fresh_filter_with_no_leftover_state() -> None:
    """`ResilientLLMProvider.stream()` only ever streams the primary once
    (documented: no mid-stream failover), but a fresh call (e.g. the caller
    retrying the whole request) must not carry over any buffered state from
    a previous, unrelated stream."""

    class _OnceThenCleanProvider(LLMProvider):
        provider_name = "fake"
        model = "fake-model"

        def __init__(self) -> None:
            self.calls = 0

        async def chat(self, messages, temperature: float = 0.1) -> LLMResponse:
            raise NotImplementedError

        async def stream(self, messages, temperature: float = 0.1) -> AsyncIterator[str]:
            self.calls += 1
            if self.calls == 1:
                # Ends mid-think -- simulates a dropped connection right
                # after an opening tag with no close ever arriving.
                yield "<think>reasoning that never finishes because the call drops"
                return
            yield "A clean answer on retry."

        async def health(self) -> bool:
            return True

    fake = _OnceThenCleanProvider()
    provider = ResilientLLMProvider(fake, [])

    async def collect() -> list[str]:
        return [chunk async for chunk in provider.stream([ChatMessage(role="user", content="hi")])]

    first = asyncio.run(collect())
    assert first == []  # the dropped mid-think call leaked nothing

    second = asyncio.run(collect())
    # The trailing few characters may arrive as a separate chunk from the
    # filter's own end-of-stream `flush()` -- that's expected (a short
    # holdback in case a tag is about to start), not a leftover-state bug.
    # What matters is the fresh filter carried over no buffered/`_in_think`
    # state from the previous, unrelated call.
    assert "".join(second) == "A clean answer on retry."
