"""The "unverified source disclosure" fallback: when the strict-RAG path finds
nothing `review_status=approved`, a `needs_review` candidate that clears the
SAME relevance gate is quoted verbatim, clearly labelled as unverified, rather
than either a bare refusal or a false "verified" claim -- see the corrected
KB audit's discussion of scaling review beyond one human's throughput.

Everything here calls `ChatService._unverified_candidate_disclosure`/
`_no_verified_context_answer` as unbound methods against a `SimpleNamespace`
stand-in for `self`, the same pattern `test_kb_existing_law_audit.py` uses --
these two methods only touch `self.retriever`, never the rest of `ChatService`.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.constants import no_verified_context_message, unverified_source_disclosure
from app.rag.kb_jurisdiction import REVIEW_NEEDS_REVIEW
from app.schemas.common import RetrievedChunk
from app.services import chat_service
from app.services.chat_service import ChatService

_CANDIDATE = RetrievedChunk(
    chunk_id="c1", score=0.4,
    text="15. Where the wages of an employed person are not paid on the day fixed under section 5, the "
    "employed person may apply to the authority appointed under this Act for a direction for payment.",
    metadata={"act_name": "The Payment of Wages Act, 1936", "source_document": "example.pdf"},
)


def _fake_self(retrieve_result=("", [])) -> SimpleNamespace:
    return SimpleNamespace(retriever=SimpleNamespace(retrieve=AsyncMock(return_value=retrieve_result)))


class TestUnverifiedCandidateDisclosure:
    def test_disabled_by_settings_returns_none_without_querying(self, monkeypatch):
        monkeypatch.setattr(chat_service.settings, "chat_unverified_disclosure_enabled", False)
        fake_self = _fake_self()
        result = asyncio.run(
            ChatService._unverified_candidate_disclosure(fake_self, "unpaid salary question", "english")
        )
        assert result is None
        fake_self.retriever.retrieve.assert_not_awaited()

    def test_queries_needs_review_only_never_approved(self, monkeypatch):
        monkeypatch.setattr(chat_service.settings, "chat_unverified_disclosure_enabled", True)
        fake_self = _fake_self(("rewritten", [_CANDIDATE]))
        asyncio.run(ChatService._unverified_candidate_disclosure(fake_self, "unpaid salary for 3 months", "english"))
        _args, kwargs = fake_self.retriever.retrieve.await_args
        assert kwargs["filters"]["review_status"] == REVIEW_NEEDS_REVIEW
        assert kwargs["filters"]["owner_session_id"] == [None]
        assert kwargs["filters"]["owner_user_id"] == [None]

    def test_relevant_candidate_is_disclosed_verbatim_with_source_label(self, monkeypatch):
        monkeypatch.setattr(chat_service.settings, "chat_unverified_disclosure_enabled", True)
        fake_self = _fake_self(("unpaid salary wages employment dues labour law appointment letter", [_CANDIDATE]))
        result = asyncio.run(
            ChatService._unverified_candidate_disclosure(fake_self, "my salary was not paid for 3 months", "english")
        )
        assert result is not None
        assert _CANDIDATE.text in result  # quoted verbatim, not paraphrased
        assert "The Payment of Wages Act, 1936" in result
        assert "NOT" in result  # the "not verified"/"not confirmed" warning is present

    def test_irrelevant_candidate_does_not_clear_the_relevance_gate(self, monkeypatch):
        monkeypatch.setattr(chat_service.settings, "chat_unverified_disclosure_enabled", True)
        unrelated = RetrievedChunk(
            chunk_id="c2", score=0.4, text="Sections 91 to 94 of this Act shall be omitted.",
            metadata={"act_name": "Some Unrelated Act", "source_document": "other.pdf"},
        )
        fake_self = _fake_self(("rewritten", [unrelated]))
        result = asyncio.run(
            ChatService._unverified_candidate_disclosure(fake_self, "what is the punishment for cheating", "english")
        )
        assert result is None

    def test_no_candidates_returns_none(self, monkeypatch):
        monkeypatch.setattr(chat_service.settings, "chat_unverified_disclosure_enabled", True)
        fake_self = _fake_self(("rewritten", []))
        result = asyncio.run(ChatService._unverified_candidate_disclosure(fake_self, "anything", "english"))
        assert result is None

    def test_a_broken_lookup_fails_silent_rather_than_raising(self, monkeypatch):
        monkeypatch.setattr(chat_service.settings, "chat_unverified_disclosure_enabled", True)
        fake_self = SimpleNamespace(
            retriever=SimpleNamespace(retrieve=AsyncMock(side_effect=RuntimeError("mongo down")))
        )
        result = asyncio.run(ChatService._unverified_candidate_disclosure(fake_self, "anything", "english"))
        assert result is None

    def test_long_excerpt_is_truncated_on_a_word_boundary(self, monkeypatch):
        monkeypatch.setattr(chat_service.settings, "chat_unverified_disclosure_enabled", True)
        long_chunk = RetrievedChunk(
            chunk_id="c3", score=0.4, text="unpaid salary wages employment " * 50,
            metadata={"act_name": "Some Wages Act", "source_document": "w.pdf"},
        )
        fake_self = _fake_self(("unpaid salary wages employment dues labour law appointment letter", [long_chunk]))
        result = asyncio.run(
            ChatService._unverified_candidate_disclosure(fake_self, "unpaid salary for months", "english")
        )
        assert result is not None
        assert not result.rstrip("”").endswith("employment")  # cut cleanly, not mid-word
        assert "…" in result


class TestNoVerifiedContextAnswerIncludesDisclosure:
    def test_disclosure_is_appended_after_the_refusal_and_guidance(self, monkeypatch):
        monkeypatch.setattr(chat_service.settings, "kb_gap_autofetch_enabled", False)
        fake_self = SimpleNamespace(
            _unverified_candidate_disclosure=AsyncMock(return_value="DISCLOSURE_BLOCK"),
        )
        answer = asyncio.run(ChatService._no_verified_context_answer(fake_self, "asdf qwerty zxcv", "english"))
        assert answer.startswith(no_verified_context_message("english"))
        assert answer.rstrip().endswith("DISCLOSURE_BLOCK")

    def test_no_disclosure_available_leaves_the_refusal_unchanged(self, monkeypatch):
        monkeypatch.setattr(chat_service.settings, "kb_gap_autofetch_enabled", False)
        fake_self = SimpleNamespace(_unverified_candidate_disclosure=AsyncMock(return_value=None))
        answer = asyncio.run(ChatService._no_verified_context_answer(fake_self, "asdf qwerty zxcv", "english"))
        assert "DISCLOSURE" not in answer


class TestUnverifiedSourceDisclosureMessage:
    def test_composes_header_source_excerpt_and_footer(self):
        message = unverified_source_disclosure("english", "Some Act, 2019", "verbatim excerpt text")
        assert "NOT yet been checked by a human reviewer" in message
        assert "Some Act, 2019" in message
        assert "verbatim excerpt text" in message
        assert "NOT confirmed law" in message

    def test_excerpt_is_never_altered(self):
        excerpt = "  weird   spacing but untouched otherwise  "
        message = unverified_source_disclosure("english", "Act", excerpt)
        assert excerpt.strip() in message

    @pytest.mark.parametrize("language", ["hindi", "hinglish", "tamil", "unknown-language", None])
    def test_every_language_falls_back_safely_never_empty(self, language):
        message = unverified_source_disclosure(language, "Act", "excerpt")
        assert message.strip()
        assert "Act" in message and "excerpt" in message
