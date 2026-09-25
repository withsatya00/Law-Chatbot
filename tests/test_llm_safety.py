"""Direct unit tests for `app.llm.safety` (Phase 1 god-object split out of
`ChatService`). Before this extraction, these branches were only reachable
indirectly through a full `answer()`/`answer_stream()` call with a mocked
failing LLM.
"""
from app.llm.base import LLMResponse
from app.llm.safety import _LLM_ERROR_PHRASES, failure_category, looks_like_llm_error, safe_llm_text


def test_safe_llm_text_returns_fallback_on_error_even_with_nonempty_content() -> None:
    response = LLMResponse(
        content="Gemini API is unreachable.", error="connection refused", provider="gemini", model="test",
    )
    assert safe_llm_text(response, fallback="fallback text") == "fallback text"


def test_safe_llm_text_returns_fallback_on_empty_content() -> None:
    response = LLMResponse(content="   ", error=None, provider="gemini", model="test")
    assert safe_llm_text(response, fallback="fallback text") == "fallback text"


def test_safe_llm_text_returns_real_content_when_no_error() -> None:
    response = LLMResponse(content="This is a real answer.", error=None, provider="gemini", model="test")
    assert safe_llm_text(response, fallback="fallback text") == "This is a real answer."


def test_failure_category_none_for_no_error() -> None:
    assert failure_category(None) is None
    assert failure_category("") is None


def test_failure_category_timeout() -> None:
    assert failure_category("request deadline exceeded") == "timeout"
    assert failure_category("operation timed out") == "timeout"


def test_failure_category_rate_limit() -> None:
    assert failure_category("429 rate limit exceeded") == "rate_limit"
    assert failure_category("quota exceeded") == "rate_limit"


def test_failure_category_provider_auth() -> None:
    assert failure_category("401 unauthorized") == "provider_auth"
    assert failure_category("invalid api key") == "provider_auth"
    assert failure_category("request was rejected") == "provider_auth"


def test_failure_category_provider_not_configured() -> None:
    assert failure_category("provider not configured") == "provider_not_configured"


def test_failure_category_defaults_to_provider_unavailable() -> None:
    assert failure_category("something unexpected broke") == "provider_unavailable"


def test_looks_like_llm_error_matches_every_known_phrase() -> None:
    for phrase in _LLM_ERROR_PHRASES:
        assert looks_like_llm_error(f"Some prefix. {phrase}. Some suffix.")


def test_looks_like_llm_error_false_for_a_real_answer() -> None:
    assert not looks_like_llm_error("Under Section 138 of the Negotiable Instruments Act...")
