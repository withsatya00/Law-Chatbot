"""Post-Phase-3 hardening, Phase 2 milestone D: a Hindi session stays in
Hindi, and Devanagari survives every boundary it crosses.

Part 1 -- the language defect, from the observed session
--------------------------------------------------------
A rent-notice draft was open. The user typed "hindi me". The next reply came
back in Hindi and the field list was in Hindi, so the user answered in Hindi
and expected a Hindi document. What they got, several turns later, was a
notice whose body was English scaffolding ("Sir/Madam,", "Introduction",
"You are hereby put to notice...") wrapped around Devanagari values.

The reason: "hindi me" mid-collection was classified as **Translation** and
treated as an interruption, so it was answered by translating the previous
reply. That translation was cosmetic and lasted exactly one turn --
`draft_language` was never set. The very next turn's questions came back in
English, and so did the document. `ChatService._is_draft_interruption` now
treats a message that NAMES a language, while a draft is collecting, as the
drafting engine's business, which is where the persistent switch already
lived.

A second, smaller defect in the same flow: Hindi was the one language of
eleven whose field labels were rendered as "Applicant Address (आवेदक का पता)"
rather than in Hindi. Ten Eighth Schedule languages were answered entirely in
themselves; Hindi was not.

Part 2 -- encoding
------------------
No mojibake was found anywhere in the observed transcript: the Devanagari in
it is correct end to end. These round-trip tests are therefore written to
*pin* that rather than to fix it, and they cover each boundary the text
actually crosses: the API's JSON encoder, the Redis value encoder, the
conversation-memory dict, the four export formats, and the bytes a download
button hands the user.
"""

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import orjson
import pytest

from app.drafting.export import ExportOptions
from app.drafting.templates import get_template
from app.llm.base import LLMResponse
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.drafting import DraftPreviewRequest
from app.services.chat_service import ChatService

# Synthetic Hindi values, covering the characters that break naive encoders:
# nukta (ज़, ग़), matras, a conjunct, the danda, and the rupee sign.
_HINDI_NAME = "अमित कुमार शर्मा"
_HINDI_ADDRESS = "24, नेहरू नगर, ग़ाज़ियाबाद, उत्तर प्रदेश – 201001"
_HINDI_FACTS = "मैंने 31 जुलाई 2026 को परिसर खाली कर दिया और चाबियाँ सौंप दीं।"
_HINDI_RELIEF = "₹50,000 की सुरक्षा जमा राशि वापस की जाए।"


def _memory_service(memory: dict[str, Any]) -> ChatService:
    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])

    async def _load(session_id: str) -> dict[str, Any]:
        return memory

    async def _append(session_id: str, role: str, content: str) -> dict[str, Any]:
        memory.setdefault("messages", []).append({"role": role, "content": content})
        return memory

    async def _update(session_id: str, **values: Any) -> dict[str, Any]:
        memory.update(values)
        return memory

    service.memory.load = _load
    service.memory.append = _append
    service.memory.update = _update
    service.memory.summarize_if_needed = AsyncMock(return_value=memory)
    service.memory.append_intent_event = AsyncMock(return_value=memory)
    service.history.insert = AsyncMock(return_value="history-id")
    service.query_log.insert = AsyncMock(return_value="query-log-id")
    service.intent_events.insert = AsyncMock(return_value="intent-event-id")
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    service.retriever.retrieve = AsyncMock(return_value=("", []))
    service.reranker.rerank = AsyncMock(return_value=[])
    service.draft_conversation.extractor._extract_with_llm = AsyncMock(return_value={})
    # A completed draft would otherwise call the real provider and persist to
    # Mongo. Neither is what these tests are about, and the deterministic path
    # is the one whose language output matters most anyway.
    engine = service.draft_conversation.draft_engine
    engine.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="unavailable", model="t", provider="t", error="provider unavailable",
            error_kind="provider_error",
        )
    )
    engine.drafts.insert = AsyncMock(return_value="draft-id")
    engine.versions.insert = AsyncMock(return_value="version-id")
    return service


def _ask(service: ChatService, question: str) -> ChatResponse:
    return asyncio.run(service.answer(ChatRequest(question=question, session_id="s-lang")))


# ---------------------------------------------------------------------------
# 1. The language switch has to stick
# ---------------------------------------------------------------------------


def test_a_language_request_mid_draft_switches_the_draft_not_just_one_reply() -> None:
    memory: dict[str, Any] = {}
    service = _memory_service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye legal notice draft karo")
    _ask(service, "hindi me")
    assert memory.get("draft_language") == "hindi"
    # And it is remembered for the next document too.
    assert memory.get("draft_language_preference") == "hindi"


def test_the_questions_stay_in_hindi_on_the_following_turns() -> None:
    memory: dict[str, Any] = {}
    service = _memory_service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye legal notice draft karo")
    hindi_turn = _ask(service, "hindi me")
    assert "मुझे बस" in hindi_turn.answer

    # An unrelated legal question parks the draft; "continue draft" resumes it.
    _ask(service, "अग्रिम जमानत क्या होती है?")
    resumed = _ask(service, "continue draft")
    assert "मुझे बस" in resumed.answer
    assert "I just need" not in resumed.answer


def test_the_parked_draft_reminder_is_in_hindi() -> None:
    memory: dict[str, Any] = {}
    service = _memory_service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye legal notice draft karo")
    _ask(service, "hindi me")
    parked = _ask(service, "अग्रिम जमानत क्या होती है?")
    assert "मसौदा सुरक्षित है" in parked.answer
    assert "draft is saved" not in parked.answer


def test_hindi_field_labels_are_shown_in_hindi() -> None:
    memory: dict[str, Any] = {}
    service = _memory_service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye legal notice draft karo")
    answer = _ask(service, "hindi me").answer
    assert "आवेदक का पता" in answer
    # Not the old "Applicant Address (आवेदक का पता)" dual form.
    assert "Applicant Address" not in answer
    assert "Respondent Address" not in answer


def test_the_document_name_is_shown_in_hindi() -> None:
    memory: dict[str, Any] = {}
    service = _memory_service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye legal notice draft karo")
    answer = _ask(service, "hindi me").answer
    template = get_template("rent_notice")
    assert template is not None
    assert template.hindi_name in answer


def test_a_translation_request_that_names_no_language_still_interrupts() -> None:
    """The carve-out is narrow: "simplify this" mid-draft is not a language
    switch and must keep its old behaviour."""
    memory: dict[str, Any] = {"draft_mode": True, "draft_stage": "collecting"}
    service = _memory_service(memory)

    class _Match:
        intent = "Translation"
        confidence = 0.9

    assert service._is_draft_interruption(_Match(), memory, "translate this into simple words") is True
    assert service._is_draft_interruption(_Match(), memory, "hindi me") is False


def test_canonical_act_names_survive_a_hindi_draft() -> None:
    """The Act a template cites is a proper name, not a label to translate."""
    engine_response = _hindi_rent_notice()
    body = "\n".join(engine_response.sections.values())
    assert "Transfer of Property Act, 1882" in body
    assert engine_response.applicable_acts == ["Transfer of Property Act, 1882", "applicable state Rent Control Act"]


# ---------------------------------------------------------------------------
# 2. Encoding round trips
# ---------------------------------------------------------------------------

_HINDI_FIELDS: dict[str, str] = {
    "applicant_name": _HINDI_NAME,
    "applicant_address": _HINDI_ADDRESS,
    "applicant_mobile": "9812345670",
    "respondent_name": "राजेश वर्मा",
    "respondent_address": "78, सेक्टर 18, नोएडा, उत्तर प्रदेश – 201301",
    "rented_premises_address": "मकान नंबर 112, गली नंबर 4, शास्त्री नगर, ग़ाज़ियाबाद – 201002",
    "security_deposit_amount": "50,000",
    "facts": _HINDI_FACTS,
    "expected_relief": _HINDI_RELIEF,
    "place": "ग़ाज़ियाबाद, उत्तर प्रदेश",
}


def _hindi_rent_notice():
    from app.drafting.engine import LegalDraftEngine
    from app.llm.base import LLMResponse

    engine = LegalDraftEngine()
    engine.llm.chat = AsyncMock(
        return_value=LLMResponse(
            content="unavailable", model="t", provider="t", error="provider unavailable"
        )
    )
    return asyncio.run(
        engine.preview(DraftPreviewRequest(draft_id="rent_notice", language="hindi", fields=_HINDI_FIELDS))
    )


def test_devanagari_survives_the_api_json_encoder() -> None:
    """FastAPI serializes the response model with this encoder."""
    from app.schemas.common import LawyerRecommendation

    response = ChatResponse(
        message_id="m",
        session_id="s",
        answer=_HINDI_FACTS,
        sources=[],
        confidence=0.6,
        lawyer_recommendation=LawyerRecommendation(category="Civil", confidence=0.5, reason="test"),
        detected_intent="General Legal Information",
        detected_language="hindi",
        latency_ms=1.0,
    )
    payload = response.model_dump_json()
    assert json.loads(payload)["answer"] == _HINDI_FACTS
    # And through the byte layer an ASGI server actually writes.
    assert json.loads(payload.encode("utf-8").decode("utf-8"))["answer"] == _HINDI_FACTS


def test_devanagari_survives_the_redis_value_encoder() -> None:
    """`RedisClient` stores with orjson and reads back with `decode_responses=False`."""
    state = {"draft_fields": _HINDI_FIELDS, "draft_language": "hindi", "answer": _HINDI_RELIEF}
    assert orjson.loads(orjson.dumps(state)) == state


def test_devanagari_survives_a_mongo_style_document_round_trip() -> None:
    """BSON strings are UTF-8 by definition; this pins that the values this app
    stores are ones it can encode."""
    bson = pytest.importorskip("bson")
    document = {"_id": "d1", "fields": _HINDI_FIELDS, "sections": {"तथ्य": _HINDI_FACTS}}
    assert bson.BSON.decode(bson.BSON.encode(document)) == document


def test_devanagari_survives_the_conversation_memory_dict() -> None:
    memory: dict[str, Any] = {}
    service = _memory_service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye legal notice draft karo")
    _ask(service, "hindi me")
    _ask(service, f"आवेदक का नाम: {_HINDI_NAME}")
    assert memory["draft_fields"]["applicant_name"] == _HINDI_NAME


@pytest.mark.parametrize("fmt", ["txt", "rtf", "docx", "pdf"])
def test_devanagari_survives_each_export_format(fmt: str, tmp_path: Path) -> None:
    """The bytes a download button hands the user are the bytes checked here --
    `DraftExporter` is the single producer for the API, the chat download and
    the Streamlit button alike."""
    from app.drafting.engine import LegalDraftEngine

    response = _hindi_rent_notice()
    # The engine's own dispatch table -- the single producer behind the export
    # API, the chat download and the Streamlit save button alike.
    exporter = LegalDraftEngine().exporters[fmt]  # type: ignore[index]
    output = tmp_path / f"draft.{fmt}"
    exporter.export(
        template_name=get_template("rent_notice").name,  # type: ignore[union-attr]
        sections=response.sections,
        output_path=output,
        language="hindi",
        options=ExportOptions(),
    )
    assert output.exists() and output.stat().st_size > 0

    if fmt == "txt":
        assert _HINDI_NAME in output.read_text(encoding="utf-8")
    elif fmt == "rtf":
        # RTF has no UTF-8 body: non-ASCII is `\uN?`, which is correct, not
        # mojibake. Decoding those escapes back must reproduce the original.
        import re

        raw = output.read_text(encoding="ascii")
        decoded = re.sub(
            r"\\u(-?\d+)\?", lambda m: chr(int(m.group(1)) % 0x10000), raw
        )
        assert _HINDI_NAME in decoded
    elif fmt == "docx":
        docx = pytest.importorskip("docx")
        text = "\n".join(p.text for p in docx.Document(str(output)).paragraphs)
        assert _HINDI_NAME in text
    else:
        assert output.read_bytes()[:4] == b"%PDF"


def test_the_export_filename_and_bytes_are_independent_of_the_console_encoding() -> None:
    """A Windows console is cp1252 here. Nothing in the export path may depend
    on the ambient encoding -- which is why every write names utf-8 explicitly."""
    import inspect

    from app.drafting import export

    source = inspect.getsource(export)
    for line in source.splitlines():
        stripped = line.strip()
        if ".write_text(" in stripped and "encoding=" not in stripped:
            pytest.fail(f"write_text without an explicit encoding: {stripped}")


def test_a_whole_block_of_labelled_hindi_answers_is_not_read_as_a_question() -> None:
    """Found live, against the running backend, on the third turn of a Hindi
    rent-notice conversation.

    `_is_draft_interruption` fell through to `looks_informational`, whose own
    comment reasoned that "a genuine field value (a name, address, amount,
    date) never looks like a question". That holds for one short value. The
    user had answered all ten printed questions in a single 1,143-character
    block whose Facts and Relief entries are full sentences -- and
    `looks_informational` returns True for it. The block was routed to
    retrieval, answered with the no-verified-context refusal, and every value
    in it was discarded.

    The check now asks the extractor -- the component that actually knows
    which labels belong to the template being collected -- before the prose
    heuristic gets a vote.
    """
    memory: dict[str, Any] = {}
    service = _memory_service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye rent notice draft karo")
    _ask(service, "hindi me")

    # Reconstruct a complete, synthetic labelled answer from the real
    # template.  The previous fixture was accidentally reduced to its first
    # field, while the assertions below still expected the remaining nine
    # values.  The facts deliberately contain the strong informational cue
    # "kaise": that recreates the routing collision this regression protects
    # against without copying private transcript data into the suite.
    template = get_template("rent_notice")
    assert template is not None
    labels = {field.key: field.hindi_label for field in template.required_fields}
    values = {
        **_HINDI_FIELDS,
        "facts": (
            f"{_HINDI_FACTS} "
            "मकान मालिक ने यह नहीं बताया कि जमा राशि कैसे वापस की जाएगी।"
        ),
    }
    block = "\n".join(
        f"{labels[key]}: {values[key]}" for key in labels
    )

    from app.drafting.intent import looks_informational

    # The heuristic still says "question" -- the fix is that it is no longer
    # asked first, not that its answer changed.
    assert looks_informational(block) is True

    response = _ask(service, block)
    assert response.conversation_intent == "Draft Generation"
    fields = memory.get("draft_fields") or {}
    assert fields.get("applicant_name") == "अमित कुमार शर्मा"
    assert fields.get("respondent_name") == "राजेश वर्मा"
    assert fields.get("security_deposit_amount") == "50,000"
    assert "Knowledge Base" not in response.answer


def test_a_genuine_question_mid_draft_still_interrupts() -> None:
    """The carve-out must not swallow real questions: it requires two or more
    labelled values for fields of the template actually being collected."""
    memory: dict[str, Any] = {}
    service = _memory_service(memory)
    _ask(service, "Mere landlord ko security deposit wapas karne ke liye rent notice draft karo")
    response = _ask(service, "What is the difference between RTI and PIL?")
    assert response.conversation_intent != "Draft Generation"
    # ...and the draft survives.
    assert memory.get("draft_template_id") == "rent_notice"
