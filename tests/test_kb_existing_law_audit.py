"""Regression tests for the 2026-09-21 KB coverage/retrieval audit correction.

The audit called NI Act s.138, Contract Act s.10, RTI ss.6/7/19, the Consumer
Protection Act 2019, CPC s.80, BNS ss.303/304/318, BNSS s.173, the RBI
limiting-liability circular and Payment of Wages Act s.15 "KB gaps". All are
indexed (checked against MongoDB); the refusals were bridge/retrieval misses
that `GapAutoFetchService` then queued for download. File names below are the
real `metadata.source_document` values from that live check.
"""

import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pypdf import PdfWriter

from app.models.collections import EMBEDDINGS_METADATA, KB_GAP_AUTOFETCH_QUEUE
from app.rag.multilingual import has_legal_concept, legal_english_variants, legal_intent_hint
from app.rag.query_rewriter import SmartQueryRewriter
from app.rag.reranker import _UNPAID_WAGE_QUERY_RE, LegalReranker
from app.schemas.common import RetrievedChunk
from app.services import kb_gap_autofetch, kb_official_source_sync
from app.services.kb_gap_autofetch import GapAutoFetchService
from app.services.kb_official_source_sync import SOURCES, OfficialSource, OfficialSourceSyncService
from app.services.kb_presence import (
    STATUS_MANUAL_ONLY,
    STATUS_PRESENT,
    find_existing_kb_law,
    match_known_laws,
)
from app.services.kb_scanned_ingestion import ScannedOfficialPdfIngestion

# (label, act name as an LLM would return it, a real KB source_document that holds it)
EXISTING_LAWS = [
    ("NI Act s.138", "Negotiable Instruments Act, 1881, Section 138", "Negotiable_Instruments_Act_1881_Complete_Act.pdf"),
    ("Contract Act s.10", "Indian Contract Act, 1872, Section 10", "Indian_Contract_Act_1872_CAG_Official.pdf"),
    ("RTI ss.6/7/19", "Right to Information Act, 2005, Sections 6, 7, 19", "RTI_Act_2005_Official_Amended.pdf"),
    ("Consumer Protection Act", "Consumer Protection Act, 2019", "The_Consumer_Protection_Act_2019_5.pdf"),
    ("CPC s.80", "Code of Civil Procedure, 1908, Section 80", "CPC_1908_Official.pdf"),
    ("BNS ss.303/304/318", "Bharatiya Nyaya Sanhita, 2023, Sections 303, 304, 318", "The_Bhara_Tiy_A_Ny_A_Y_A_Sanhita_2023_5.pdf"),
    ("BNSS s.173", "Bharatiya Nagarik Suraksha Sanhita, 2023, Section 173", "BNSS_2023_Official_Gazette.pdf"),
    (
        "RBI limiting-liability circular",
        "RBI Master Direction on Limiting Liability of Customers in Unauthorised Electronic Banking Transactions",
        "RBI_Limiting_Liability_Unauthorised_Electronic_Banking_2017.html",
    ),
    ("Payment of Wages s.15", "Payment of Wages Act, 1936, Section 15", "mh_acts_1936.04_payment-of-wages-act-1936.pdf"),
    ("Code on Wages", "Code on Wages, 2019", "2589gi_P65_6.pdf"),
]
_IDS = [case[0] for case in EXISTING_LAWS]


class _Collection:
    def __init__(self, docs: list[dict] | None = None) -> None:
        self.docs = docs or []
        self.upserts: list[dict] = []

    async def find_one(self, query, projection=None):
        for clause in query.get("$or", [query]):
            for path, condition in clause.items():
                wanted = set(condition["$in"])
                for doc in self.docs:
                    meta = doc.get("metadata", {})
                    if meta.get(path.removeprefix("metadata.")) in wanted:
                        return doc
        return None

    async def update_one(self, filter_, update, upsert=False):
        self.upserts.append({"filter": filter_, "update": update})


def _db(*source_documents: str, keys: tuple[str, ...] = ()) -> dict:
    docs = [{"metadata": {"source_document": name}} for name in source_documents]
    docs += [{"metadata": {"document_key": key}} for key in keys]
    return {EMBEDDINGS_METADATA: _Collection(docs), KB_GAP_AUTOFETCH_QUEUE: _Collection()}


def _service(db: dict, llm_answer: str) -> GapAutoFetchService:
    llm = SimpleNamespace(chat=AsyncMock(return_value=SimpleNamespace(error=None, content=llm_answer)))
    return GapAutoFetchService(
        db=db, downloader=object(), ingestion=object(), scanner=object(),
        llm_factory=SimpleNamespace(create_resilient=lambda: llm),
    )


@pytest.fixture
def autofetch_enabled(monkeypatch):
    monkeypatch.setattr(kb_gap_autofetch.settings, "kb_gap_autofetch_enabled", True)


# --------------------------------------------------- existing-law bridge matching


@pytest.mark.parametrize(("label", "act_name", "kb_file"), EXISTING_LAWS, ids=_IDS)
def test_existing_law_titles_match_the_catalog(label, act_name, kb_file):
    laws = match_known_laws(act_name)
    assert laws, f"{label}: no catalog entry matched {act_name!r}"
    assert any(kb_file in law.source_documents for law in laws)


@pytest.mark.parametrize(
    "act_name",
    [
        "Section 138 of the Negotiable Instruments Act, 1881",
        "The Negotiable Instruments Act, 1881",
        "Consumer Protection Act",  # no year: the current Act is the only one indexed
        "Bharatiya Nyaya (Second) Sanhita, 2023",  # the Act's enacted title
        "Bharatiya Nagarik Suraksha (Second) Sanhita, 2023",
        "The Right to Information Act, 2005",
    ],
)
def test_title_variants_still_match(act_name):
    assert match_known_laws(act_name)


@pytest.mark.parametrize(
    "act_name",
    [
        "Consumer Protection Act, 1986",  # repealed predecessor, not indexed
        "Code of Civil Procedure (Amendment) Act, 2002",  # a different instrument
        "Right to Information (Repeal) Act, 2005",
        "Motor Vehicles Act, 1988",
        "",
    ],
)
def test_other_instruments_and_absent_acts_do_not_match(act_name):
    assert match_known_laws(act_name) == []


@pytest.mark.parametrize(("label", "act_name", "kb_file"), EXISTING_LAWS, ids=_IDS)
def test_presence_requires_real_indexed_chunks(label, act_name, kb_file):
    present = asyncio.run(find_existing_kb_law(_db(kb_file), act_name))
    assert present is not None
    assert present.status == STATUS_PRESENT
    # Same title, empty KB: a registered-but-never-ingested Act is a genuine gap.
    assert asyncio.run(find_existing_kb_law(_db(), act_name)) is None


def test_document_key_alone_proves_presence():
    # `document_key` is null for several central Acts, but where it is set it counts.
    present = asyncio.run(find_existing_kb_law(_db(keys=("consumer-protection-act-2019",)), "Consumer Protection Act, 2019"))
    assert present is not None


def test_it_amendment_is_manual_only_even_with_an_empty_kb():
    presence = asyncio.run(find_existing_kb_law(_db(), "Information Technology (Amendment) Act, 2008"))
    assert presence is not None and presence.status == STATUS_MANUAL_ONLY


def test_it_act_2000_is_present_but_flags_the_amended_text_gap():
    presence = asyncio.run(find_existing_kb_law(_db("IT_Act_2000_India_Code.pdf"), "Information Technology Act, 2000"))
    assert presence is not None and "amend" in presence.note.lower()


# ------------------------------------------ false autofetch queue entries prevented


@pytest.mark.parametrize(("label", "act_name", "kb_file"), EXISTING_LAWS, ids=_IDS)
def test_record_gap_never_queues_an_act_the_kb_holds(autofetch_enabled, label, act_name, kb_file):
    db = _db(kb_file)
    asyncio.run(_service(db, act_name).record_gap("any question"))
    assert db[KB_GAP_AUTOFETCH_QUEUE].upserts == []


def test_record_gap_still_queues_a_genuine_gap(autofetch_enabled):
    db = _db("Negotiable_Instruments_Act_1881_Complete_Act.pdf")
    asyncio.run(_service(db, "Motor Vehicles Act, 1988").record_gap("my bike was hit by a truck"))
    assert len(db[KB_GAP_AUTOFETCH_QUEUE].upserts) == 1
    assert db[KB_GAP_AUTOFETCH_QUEUE].upserts[0]["filter"] == {"act_name": "Motor Vehicles Act, 1988"}


def test_record_gap_queues_a_registered_act_that_was_never_ingested(autofetch_enabled):
    db = _db()  # Hindu Marriage Act is in the registry but has no chunks
    asyncio.run(_service(db, "Hindu Marriage Act, 1955").record_gap("grounds of divorce"))
    assert len(db[KB_GAP_AUTOFETCH_QUEUE].upserts) == 1


def test_record_gap_ignores_unidentifiable_questions(autofetch_enabled):
    db = _db()
    asyncio.run(_service(db, "NONE").record_gap("asdf"))
    assert db[KB_GAP_AUTOFETCH_QUEUE].upserts == []


def test_record_gap_never_queues_the_manual_only_it_amendment(autofetch_enabled):
    db = _db()
    asyncio.run(_service(db, "Information Technology (Amendment) Act, 2008").record_gap("section 66C"))
    assert db[KB_GAP_AUTOFETCH_QUEUE].upserts == []


def test_stale_pending_job_for_an_existing_act_is_closed_not_fetched(autofetch_enabled):
    db = _db("Indian_Contract_Act_1872_CAG_Official.pdf")
    service = _service(db, "unused")
    job = {"_id": "job-1", "act_name": "Indian Contract Act, 1872", "attempts": 1}
    service._claim = AsyncMock(side_effect=[job, None])
    service._search_candidates = AsyncMock()

    counts = asyncio.run(service.process_due())

    service._search_candidates.assert_not_awaited()
    assert counts["skipped_existing"] == 1 and counts["failed"] == 0
    status_update = db[KB_GAP_AUTOFETCH_QUEUE].upserts[0]["update"]["$set"]
    assert status_update["status"] == "skipped_existing"
    assert "retrieval miss" in status_update["reason"]


def test_unsupported_question_still_gets_the_safe_refusal(monkeypatch):
    from app.core.constants import no_verified_context_message
    from app.services import chat_service
    from app.services.chat_service import ChatService

    monkeypatch.setattr(chat_service.settings, "kb_gap_autofetch_enabled", False)

    fake_self = SimpleNamespace(_unverified_candidate_disclosure=AsyncMock(return_value=None))
    answer = asyncio.run(ChatService._no_verified_context_answer(fake_self, "asdf qwerty zxcv", "english"))
    assert answer.startswith(no_verified_context_message("english"))


# --------------------------------------------- query normalisation / bridge misses

_BRIDGE_CASES = [
    ("CPC s.80 (hindi)", "सरकार पर मुकदमा करने से पहले नोटिस देना जरूरी है क्या?", "section 80"),
    ("CPC s.80 (hinglish)", "sarkar par mukadma karna hai to kya notice chahiye", "section 80"),
    (
        "Consumer Act (santali)",
        "ᱤᱧ ᱮᱠᱚ ᱠᱷᱚᱨᱟᱯ ᱢᱚᱵᱭᱞ ᱯᱷᱳᱱ ᱠᱤᱨᱤᱱ ᱠᱮᱰᱟᱭ ᱠᱟᱱᱟ। ᱠᱚᱱᱥᱩᱢᱟᱨ ᱚᱸᱜᱚᱲ ᱞᱟᱹᱜᱤᱫ ᱪᱮᱫ ᱥᱟᱠᱥᱤ ᱛᱷᱚᱠᱟ ᱚᱰᱚᱠ ᱠᱟᱱᱟ?",
        "consumer protection act",
    ),
    ("BNS theft s.303 (hindi)", "मेरी बाइक चोरी हो गई, BNS में कौन सी धारा लगेगी?", "section 303"),
    ("BNS cheating s.318 (hinglish)", "kisi ne dhokhadhadi se paise le liye, kaunsi dhara lagegi", "section 318"),
    ("BNSS s.173 / Zero FIR (hindi)", "जीरो एफआईआर क्या होती है? थाना क्षेत्र से बाहर की घटना पर", "section 173"),
    ("RTI (hindi)", "आरटीआई का जवाब नहीं मिला, प्रथम अपील कैसे करें?", "first appeal"),
    ("NI s.138 (hindi)", "चेक बाउंस हो गया, अब क्या करें?", "section 138"),
    ("Payment of Wages (hindi)", "मेरे मालिक ने तीन महीने से वेतन नहीं दिया", "unpaid salary"),
]


@pytest.mark.parametrize(("label", "question", "marker"), _BRIDGE_CASES, ids=[case[0] for case in _BRIDGE_CASES])
def test_previously_missed_questions_reach_the_english_statutory_vocabulary(label, question, marker):
    expanded = " ".join(legal_english_variants(question) + SmartQueryRewriter().expand_queries(question)).lower()
    assert marker in expanded, f"{label}: {marker!r} missing from {expanded!r}"


@pytest.mark.parametrize(
    "question",
    [
        "Do I need to give notice before suing the government?",
        "can I sue the government for a road accident",
        "what does CPC section 80 say",
    ],
)
def test_english_government_suit_question_reaches_cpc_s80(question):
    assert "section 80" in " ".join(SmartQueryRewriter().expand_queries(question)).lower()


def test_government_suit_bridge_is_phrase_level_not_a_bare_government_trigger():
    assert not has_legal_concept("government job vacancy in my city")
    assert not SmartQueryRewriter().expand_queries("government job vacancy in my city")[1:]
    assert legal_intent_hint("सरकार पर मुकदमा कैसे करें") == ("Legal Notice", "Civil Law")


def test_cpc_s80_chunk_wins_the_top_slot_for_the_government_suit_question():
    s80 = RetrievedChunk(
        chunk_id="s80", score=0.18, metadata={"source_document": "CPC_1908_Official.pdf"},
        text="80. Notice. Save as otherwise provided in sub-section (2), no suits shall be instituted against the "
        "Government or against a public officer in respect of any act purporting to be done by such public officer",
    )
    other = RetrievedChunk(
        chunk_id="other", score=0.26, metadata={"source_document": "faq.pdf"},
        text="Notice of the date of hearing shall be served on the parties.",
    )
    question = " ".join(SmartQueryRewriter().expand_queries("Do I need to give notice before suing the government?"))
    ranked = asyncio.run(LegalReranker().rerank(question, [other, s80], top_k=2))
    assert ranked[0].chunk_id == "s80"


# ------------------------------------------------ Code on Wages retrieval preference

# Excerpts of the indexed text (2589gi_P65_6.pdf / mh_acts_1936.04_...).
_CODE = RetrievedChunk(
    chunk_id="code", score=0.28,
    text="Mode of payment of wages. Wages for two or more classes of work. Deductions from the wages of an employee",
    metadata={"source_document": "2589gi_P65_6.pdf", "document_key": "code-on-wages-2019", "issuing_level": "central"},
)
_PAYMENT_OF_WAGES_ACT = RetrievedChunk(
    chunk_id="pow", score=0.30,
    text="Wages means all remuneration (whether by way of salary, allowances or otherwise) expressed in terms of money",
    metadata={"source_document": "mh_acts_1936.04_payment-of-wages-act-1936.pdf", "document_key": "mh-1936.04-payment-of-wages-act-1936"},
)
_UNRELATED = RetrievedChunk(
    chunk_id="unrelated", score=0.25, text="Recognition of trade unions and registration of standing orders",
    metadata={"source_document": "The_Industrial_Relations_Code_2020_4.pdf", "document_key": "industrial-relations-code-2020"},
)


def _fresh(*chunks: RetrievedChunk) -> list[RetrievedChunk]:
    return [chunk.model_copy(deep=True) for chunk in chunks]


@pytest.mark.parametrize(
    "question",
    [
        "my employer has not paid my salary for three months",
        "unpaid salary wages employment dues labour law appointment letter",
        "salary pending since january what can I do",
    ],
)
def test_unpaid_salary_prefers_central_code_on_wages_and_keeps_payment_of_wages(question):
    ranked = asyncio.run(LegalReranker().rerank(question, _fresh(_PAYMENT_OF_WAGES_ACT, _UNRELATED, _CODE), top_k=3))
    assert [chunk.chunk_id for chunk in ranked] == ["code", "pow", "unrelated"]


def test_maithili_unpaid_salary_question_reaches_the_wage_preference():
    question = "हमर मालिक दू महिनाक तनखा नहि देलनि अछि। उचित उपाय चिन्हित करबाक लेल कोन-कोन कागजात आ तथ्य आवश्यक अछि?"
    rewritten = " ".join(SmartQueryRewriter().expand_queries(question))
    assert _UNPAID_WAGE_QUERY_RE.search(rewritten)
    ranked = asyncio.run(LegalReranker().rerank(rewritten, _fresh(_PAYMENT_OF_WAGES_ACT, _CODE), top_k=2))
    assert ranked[0].chunk_id == "code"


@pytest.mark.parametrize("question", ["what is my salary structure", "cheque bounced what to do", "FIR kaise darj karein"])
def test_wage_preference_is_inert_for_other_questions(question):
    for chunk in (_CODE, _PAYMENT_OF_WAGES_ACT):
        assert LegalReranker._wage_source_bonus(question.lower(), chunk.text.lower(), chunk.metadata) == 0.0


def test_wage_preference_ignores_chunks_from_other_acts():
    assert LegalReranker._wage_source_bonus("unpaid salary", _UNRELATED.text.lower(), _UNRELATED.metadata) == 0.0


# ----------------------------------------------- no regression in other retrieval


def test_cheque_bounce_still_ranks_ni_s138_above_wage_text():
    s138 = RetrievedChunk(
        chunk_id="s138", score=0.15, metadata={"source_document": "Negotiable_Instruments_Act_1881_Complete_Act.pdf"},
        text="138. Dishonour of cheque for insufficiency, etc., of funds in the account.",
    )
    ranked = asyncio.run(LegalReranker().rerank("cheque bounced what can I do", _fresh(_PAYMENT_OF_WAGES_ACT, _CODE, s138), top_k=3))
    assert ranked[0].chunk_id == "s138"


def test_rti_question_still_ranks_the_rti_act_first():
    rti = RetrievedChunk(
        chunk_id="rti", score=0.15, metadata={"source_document": "RTI_Act_2005_Official_Amended.pdf"},
        text="6. Request for obtaining information. A person who desires to obtain any information under this Act "
        "shall make a request in writing to the Public Information Officer of the public authority",
    )
    ranked = asyncio.run(LegalReranker().rerank("RTI application ka jawab nahi mila", _fresh(_PAYMENT_OF_WAGES_ACT, rti), top_k=2))
    assert ranked[0].chunk_id == "rti"


def test_bnss_s173_still_wins_for_a_fir_question():
    s173 = RetrievedChunk(
        chunk_id="s173", score=0.22, metadata={"source_document": "BNSS_2023_Official_Gazette.pdf"},
        text="173. (1) Every information relating to the commission of a cognizable offence, irrespective of the "
        "area where the offence is committed, may be given orally or by electronic communication to an officer in "
        "charge of a police station, and if given orally, it shall be reduced to writing.",
    )
    faq = RetrievedChunk(
        chunk_id="faq", score=0.30, metadata={"source_document": "delhi_police_faq.pdf"},
        text="Property dispute cases: people want to register FIRs in the police station of their choice.",
    )
    ranked = asyncio.run(LegalReranker().rerank("BNSS ke under FIR kaise darj karayi jaati hai?", _fresh(faq, s173), top_k=2))
    assert ranked[0].chunk_id == "s173"


def test_bns_ipc_420_still_crosswalks_to_s318():
    s318 = RetrievedChunk(
        chunk_id="s318", score=0.15, metadata={"source_document": "The_Bhara_Tiy_A_Ny_A_Y_A_Sanhita_2023_5.pdf"},
        text="318. Cheating and dishonestly inducing delivery of property",
    )
    other = RetrievedChunk(
        chunk_id="other", score=0.30, metadata={"source_document": "faq.pdf"}, text="Property delivered to the bailee",
    )
    ranked = asyncio.run(LegalReranker().rerank("IPC 420 ab BNS ki kaunsi dhara hai?", _fresh(other, s318), top_k=2))
    assert ranked[0].chunk_id == "s318"


# ------------------------------------------ registry / sync / coverage audit fixes


def _source(key: str) -> OfficialSource:
    return next(source for source in SOURCES if source.key == key)


def test_scanned_it_amendment_is_registered_as_ocr_only_with_an_official_url():
    from app.schemas.law_monitoring import official_url

    source = _source("IT_ACT_2008_AMENDMENT")
    assert source.requires_ocr
    assert official_url(source.url) == source.url
    assert source.url.endswith("IT_amendment_act2008-1_0.pdf")
    # The unamended 2000 text stays a normal, text-extractable source.
    assert not _source("IT_ACT").requires_ocr


def test_sync_skips_a_scanned_source_without_fetching_it():
    fetcher = AsyncMock()
    ingestion = SimpleNamespace(ingest=AsyncMock())
    result = asyncio.run(OfficialSourceSyncService(fetcher, ingestion).sync(_source("IT_ACT_2008_AMENDMENT")))
    assert result["status"] == "manual_ocr_required"
    assert "ingest_scanned_official_pdf" in result["reason"]
    fetcher.assert_not_awaited()
    ingestion.ingest.assert_not_awaited()


def test_code_on_wages_is_registered_under_its_real_kb_filename():
    source = _source("CODE_ON_WAGES_2019")
    assert "2589gi_P65_6.pdf" in source.kb_filenames
    assert source.document_key == "code-on-wages-2019"


def test_coverage_reports_manually_ingested_acts_as_indexed(monkeypatch):
    class _Coll:
        async def find_one(self, query):
            return {"filename": "x", "metadata": {"review_status": "approved"}} if "2589gi_P65_6.pdf" in str(query) else None

        async def count_documents(self, query):
            return 45 if "2589gi_P65_6.pdf" in str(query) else 0

    monkeypatch.setattr(kb_official_source_sync, "mongodb", SimpleNamespace(db={
        kb_official_source_sync.UPLOADED_DOCUMENTS: _Coll(), EMBEDDINGS_METADATA: _Coll(),
    }))
    rows = {row["law"]: row for row in asyncio.run(OfficialSourceSyncService(AsyncMock(), None).coverage())["sources"]}
    assert rows["CODE_ON_WAGES_2019"]["indexed"] is True
    assert rows["CODE_ON_WAGES_2019"]["chunks"] == 45
    assert rows["CODE_ON_WAGES_2019"]["review_status"] == "approved"
    assert rows["IT_ACT_2008_AMENDMENT"]["indexed"] is False and rows["IT_ACT_2008_AMENDMENT"]["requires_ocr"] is True


# ------------------------------------ OCR manual ingestion of a scanned official PDF

_SCAN = OfficialSource(
    "TESTSCAN", "scan.pdf", "https://www.meity.gov.in/static/uploads/test-scan.pdf", "test-scan-act-2008",
    "Official scan", "bare_act", ("informationtechnologyamendmentact2008",), requires_ocr=True,
)
_TITLE = "THE INFORMATION TECHNOLOGY (AMENDMENT) ACT, 2008"


def _blank_pdf(pages: int = 2) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class _Ocr:
    def __init__(self, pages=None, error=None) -> None:
        self.pages, self.error = pages, error

    async def extract_pdf_pages_individually(self, path, max_pages=50):
        assert path.is_file()
        if self.error:
            raise self.error
        return self.pages


def _ingestion(tmp_path, ocr, review_status="needs_review"):
    response = SimpleNamespace(
        status="indexed", generated_filename="scan.pdf", document_id="doc-1", chunks_indexed=7,
        review_status=review_status, reason=None,
    )
    ingest = AsyncMock(return_value=response)
    propagate = AsyncMock(return_value=7)
    service = ScannedOfficialPdfIngestion(
        ingestion=SimpleNamespace(ingest=ingest), ocr=ocr, propagate=propagate, evidence_dir=tmp_path,
    )
    return service, ingest, propagate


def _run(service, body=None, **kwargs):
    return asyncio.run(service.ingest(_SCAN, body or _blank_pdf(), acquisition="test", **kwargs))


def test_scanned_pdf_is_indexed_needs_review_with_page_level_provenance(tmp_path):
    ocr = _Ocr([(1, f"{_TITLE}\nAn Act to amend", 93.5), (2, "Sections 43A 66C", 88.0)])
    service, ingest, propagate = _ingestion(tmp_path, ocr)

    result = _run(service)

    assert result["status"] == "indexed_needs_review"
    assert result["human_review_required"] is True
    metadata = ingest.await_args.kwargs["jurisdiction_metadata"]
    assert metadata["review_status"] == "needs_review"
    assert metadata["verification_status"] == "unverified"
    assert metadata["last_verified_at"] is None and metadata["verified_by"] is None
    assert metadata["source_url"] == _SCAN.url and metadata["document_key"] == "test-scan-act-2008"
    assert metadata["jurisdiction_source_type"] == "bare_act"
    assert any("OCR" in reason and "human reviewer" in reason for reason in metadata["review_reasons"])

    document_id, source_document, fields = propagate.await_args.args
    assert (document_id, source_document) == ("doc-1", "scan.pdf")
    assert "review_status" not in fields and "verification_status" not in fields
    provenance = fields["ocr_provenance"]
    assert fields["ingestion_route"] == "manual_ocr_scanned_official_pdf"
    assert provenance["source_url"] == _SCAN.url and provenance["source_type"] == "bare_act"
    assert len(provenance["source_sha256"]) == 64 and provenance["human_review_required"] is True
    assert [page["page"] for page in provenance["pages"]] == [1, 2]
    assert provenance["pages"][0]["confidence"] == 93.5
    assert provenance["mean_confidence"] == 90.75 and provenance["low_confidence_pages"] == []
    assert provenance["identity_tokens_matched"] == list(_SCAN.identity_tokens)

    manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
    assert manifest["outcome"]["status"] == "indexed_needs_review"
    assert manifest["pages"][1] == {"page": 2, "chars": len("Sections 43A 66C"), "confidence": 88.0}


def test_low_confidence_pages_are_flagged_for_the_reviewer(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.kb_scanned_ingestion.settings.ocr_min_confidence", 45.0)
    ocr = _Ocr([(1, _TITLE, 90.0), (2, "garbled", 30.0)])
    service, ingest, _ = _ingestion(tmp_path, ocr)

    result = _run(service)

    assert result["ocr"]["low_confidence_pages"] == [2]
    reasons = ingest.await_args.kwargs["jurisdiction_metadata"]["review_reasons"]
    assert any("pages [2]" in reason for reason in reasons)


def test_scan_that_does_not_identify_itself_is_held_and_never_indexed(tmp_path):
    service, ingest, propagate = _ingestion(tmp_path, _Ocr([(1, "THE SOME OTHER ACT, 1999", 95.0), (2, "text", 95.0)]))

    result = _run(service)

    assert result["status"] == "held" and "identity" in result["reason"].lower()
    ingest.assert_not_awaited()
    propagate.assert_not_awaited()
    assert json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))["identity_tokens_matched"] == []


def test_missing_ocr_engine_holds_instead_of_indexing_an_empty_document(tmp_path):
    from app.rag.ocr import OcrUnavailableError

    service, ingest, _ = _ingestion(tmp_path, _Ocr(error=OcrUnavailableError("no tesseract")))
    result = _run(service)
    assert result["status"] == "held" and "OCR unavailable" in result["reason"]
    ingest.assert_not_awaited()


def test_dry_run_writes_a_manifest_but_touches_neither_kb_nor_db(tmp_path):
    service, ingest, propagate = _ingestion(tmp_path, _Ocr([(1, _TITLE, 91.0), (2, "x", 91.0)]))
    result = _run(service, dry_run=True)
    assert result["status"] == "dry_run_ok" and result["review_status"] == "needs_review"
    ingest.assert_not_awaited()
    propagate.assert_not_awaited()
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_text_extractable_or_unregistered_pdfs_are_rejected_not_ocred(tmp_path, monkeypatch):
    service, ingest, _ = _ingestion(tmp_path, _Ocr([]))

    class _TextPage:
        def extract_text(self):
            return "A long extractable text layer " * 10

    monkeypatch.setattr(
        "app.services.kb_scanned_ingestion.PdfReader", lambda _stream: SimpleNamespace(pages=[_TextPage()]),
    )
    assert _run(service)["status"] == "rejected"
    normal = OfficialSource("NORMAL", "n.pdf", "https://x.gov.in/n.pdf", "n-act-2000", "l", "bare_act", ("n",))
    assert asyncio.run(service.ingest(normal, _blank_pdf(), acquisition="test"))["status"] == "rejected"
    assert asyncio.run(service.ingest(_SCAN, b"not a pdf", acquisition="test"))["status"] == "rejected"
    ingest.assert_not_awaited()


def test_an_approved_review_status_from_ingestion_is_treated_as_a_bug(tmp_path):
    service, _, propagate = _ingestion(tmp_path, _Ocr([(1, _TITLE, 91.0), (2, "x", 91.0)]), review_status="approved")
    with pytest.raises(RuntimeError, match="needs_review"):
        _run(service)
    propagate.assert_not_awaited()
