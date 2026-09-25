import asyncio
from typing import Any

from app.rag.query_rewriter import SmartQueryRewriter
from app.rag.reranker import LegalReranker
from app.rag.retriever import _EXACT_SECTION_SCORE_FLOOR, LegalRetriever
from app.rag.types import DocumentChunk
from app.rag.vector_store import VectorStore
from app.schemas.common import RetrievedChunk
from app.services.chat_service import ChatService


def test_query_expansion_preserves_original_and_adds_zero_fir_terms() -> None:
    rewriter = SmartQueryRewriter()

    queries = rewriter.expand_queries("What is Zero FIR?", intent="General Legal Query")

    assert queries[0] == "What is Zero FIR?"
    combined = rewriter.combine(queries).lower()
    assert "zero fir" in combined
    assert "police station" in combined
    assert "jurisdiction transfer" in combined
    assert "section 173" in combined


def test_reranker_promotes_specific_zero_fir_chunk_over_generic_fir() -> None:
    reranker = LegalReranker()
    generic = RetrievedChunk(
        chunk_id="generic",
        text="Registration of FIR is mandatory when information discloses a cognizable offence.",
        score=0.30,
        metadata={"source_document": "lalita.pdf", "source_type": "case_law"},
    )
    zero = RetrievedChunk(
        chunk_id="zero",
        text="On receipt of information of cognizable offence, Zero FIR will be registered irrespective of the area.",
        score=0.20,
        metadata={"source_document": "bprd.pdf", "source_type": "commentary"},
    )

    ranked = asyncio.run(reranker.rerank("What is Zero FIR?", [generic, zero], top_k=2))

    assert ranked[0].chunk_id == "zero"


def test_reranker_promotes_bns_fir_provision_for_a_generic_fir_question() -> None:
    # Real corpus text (BNSS_2023_Official_Gazette.pdf) -- confirmed live that
    # without a topic bonus this loses the top_k cut to shorter, more
    # lexically-dense FAQ/commentary chunks that merely mention "FIR".
    reranker = LegalReranker()
    faq = RetrievedChunk(
        chunk_id="faq",
        text="Property dispute cases: people want to register FIRs in the police station of their choice.",
        score=0.30,
        metadata={"source_document": "delhi_police_faq.pdf"},
    )
    provision = RetrievedChunk(
        chunk_id="s173",
        text=(
            "173. (1) Every information relating to the commission of a cognizable offence, irrespective of "
            "the area where the offence is committed, may be given orally or by electronic communication to "
            "an officer in charge of a police station, and if given orally, it shall be reduced to writing."
        ),
        score=0.22,
        metadata={"source_document": "BNSS_2023_Official_Gazette.pdf", "section_number": "64"},
    )

    ranked = asyncio.run(reranker.rerank("BNSS ke under FIR kaise darj karayi jaati hai?", [faq, provision], top_k=2))

    assert ranked[0].chunk_id == "s173"


def test_reranker_promotes_contract_act_s10_for_essential_elements_question() -> None:
    reranker = LegalReranker()
    unrelated = RetrievedChunk(
        chunk_id="unrelated",
        text="A mortgage is a transfer of an interest in specific immovable property for securing payment.",
        score=0.28,
        metadata={"source_document": "property.pdf"},
    )
    s10 = RetrievedChunk(
        chunk_id="s10",
        text=(
            "10. What agreements are contracts.—All agreements are contracts if they are made by the free "
            "consent of parties competent to contract, for a lawful consideration and with a lawful object."
        ),
        score=0.20,
        metadata={"source_document": "Indian_Contract_Act_1872_CAG_Official.pdf", "section_number": "10"},
    )

    ranked = asyncio.run(
        reranker.rerank("Valid contract ke liye kya essential elements hote hain?", [unrelated, s10], top_k=2)
    )

    assert ranked[0].chunk_id == "s10"


def test_reranker_promotes_consumer_protection_s69_for_time_limit_question() -> None:
    reranker = LegalReranker()
    unrelated = RetrievedChunk(
        chunk_id="unrelated",
        text="A trader who imports goods for resale must register under the applicable state law.",
        score=0.28,
        metadata={"source_document": "trade.pdf"},
    )
    s69 = RetrievedChunk(
        chunk_id="s69",
        text=(
            "69. Limitation period.—(1) The District Commission, the State Commission or the National "
            "Commission shall not admit a complaint unless it is filed within two years from the date on "
            "which the cause of action has arisen."
        ),
        score=0.20,
        metadata={"source_document": "The_Consumer_Protection_Act_2019_5.pdf", "section_number": "69"},
    )

    ranked = asyncio.run(
        reranker.rerank("Consumer complaint file karne ke liye kitna time limit hai?", [unrelated, s69], top_k=2)
    )

    assert ranked[0].chunk_id == "s69"


def test_reranker_does_not_boost_a_bare_landlord_mention_in_an_unrelated_schedule() -> None:
    # QA pass 2026-09-24 (T026/T027/T028): a single bare "landlord" mention
    # used to be enough for the security-deposit topic bonus, letting an
    # unrelated stamp-duty schedule (real corpus text: "...the landlord's
    # share of cesses..." inside an Indian Stamp Act instrument list) score
    # ahead of the genuine Bombay Rent Control Act tenancy text -- confirmed
    # live via direct pipeline replication (0.41 vs 0.38).
    reranker = LegalReranker()
    stamp_schedule = RetrievedChunk(
        chunk_id="stamp", text="the landlord's share of cesses is deemed to be part of the rent for stamp duty.",
        score=0.016, metadata={"source_document": "stamp_act.pdf", "section_number": "26"},
    )
    tenancy_text = RetrievedChunk(
        chunk_id="tenancy",
        text="If a landlord takes a deposit from a tenant that is not legally permitted, the tenant may recover it.",
        score=0.016, metadata={"source_document": "bombay_rent_act.pdf"},
    )

    ranked = asyncio.run(
        reranker.rerank(
            "mera landlord mera security deposit wapas nahi de raha hai", [stamp_schedule, tenancy_text], top_k=2
        )
    )

    assert ranked[0].chunk_id == "tenancy"


def test_reranker_promotes_electronic_evidence_provision_over_faq_commentary() -> None:
    # QA pass 2026-09-24 (T074): a Delhi Police Academy FAQ document about
    # the new criminal laws generally used to outrank the Bharatiya Sakshya
    # Adhiniyam's own section 63 (the electronic-evidence admissibility
    # provision) -- confirmed live via direct pipeline replication against
    # the real corpus -- because the FAQ's own running page header repeats
    # the Act's full name, and neither the reranker nor the relevance-gate
    # reorder step had a way to recognize the statute's OWN operative text
    # as more authoritative than commentary that merely names the Act.
    reranker = LegalReranker()
    faq = RetrievedChunk(
        chunk_id="faq",
        text=(
            "What will be the format of Chain of custody memo in case of digital evidence required? "
            "The Bharatiya Sakshya Adhiniyam"
        ),
        score=0.016,
        metadata={"source_document": "Delhi_Police_Academy_FAQs_New_Criminal_Laws.pdf", "source_type": "faq"},
    )
    section63 = RetrievedChunk(
        chunk_id="s63",
        text=(
            "63.(1) Notwithstanding anything contained in this Adhiniyam, any information contained in an "
            "electronic record which is printed on paper, stored, recorded or copied in optical or magnetic "
            "media or semiconductor memory which is produced by a computer output shall be deemed to be also "
            "a document, if the conditions mentioned in this section are satisfied."
        ),
        score=0.016,
        metadata={
            "source_document": "The_Bhara_Tiy_A_Sakshy_A_Adhiniy_Am_2023_6.pdf",
            "section_number": "63",
            "source_type": "definition",
        },
    )

    ranked = asyncio.run(
        reranker.rerank(
            "admissibility of electronic evidence Bharatiya Sakshya Adhiniyam", [faq, section63], top_k=2
        )
    )

    assert ranked[0].chunk_id == "s63"


def test_reorder_by_relevance_does_not_let_named_act_repetition_override_the_statutes_own_text() -> None:
    # Companion to the reranker test above: even once reranking correctly
    # ranks the statute's own text first, `reorder_by_relevance`'s lexical
    # sanity check used to override it back to whichever chunk merely
    # repeats the Act's own name the most (a page header, in the live
    # case) -- see `app.rag.relevance._drop_named_instrument_terms`.
    from app.rag.relevance import reorder_by_relevance

    statute_first = RetrievedChunk(
        chunk_id="s63", text="electronic record computer output document", score=0.58,
        metadata={"source_document": "bsa.pdf", "section_number": "63"},
    )
    faq_header_only = RetrievedChunk(
        chunk_id="faq", text="The Bharatiya Sakshya Adhiniyam The Bharatiya Sakshya Adhiniyam admissibility",
        score=0.36, metadata={"source_document": "faq.pdf"},
    )

    reordered = reorder_by_relevance(
        "admissibility of electronic evidence Bharatiya Sakshya Adhiniyam",
        [statute_first, faq_header_only],
    )

    assert reordered[0].chunk_id == "s63"


def test_reorder_by_relevance_still_promotes_genuine_off_topic_overlap() -> None:
    # `_drop_named_instrument_terms` must only strip the NAMED Act's own
    # words -- a query with no Act name at all keeps the exact prior
    # behavior of preferring genuinely higher lexical overlap.
    from app.rag.relevance import reorder_by_relevance

    reranked_first = RetrievedChunk(
        chunk_id="wrong", text="mortgage transfer of immovable property", score=0.30,
        metadata={"source_document": "property.pdf"},
    )
    genuinely_on_topic = RetrievedChunk(
        chunk_id="right", text="zero fir police station jurisdiction cognizable offence", score=0.20,
        metadata={"source_document": "fir.pdf"},
    )

    reordered = reorder_by_relevance("what is zero fir jurisdiction", [reranked_first, genuinely_on_topic])

    assert reordered[0].chunk_id == "right"


def test_relevance_gate_rejects_off_topic_chunk_before_llm_context() -> None:
    service = ChatService()
    off_topic = RetrievedChunk(
        chunk_id="mortgage",
        text="A mortgage is a transfer of an interest in specific immovable property for securing payment.",
        score=0.22,
        metadata={"source_document": "property.pdf"},
    )
    on_topic = RetrievedChunk(
        chunk_id="deposit",
        text="A rent agreement may require the landlord to refund the tenant security deposit after possession is handed over.",
        score=0.18,
        metadata={"source_document": "rent_agreement.txt"},
    )

    accepted = service._filter_relevant_context(
        "PG deposit nahi mil raha",
        "PG deposit nahi mil raha security deposit tenant landlord rental deposit refund rent agreement property law",
        [off_topic, on_topic],
    )

    assert [chunk.chunk_id for chunk in accepted] == ["deposit"]


def test_relevance_gate_accepts_bnss_fir_provision_despite_low_score() -> None:
    service = ChatService()
    off_topic = RetrievedChunk(
        chunk_id="mortgage",
        text="A mortgage is a transfer of an interest in specific immovable property for securing payment.",
        score=0.22,
        metadata={"source_document": "property.pdf"},
    )
    provision = RetrievedChunk(
        chunk_id="s173",
        text=(
            "173. (1) Every information relating to the commission of a cognizable offence may be given "
            "orally or by electronic communication to an officer in charge of a police station, and if given "
            "orally, it shall be reduced to writing."
        ),
        score=0.20,
        metadata={"source_document": "BNSS_2023_Official_Gazette.pdf"},
    )

    accepted = service._filter_relevant_context(
        "BNSS ke under FIR kaise darj karayi jaati hai?", "", [off_topic, provision],
    )

    assert [chunk.chunk_id for chunk in accepted] == ["s173"]


def test_fir_gate_override_does_not_fire_on_unrelated_words_containing_fir() -> None:
    # "fir" is a substring of "first", "firm", "confirm" -- the override must
    # match the word "FIR" itself, not any query that happens to contain it.
    service = ChatService()
    off_topic = RetrievedChunk(
        chunk_id="off_topic",
        text="A mortgage is a transfer of an interest in specific immovable property for securing payment.",
        score=0.20,
        metadata={"source_document": "property.pdf"},
    )

    accepted = service._filter_relevant_context(
        "What is the first step to confirm a firm's registration?", "", [off_topic],
    )

    assert accepted == []


def test_banking_fraud_gate_does_not_accept_generic_rbi_recovery_material() -> None:
    service = ChatService()
    recovery = RetrievedChunk(
        chunk_id="recovery",
        text="The Reserve Bank of India says regulated entities are responsible for recovery agents.",
        score=0.34,
        metadata={"source_document": "rbi_recovery.html"},
    )
    banking = RetrievedChunk(
        chunk_id="banking",
        text="In an unauthorised electronic banking transaction, customer liability depends on timely reporting.",
        score=0.25,
        metadata={"source_document": "rbi_customer_liability.pdf"},
    )

    accepted = service._filter_relevant_context(
        "Unauthorized bank transaction",
        "Unauthorized bank transaction unauthorized electronic banking transaction UPI wrong transfer RBI customer liability",
        [recovery, banking],
    )

    assert [chunk.chunk_id for chunk in accepted] == ["banking"]


class _FakeEmbeddings:
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class _FakeVectorStore(VectorStore):
    """Records every `find_by_section_number` call's Act filter so the test
    can assert retrieval actually SCOPED the exact-match lookup to the named
    Act, not just that the final scores happened to come out right."""

    def __init__(self, search_results: list[RetrievedChunk], section_matches: list[RetrievedChunk]) -> None:
        self._search_results = search_results
        self._section_matches = section_matches
        self.section_lookup_calls: list[tuple[str, list[str] | None]] = []

    async def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    async def search(
        self, query_embedding: list[float], query: str, top_k: int, filters: dict[str, Any],
        mode: str = "hybrid",
    ) -> list[RetrievedChunk]:
        return list(self._search_results)

    async def delete_by_source(self, source_document: str) -> int:
        raise NotImplementedError

    async def find_by_section_number(self, section_number: str, filters: dict[str, Any]) -> list[RetrievedChunk]:
        act_filter = filters.get("act_name")
        self.section_lookup_calls.append((section_number, act_filter))
        if act_filter:
            return [chunk for chunk in self._section_matches if chunk.metadata.get("act_name") in act_filter]
        return list(self._section_matches)


def test_named_act_disambiguates_same_numbered_chunks_from_different_acts() -> None:
    # Regression for the "BNS 318" bug: BNS Section 318 ("Cheating") and BNSS
    # Section 318 ("Record in High Court") are two different, real
    # provisions that happen to share a number. Confirmed live before this
    # fix: both were floored to the IDENTICAL exact-match score (0.6175 ==
    # 0.6175 post-rerank) for the query "BNS 318" -- the final answer was
    # only correct because the LLM happened to prefer the right chunk, not
    # because retrieval ever picked one.
    bns_metadata = {"act_name": "The Bharatiya Nyaya Sanhita", "section_number": "318"}
    bnss_metadata = {"act_name": "The Bharatiya Nagarik Suraksha Sanhita", "section_number": "318"}
    bns_chunk = RetrievedChunk(chunk_id="bns-318", text="318. Cheating.", score=0.0, metadata=bns_metadata)
    bnss_chunk = RetrievedChunk(chunk_id="bnss-318", text="318. Record in High Court.", score=0.0, metadata=bnss_metadata)
    # Raw (RRF-fused) retrieval already contains both at a low, near-tied
    # score -- mirrors the real corpus, where neither chunk's own text
    # repeats its Act's name, so similarity search barely separates them.
    raw_results = [
        RetrievedChunk(chunk_id="bns-318", text=bns_chunk.text, score=0.02, metadata=bns_metadata),
        RetrievedChunk(chunk_id="bnss-318", text=bnss_chunk.text, score=0.019, metadata=bnss_metadata),
    ]
    fake_store = _FakeVectorStore(search_results=raw_results, section_matches=[bns_chunk, bnss_chunk])
    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=fake_store)

    _, results = asyncio.run(retriever.retrieve("BNS 318", top_k=10, filters={}, intent="SECTION_LOOKUP"))

    by_id = {chunk.chunk_id: chunk for chunk in results}
    assert by_id["bns-318"].score == _EXACT_SECTION_SCORE_FLOOR
    # The wrong-Act chunk must never receive the exact-match floor.
    assert by_id["bnss-318"].score < _EXACT_SECTION_SCORE_FLOOR
    # The exact-match lookup that hit must have been scoped to the named Act.
    # Scoped means "only spellings of THIS Act" -- not "exactly one spelling".
    # `_ACT_ABBREVIATION_TO_METADATA_NAMES` deliberately passes every
    # `act_name` variant the live corpus actually stores for BNS (verified
    # against it on 2026-08-25: the bare-act document is filed under
    # "Bharatiya Nyaya Sanhita" and the FAQ document under "Bhartiya Nyay
    # Sanhita", so pinning this to the single canonical spelling matched zero
    # chunks and silently fell through to the unscoped, ambiguous lookup this
    # test exists to prevent).
    section_number, act_names = fake_store.section_lookup_calls[0]
    assert section_number == "318"
    assert "The Bharatiya Nyaya Sanhita" in act_names
    # "BHARA TIY A NY A Y A SANHITA" (added 2026-09-16, a fourth real corpus
    # variant discovered once that document was approved out of
    # needs_review) has "Nyaya" itself further garbled into "NY A Y A", so a
    # plain substring check on "Nyay" can no longer hold for every listed
    # variant -- every one must still at least carry "Sanhita" (BNS's own
    # generic instrument noun), and the un-garbled ones must still say "Nyay".
    assert all("Sanhita" in name or "SANHITA" in name for name in act_names), act_names
    assert all("Nyay" in name for name in act_names if "TIY A" not in name), act_names


def test_full_act_name_disambiguates_same_numbered_chunks_from_different_acts() -> None:
    """Security finding C6, live QA repro: a question explicitly naming the
    Bharatiya Nyaya Sanhita IN FULL (not the "BNS" abbreviation) was
    answered from Bharatiya Nagarik Suraksha Sanhita (BNSS) Section 302 -- a
    different, real provision sharing the same number. Root cause:
    `parse_section_lookup_act` only recognised the closed abbreviation
    vocabulary (BNS/BNSS/BSA/...), so a fully-spelled-out Act name found no
    match, `_apply_section_number_floor` fell through to the Act-agnostic
    lookup, and both Acts' Section 302 chunks tied at the identical exact-
    match floor -- functionally identical to the pre-fix "BNS 318" bug
    `test_named_act_disambiguates_same_numbered_chunks_from_different_acts`
    covers, just reached through the full name instead of the abbreviation.
    """
    bns_metadata = {"act_name": "The Bharatiya Nyaya Sanhita", "section_number": "302"}
    bnss_metadata = {"act_name": "The Bharatiya Nagarik Suraksha Sanhita", "section_number": "302"}
    bns_chunk = RetrievedChunk(chunk_id="bns-302", text="302. Murder.", score=0.0, metadata=bns_metadata)
    bnss_chunk = RetrievedChunk(
        chunk_id="bnss-302", text="302. Procedure on receiving report.", score=0.0, metadata=bnss_metadata,
    )
    raw_results = [
        RetrievedChunk(chunk_id="bns-302", text=bns_chunk.text, score=0.02, metadata=bns_metadata),
        RetrievedChunk(chunk_id="bnss-302", text=bnss_chunk.text, score=0.019, metadata=bnss_metadata),
    ]
    fake_store = _FakeVectorStore(search_results=raw_results, section_matches=[bns_chunk, bnss_chunk])
    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=fake_store)

    _, results = asyncio.run(retriever.retrieve(
        "What is Section 302 of the Bharatiya Nyaya Sanhita?", top_k=10, filters={}, intent="SECTION_LOOKUP",
    ))

    by_id = {chunk.chunk_id: chunk for chunk in results}
    assert by_id["bns-302"].score == _EXACT_SECTION_SCORE_FLOOR
    # The wrong-Act chunk must never receive the exact-match floor, even
    # though it was returned alongside the right one by the underlying
    # (filter-blind, in this fake) search -- this is the per-chunk `act_name`
    # check added for this fix, not the `find_by_section_number` scoping
    # `test_named_act_disambiguates_same_numbered_chunks_from_different_acts`
    # exercises (a different code path: this query matches
    # `_named_section_citation`, not the bare-number `SECTION_LOOKUP` path).
    assert by_id["bnss-302"].score < _EXACT_SECTION_SCORE_FLOOR


def test_rape_concept_query_forces_both_definition_and_punishment_sections() -> None:
    """Live repro: "which section of BNS talks about rape?" answered from
    Section 64 (punishment) alone -- Section 63 (the definition) never
    entered the top-8 similarity-fused candidates, so the answer covered
    only the punishment with no definition. Both are verified BNS sections
    (`RAPE_BNS_SECTION_NUMBERS`); the concept-merge must force both present
    and scoped to BNS specifically (never BSA's own unrelated Section 63).
    """
    bns_63 = RetrievedChunk(chunk_id="bns-63", text="63. Rape.", score=0.0, metadata={
        "act_name": "The Bharatiya Nyaya Sanhita", "section_number": "63",
    })
    bns_64 = RetrievedChunk(chunk_id="bns-64", text="64. Punishment for rape.", score=0.0, metadata={
        "act_name": "The Bharatiya Nyaya Sanhita", "section_number": "64",
    })
    bsa_63 = RetrievedChunk(chunk_id="bsa-63", text="63. Admissibility of electronic records.", score=0.0, metadata={
        "act_name": "The Bharatiya Sakshya Adhiniyam", "section_number": "63",
    })
    fake_store = _FakeVectorStore(
        search_results=[RetrievedChunk(chunk_id="noise", text="unrelated", score=0.03, metadata={})],
        section_matches=[bns_63, bns_64, bsa_63],
    )
    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=fake_store)

    _, results = asyncio.run(
        retriever.retrieve("which section of bns talks about rape", top_k=8, filters={})
    )

    by_id = {chunk.chunk_id: chunk for chunk in results}
    assert by_id["bns-63"].score == _EXACT_SECTION_SCORE_FLOOR
    assert by_id["bns-64"].score == _EXACT_SECTION_SCORE_FLOOR
    assert "bsa-63" not in by_id


def test_theft_concept_query_forces_bns_section_303_present() -> None:
    """Live repro: "theft ki definition batao BNS par" returned "no verified
    document" even after BNS's own document was approved -- Section 303
    never entered the top-8 similarity-fused candidates for the same reason
    FIR's Section 173 didn't (shares more incidental penalty vocabulary with
    unrelated Acts than theft-specific vocabulary with the query).
    """
    bns_303 = RetrievedChunk(chunk_id="bns-303", text="303. Theft.", score=0.0, metadata={
        "act_name": "BHARA TIY A NY A Y A SANHITA", "section_number": "303",
    })
    fake_store = _FakeVectorStore(
        search_results=[RetrievedChunk(chunk_id="noise", text="unrelated", score=0.03, metadata={})],
        section_matches=[bns_303],
    )
    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=fake_store)

    _, results = asyncio.run(retriever.retrieve("theft ki definition batao BNS par", top_k=8, filters={}))

    by_id = {chunk.chunk_id: chunk for chunk in results}
    assert by_id["bns-303"].score == _EXACT_SECTION_SCORE_FLOOR


def test_unnamed_act_bare_number_lookup_is_unchanged() -> None:
    # No Act named ("Section 318") -- `_apply_section_number_floor` must
    # fall back to today's Act-agnostic lookup unchanged, still floors every
    # exact section match (the pre-existing, documented multi-Act ambiguity
    # for a truly bare number is explicitly out of scope, not reintroduced
    # or worsened here).
    bns_metadata = {"act_name": "The Bharatiya Nyaya Sanhita", "section_number": "318"}
    bnss_metadata = {"act_name": "The Bharatiya Nagarik Suraksha Sanhita", "section_number": "318"}
    bns_chunk = RetrievedChunk(chunk_id="bns-318", text="318. Cheating.", score=0.0, metadata=bns_metadata)
    bnss_chunk = RetrievedChunk(chunk_id="bnss-318", text="318. Record in High Court.", score=0.0, metadata=bnss_metadata)
    raw_results = [
        RetrievedChunk(chunk_id="bns-318", text=bns_chunk.text, score=0.02, metadata=bns_metadata),
        RetrievedChunk(chunk_id="bnss-318", text=bnss_chunk.text, score=0.019, metadata=bnss_metadata),
    ]
    fake_store = _FakeVectorStore(search_results=raw_results, section_matches=[bns_chunk, bnss_chunk])
    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=fake_store)

    _, results = asyncio.run(retriever.retrieve("Section 318", top_k=10, filters={}, intent="SECTION_LOOKUP"))

    by_id = {chunk.chunk_id: chunk for chunk in results}
    assert by_id["bns-318"].score == _EXACT_SECTION_SCORE_FLOOR
    assert by_id["bnss-318"].score == _EXACT_SECTION_SCORE_FLOOR
    assert fake_store.section_lookup_calls[0] == ("318", None)


def test_bare_number_with_recent_conversation_act_hint_breaks_the_tie() -> None:
    # "Section 2" alone can't name an Act (every Act numbers its own
    # provisions independently), but if the last couple of turns already
    # named one ("...Consumer Protection Act...") that's a real disambiguation
    # signal this app has -- unlike the prior tie, it now gets a small bonus
    # over the other, unrelated Act's same-numbered chunk.
    consumer_metadata = {"act_name": "Consumer Protection Act, 2019", "section_number": "2"}
    it_metadata = {"act_name": "Information Technology Act", "section_number": "2"}
    consumer_chunk = RetrievedChunk(chunk_id="consumer-2", text="2. Definitions.", score=0.0, metadata=consumer_metadata)
    it_chunk = RetrievedChunk(chunk_id="it-2", text="2. Definitions.", score=0.0, metadata=it_metadata)
    raw_results = [
        RetrievedChunk(chunk_id="consumer-2", text=consumer_chunk.text, score=0.02, metadata=consumer_metadata),
        RetrievedChunk(chunk_id="it-2", text=it_chunk.text, score=0.019, metadata=it_metadata),
    ]
    fake_store = _FakeVectorStore(search_results=raw_results, section_matches=[consumer_chunk, it_chunk])
    retriever = LegalRetriever(embeddings=_FakeEmbeddings(), vector_store=fake_store)

    _, results = asyncio.run(
        retriever.retrieve(
            "Section 2", top_k=10, filters={}, intent="SECTION_LOOKUP",
            context_hint="What does the Consumer Protection Act say about unfair trade practices?",
        )
    )

    by_id = {chunk.chunk_id: chunk for chunk in results}
    assert by_id["consumer-2"].score > _EXACT_SECTION_SCORE_FLOOR
    assert by_id["it-2"].score == _EXACT_SECTION_SCORE_FLOOR
    assert by_id["consumer-2"].score > by_id["it-2"].score
