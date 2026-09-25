import asyncio

from app.rag.metadata import MetadataExtractor

# Real text (BNSS_2023_Official_Gazette.pdf, section 173 -- FIR registration).
# The official gazette prints this section with no title/dash before the
# body ("173. (1) Every information...", not "173. Information in
# cognizable cases.—(1) Every information..." the way India Code's cleaned-up
# text does), and the same chunk carries a mid-sentence cross-reference to a
# DIFFERENT act's section 64. Confirmed live: before this fix, every chunk of
# this document fell through to the bare "section N" fallback and picked up
# whichever cross-reference happened to appear first -- 11 chunks whose real
# text was sections 173/181/223/397/etc. all got stamped "64".
_BNSS_S173_TEXT = (
    "ignoring or\ndisregarding to conform to any direction given by him under sub-section (1) and may either\n"
    "take such person before a Magistrate or, in petty cases, release him as soon as possible\n"
    "within a period of twenty-four hours.\n"
    "CHAPTER XIII\n"
    "INFORMATION TO THE POLICE AND THEIR POWERS TO INVESTIGATE\n"
    "173. (1) Every information relating to the commission of a cognizable offence,\n"
    "irrespective of the area where the offence is committed, may be given orally or by electronic\n"
    "communication to an officer in charge of a police station, and if given—\n"
    "(i) orally, it shall be reduced to writing by him or under his direction, and be read\n"
    "over to the informant; and every such information, whether given in writing or reduced to\n"
    "writing as aforesaid, shall be signed by the person giving it.\n"
    "Provided that if the information is given by the woman against whom an offence under\n"
    "section 64, section 65, section 66, section 67, section 68, section 69, section 70 or section 71\n"
    "of the Bharatiya Nyaya Sanhita, 2023 is alleged to have been committed or attempted, then\n"
    "such information shall be recorded, by a woman police officer or any woman officer."
)


def test_section_with_no_title_is_read_from_its_own_heading_not_a_cross_reference() -> None:
    extractor = MetadataExtractor()

    metadata = asyncio.run(
        extractor.extract(_BNSS_S173_TEXT, {"source_document": "BNSS_2023_Official_Gazette.pdf"})
    )

    assert metadata["section_number"] == "173"
    assert metadata["section_number_provenance"] == "heading"


# ---------------------------------------------------------------------------
# `act_name` extraction: three real, live-confirmed misdetections, found while
# investigating why questions about content that IS in the Knowledge Base
# (Code of Civil Procedure, Negotiable Instruments Act) were answered
# "no verified document available" -- the chunks existed and were approved,
# but `act_name` had been stamped with the wrong value, which breaks the
# exact-citation retrieval path (`LegalRetriever._named_section_citation`
# filters on `act_name`, and a wrong value simply never matches).
# ---------------------------------------------------------------------------

# Real text (CPC_1908_Official.pdf's own preamble/short-title clause).
_CPC_PREAMBLE_TEXT = (
    "An Act to consolidate and amend the laws relating to the procedure of the Courts of Civil "
    "Judicature. WHEREAS it is expedient to consolidate and amend the laws relating to the "
    "procedure of the Courts of Civil Judicature; It is hereby enacted as follows:-- "
    "1. Short title, commencement and extent.—(1) This Act may be called the Code of Civil "
    "Procedure, 1908."
)

# Real text (a later chunk of the same document, deep in the operative text,
# with no title anywhere in it -- only the Act's own short-form
# self-reference "the Code").
_CPC_OPERATIVE_TEXT_NO_TITLE = (
    "the plaintiff or as the Court thinks fit having regard to the provisions of the Code and any "
    "orders passed thereunder shall be final and shall not be appealed against."
)

# Real text (NEGOTIABLE_INSTRUMENTS_ACT_1881_Official.pdf). The chunk's only
# "Act"-shaped phrase before the real title is an amendment footnote whose
# own leading verb ("Added") is what the regex actually captures.
_NI_ACT_AMENDMENT_FOOTNOTE_TEXT = (
    "67. Presentment for payment of promissory note payable by instalments.—A promissory note "
    "payable by instalments must be presented for payment on the third day after the date fixed "
    "for payment of each instalment; and non-payment on such presentment has the same effect as "
    "non-payment of a note at maturity.\n"
    "1. Added by Act 2 of 1885, s. 4.\n"
    "2. Subs. by Act 12 of 1921, s. 2, for “twenty-four”."
)

# Real text (same document, a different chunk): the schedule's repeal note
# cites an entirely different, but genuine, Act ("the Repealing and Amending
# Act, 1891") as the authority that repealed the schedule -- immediately
# followed, later in the SAME chunk, by the document's own real title.
_NI_ACT_REPEAL_NOTE_THEN_REAL_TITLE_TEXT = (
    "SCHEDULE.—[Enactments repealed]. Rep. by the Repealing and Amending Act, 1891 (12 of 1891), "
    "s. 2 and Schedule I.\n"
    "THE NEGOTIABLE INSTRUMENTS ACT, 1881\n"
    "ACT NO. 26 OF 1881 [9th December, 1881.]\n"
    "An Act to define and amend the law relating to Promissory Notes, Bills of Exchange."
)


def test_code_of_civil_procedure_is_detected_from_its_own_short_title_clause() -> None:
    # `ACT_NAME_RE` requires the anchor keyword (Act/Sanhita/Adhiniyam/Code)
    # to be the LAST word of the title -- structurally unable to match "Code
    # of Civil Procedure", where "Code" comes FIRST. See `_CODE_OF_PROCEDURE_RE`.
    metadata = asyncio.run(
        MetadataExtractor().extract(_CPC_PREAMBLE_TEXT, {"source_document": "CPC_1908_Official.pdf"})
    )
    assert metadata["act_name"] == "Code of Civil Procedure, 1908"


def test_a_bare_self_reference_to_the_code_is_never_mistaken_for_the_documents_name() -> None:
    # Previously: with no real title in view, the bare self-reference "the
    # Code" was the only "...Code" phrase found and got accepted as the
    # act_name outright -- `_is_generic_act_name("The Code")` had a gap that
    # let it through. Correct behaviour is to find NOTHING here; the real
    # name comes from whichever chunk actually contains the title (or the
    # document-level extraction pass), never from this one.
    metadata = asyncio.run(
        MetadataExtractor().extract(_CPC_OPERATIVE_TEXT_NO_TITLE, {"source_document": "CPC_1908_Official.pdf"})
    )
    assert "act_name" not in metadata


def test_an_amendment_footnotes_own_leading_verb_is_not_mistaken_for_the_documents_name() -> None:
    metadata = asyncio.run(
        MetadataExtractor().extract(
            _NI_ACT_AMENDMENT_FOOTNOTE_TEXT, {"source_document": "NEGOTIABLE_INSTRUMENTS_ACT_1881_Official.pdf"}
        )
    )
    assert "act_name" not in metadata


def test_a_repeal_notes_cited_act_is_skipped_in_favour_of_the_real_title_later_in_the_chunk() -> None:
    metadata = asyncio.run(
        MetadataExtractor().extract(
            _NI_ACT_REPEAL_NOTE_THEN_REAL_TITLE_TEXT,
            {"source_document": "NEGOTIABLE_INSTRUMENTS_ACT_1881_Official.pdf"},
        )
    )
    assert metadata["act_name"] == "THE NEGOTIABLE INSTRUMENTS ACT"


def test_all_caps_gazette_titles_still_match_unaffected_by_the_amendment_footnote_guards() -> None:
    text = "THE BHARATIYA NYAYA SANHITA, 2023 An Act to consolidate and amend the provisions relating to offences."
    metadata = asyncio.run(MetadataExtractor().extract(text, {"source_document": "x.pdf"}))
    assert metadata["act_name"] == "THE BHARATIYA NYAYA SANHITA"


def test_a_true_cross_reference_only_chunk_still_falls_back_honestly() -> None:
    # No heading of any shape actually starts this chunk -- it is genuinely
    # just prose that cites another act's section 64. The fallback value is
    # still the best available guess, but it must stay tagged "fallback" (low
    # confidence), never "heading".
    extractor = MetadataExtractor()
    text = "the offence continues where the accused acted contrary to section 64 of that other Act."

    metadata = asyncio.run(extractor.extract(text, {"source_document": "some_document.pdf"}))

    assert metadata["section_number"] == "64"
    assert metadata["section_number_provenance"] == "fallback"
