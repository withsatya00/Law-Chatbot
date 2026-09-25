"""Phase 1 "Jurisdiction-Aware Knowledge Base": unit tests for
`app.rag.kb_jurisdiction` -- normalization/validation of a document's
jurisdiction metadata, section-level overrides, and the shared-retrieval gate.

Pure-function tests, no Mongo/pipeline involved (mirrors `test_bm25_index.py`'s
"none of this touches MongoDB" convention).
"""

from app.rag import kb_jurisdiction as kj


def test_valid_central_metadata_normalizes_to_approved() -> None:
    """A fully-described, explicitly-verified Central Act with a real source
    URL has no review reasons and is approved for shared retrieval."""
    normalized = kj.normalize_jurisdiction(
        {
            "issuing_level": "central",
            "applicability": "all_india",
            "jurisdiction_source_type": "bare_act",
            "source_url": "https://www.indiacode.nic.in/example",
            "verification_status": "verified",
            "verified_by": "admin@example.com",
        }
    )
    assert normalized.metadata["issuing_level"] == "central"
    assert normalized.metadata["applicability"] == "all_india"
    assert normalized.metadata["applicable_state_codes"] == []
    assert normalized.metadata["verification_status"] == "verified"
    assert normalized.metadata["last_verified_at"] is not None
    assert normalized.metadata["review_status"] == kj.REVIEW_APPROVED
    assert normalized.metadata["review_reasons"] == []


def test_valid_state_metadata_normalizes_with_state_codes() -> None:
    """A State Act applying to Maharashtra and Karnataka, State/UT names and
    codes both accepted, both normalized to codes."""
    normalized = kj.normalize_jurisdiction(
        {
            "issuing_level": "state",
            "applicability": "specific_states",
            "applicable_state_codes": ["Maharashtra", "ka"],
            "jurisdiction_source_type": "bare_act",
            "source_url": "https://example.gov.in/act",
            "verification_status": "verified",
            "verified_by": "admin@example.com",
        }
    )
    assert normalized.metadata["applicable_state_codes"] == ["KA", "MH"]
    assert normalized.metadata["review_status"] == kj.REVIEW_APPROVED


def test_central_issuing_level_does_not_imply_all_india_applicability() -> None:
    """Rule 1: a Central Act with no applicability stated stays `unknown` --
    never silently defaulted to `all_india` just because Parliament enacted
    it (the Model Tenancy Act, 2021 is exactly this shape in production:
    Central, but binding nobody until a State adopts it)."""
    normalized = kj.normalize_jurisdiction({"issuing_level": "central"})
    assert normalized.metadata["issuing_level"] == "central"
    assert normalized.metadata["applicability"] == kj.APPLICABILITY_UNKNOWN
    assert normalized.metadata["review_status"] == kj.REVIEW_NEEDS_REVIEW
    assert "Applicability is unknown." in normalized.metadata["review_reasons"]


def test_missing_metadata_defaults_to_unknown_and_needs_review_never_fabricated() -> None:
    """Rule 2: nothing is guessed. An empty submission resolves to `unknown`/
    `unverified` on every field, dates stay `None`, and the record is gated
    out of shared retrieval -- never assigned a plausible-looking default."""
    normalized = kj.normalize_jurisdiction({})
    assert normalized.metadata["issuing_level"] == kj.ISSUING_LEVEL_UNKNOWN
    assert normalized.metadata["applicability"] == kj.APPLICABILITY_UNKNOWN
    assert normalized.metadata["effective_from"] is None
    assert normalized.metadata["effective_to"] is None
    assert normalized.metadata["verification_status"] == kj.VERIFICATION_UNVERIFIED
    assert normalized.metadata["review_status"] == kj.REVIEW_NEEDS_REVIEW


def test_unknown_state_code_is_rejected() -> None:
    try:
        kj.normalize_jurisdiction({"applicability": "specific_states", "applicable_state_codes": ["ZZ"]})
        raise AssertionError("expected JurisdictionMetadataError")
    except kj.JurisdictionMetadataError as exc:
        assert any("ZZ" in error for error in exc.errors)


def test_specific_states_without_any_state_code_is_rejected() -> None:
    try:
        kj.normalize_jurisdiction({"applicability": "specific_states"})
        raise AssertionError("expected JurisdictionMetadataError")
    except kj.JurisdictionMetadataError as exc:
        assert any("specific_states" in error for error in exc.errors)


def test_effective_to_before_effective_from_is_rejected() -> None:
    try:
        kj.normalize_jurisdiction({"effective_from": "2020-01-01", "effective_to": "2019-01-01"})
        raise AssertionError("expected JurisdictionMetadataError")
    except kj.JurisdictionMetadataError as exc:
        assert any("effective_to" in error for error in exc.errors)


def test_malformed_date_is_rejected_not_coerced() -> None:
    try:
        kj.normalize_jurisdiction({"effective_from": "sometime in 2019"})
        raise AssertionError("expected JurisdictionMetadataError")
    except kj.JurisdictionMetadataError as exc:
        assert any("effective_from" in error for error in exc.errors)


def test_ai_inferred_metadata_can_never_be_marked_verified() -> None:
    """Rule 3: whatever an extractor/regex/LLM pipeline claims, provenance
    `inferred` caps `verification_status` at `inferred` -- never `verified`,
    which is reserved for a named human's explicit say-so."""
    normalized = kj.normalize_jurisdiction(
        {"issuing_level": "central", "applicability": "all_india", "verification_status": "verified"},
        provenance=kj.PROVENANCE_INFERRED,
    )
    assert normalized.metadata["verification_status"] == kj.VERIFICATION_INFERRED
    assert normalized.metadata["review_status"] == kj.REVIEW_NEEDS_REVIEW


def test_section_overrides_normalize_and_reject_unknown_fields() -> None:
    normalized = kj.normalize_jurisdiction(
        {
            "issuing_level": "central",
            "applicability": "all_india",
            "jurisdiction_source_type": "bare_act",
            "source_url": "https://example.gov.in/act",
            "verification_status": "verified",
            "verified_by": "admin",
            "section_overrides": {
                "12": {"applicability": "specific_states", "applicable_state_codes": ["mh"], "effective_from": "2021-04-01"}
            },
        }
    )
    assert normalized.section_overrides["12"]["applicability"] == "specific_states"
    assert normalized.section_overrides["12"]["applicable_state_codes"] == ["MH"]

    try:
        kj.normalize_jurisdiction({"section_overrides": {"12": {"document_key": "nope"}}})
        raise AssertionError("expected JurisdictionMetadataError")
    except kj.JurisdictionMetadataError as exc:
        assert any("document_key" in error for error in exc.errors)


def test_apply_to_chunk_copies_document_fields_and_applies_matching_section_override() -> None:
    document_metadata = kj.document_metadata_fields(
        kj.normalize_jurisdiction(
            {
                "issuing_level": "central",
                "applicability": "all_india",
                "jurisdiction_source_type": "bare_act",
                "source_url": "https://example.gov.in/act",
                "verification_status": "verified",
                "verified_by": "admin",
                "section_overrides": {"12": {"applicability": "specific_states", "applicable_state_codes": ["MH"]}},
            }
        )
    )

    section_12_chunk = kj.apply_to_chunk({"section_number": "12"}, document_metadata)
    assert section_12_chunk["issuing_level"] == "central"  # inherited from the document
    assert section_12_chunk["applicability"] == "specific_states"  # section's own override wins
    assert section_12_chunk["applicable_state_codes"] == ["MH"]
    assert section_12_chunk["section_override_applied"] is True

    section_5_chunk = kj.apply_to_chunk({"section_number": "5"}, document_metadata)
    assert section_5_chunk["applicability"] == "all_india"  # no override for this section
    assert "section_override_applied" not in section_5_chunk


def test_shared_retrieval_filters_requires_explicit_approval() -> None:
    """Gap 1 fix: strict `review_status == approved` -- NOT `[None, approved]`.
    A document that has never been reviewed at all (no `review_status` field,
    e.g. every pre-Phase-1 document before the backfill runs) must not pass
    through the shared/unowned branch just because the field is absent."""
    assert kj.shared_retrieval_filters() == {"review_status": kj.REVIEW_APPROVED}
