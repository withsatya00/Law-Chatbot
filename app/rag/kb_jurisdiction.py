"""Phase 1 "Jurisdiction-Aware Knowledge Base": the vocabulary, validation and
propagation rules for a Knowledge Base document's jurisdiction, source and
version metadata.

`app/rag/jurisdiction.py` (deliberately a different module) is a *generation
time* guard: it stops an answer presenting a model law as binding. This module
is the *ingest time* half of the same problem -- recording, for each document
and each searchable chunk, which law-making authority issued it, where it
actually applies, which version it is, and whether a human has verified any of
that.

Three rules the rest of this module exists to enforce, each of which is a way
the naive version gets a user's answer wrong:

1. **A central issuing level does NOT imply all-India applicability.** Parliament
   legislates for the whole Union on Union List subjects, but it also enacts
   laws for a single UT, and a Central "model" law (the Model Tenancy Act, 2021
   -- see `app/rag/jurisdiction.py`) binds nobody at all until a State adopts
   it. `issuing_level` and `applicability` are therefore two independent fields
   and neither is ever derived from the other.
2. **Unknown is a value, not a blank to fill in.** A missing `effective_from`
   stays `None`; it never becomes the ingest date, the file's mtime, or "today".
   A date a user relies on for a limitation period must be the Act's real date
   or absent, never a plausible-looking guess.
3. **Nothing a regex or an LLM inferred is "verified".** `verification_status`
   is only ever `VERIFICATION_VERIFIED` when a named human said so; anything
   derived by the extractor is `VERIFICATION_INFERRED`, which routes to
   `REVIEW_NEEDS_REVIEW` and therefore stays out of shared retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

# ISO 3166-2:IN codes for the 28 States and 8 Union Territories. The code, not
# the name, is what gets stored: names are spelled inconsistently across
# gazette sources ("Odisha"/"Orissa", "Puducherry"/"Pondicherry") and a
# renaming would silently orphan every row keyed on the old string.
STATE_UT_CODES: dict[str, str] = {
    "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh", "AS": "Assam", "BR": "Bihar",
    "CT": "Chhattisgarh", "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana",
    "HP": "Himachal Pradesh", "JH": "Jharkhand", "KA": "Karnataka", "KL": "Kerala",
    "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur", "ML": "Meghalaya",
    "MZ": "Mizoram", "NL": "Nagaland", "OR": "Odisha", "PB": "Punjab",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TG": "Telangana",
    "TR": "Tripura", "UP": "Uttar Pradesh", "UT": "Uttarakhand", "WB": "West Bengal",
    "AN": "Andaman and Nicobar Islands", "CH": "Chandigarh",
    "DH": "Dadra and Nagar Haveli and Daman and Diu", "DL": "Delhi",
    "JK": "Jammu and Kashmir", "LA": "Ladakh", "LD": "Lakshadweep", "PY": "Puducherry",
}
_NAME_TO_CODE: dict[str, str] = {name.lower(): code for code, name in STATE_UT_CODES.items()}

# Which authority enacted the instrument. Independent of `applicability` (rule 1).
ISSUING_LEVEL_CENTRAL = "central"
ISSUING_LEVEL_STATE = "state"
ISSUING_LEVEL_LOCAL = "local"
ISSUING_LEVEL_UNKNOWN = "unknown"
ISSUING_LEVELS = (ISSUING_LEVEL_CENTRAL, ISSUING_LEVEL_STATE, ISSUING_LEVEL_LOCAL, ISSUING_LEVEL_UNKNOWN)

# Where the instrument actually binds.
APPLICABILITY_ALL_INDIA = "all_india"
APPLICABILITY_SPECIFIC_STATES = "specific_states"
APPLICABILITY_UNKNOWN = "unknown"
APPLICABILITIES = (APPLICABILITY_ALL_INDIA, APPLICABILITY_SPECIFIC_STATES, APPLICABILITY_UNKNOWN)

VERIFICATION_VERIFIED = "verified"
VERIFICATION_MACHINE_VERIFIED = "machine_verified"
VERIFICATION_UNVERIFIED = "unverified"
VERIFICATION_INFERRED = "inferred"
VERIFICATION_STATUSES = (
    VERIFICATION_VERIFIED, VERIFICATION_MACHINE_VERIFIED,
    VERIFICATION_UNVERIFIED, VERIFICATION_INFERRED,
)

# How the values got here. `PROVENANCE_INFERRED` can never reach
# `VERIFICATION_VERIFIED` -- see `_resolve_verification`.
PROVENANCE_MANUAL = "manual"
PROVENANCE_AUTOMATED_OFFICIAL = "automated_official"
PROVENANCE_INFERRED = "inferred"
PROVENANCE_UNKNOWN = "unknown"
PROVENANCES = (PROVENANCE_MANUAL, PROVENANCE_AUTOMATED_OFFICIAL, PROVENANCE_INFERRED, PROVENANCE_UNKNOWN)

# The single field shared retrieval gates on. `REVIEW_APPROVED` is the ONLY
# value that reaches the shared corpus; a chunk with no `review_status` key at
# all (everything indexed before this phase) is treated as legacy-visible, so
# adding this field required no reindex and broke no existing installation.
REVIEW_APPROVED = "approved"
REVIEW_NEEDS_REVIEW = "needs_review"
REVIEW_PENDING = "pending"
REVIEW_STATUSES = (REVIEW_APPROVED, REVIEW_NEEDS_REVIEW, REVIEW_PENDING)

SOURCE_TYPE_VALUES = (
    "bare_act", "amendment_act", "rules", "regulation", "notification", "circular",
    "ordinance", "bill", "case_law", "commentary", "faq", "model_law", "unknown",
)

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DOCUMENT_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,120}$")

# Every jurisdiction field, in one place: `IndexingPipeline` copies exactly
# this set from the document record onto each chunk, the backfill writes
# exactly this set, and the admin API returns exactly this set. Adding a field
# in one place and forgetting another is how document and chunk metadata drift
# apart until a filter matches the document but none of its chunks.
JURISDICTION_FIELDS: tuple[str, ...] = (
    "document_key", "issuing_level", "applicability", "applicable_state_codes",
    "applicable_localities", "jurisdiction_source_type", "source_url", "source_page_reference",
    "subsection", "parent_reference", "version_label", "effective_from", "effective_to",
    "amends", "amended_by", "supersedes", "superseded_by", "verification_status",
    "last_verified_at", "verified_by", "metadata_provenance", "review_status",
    "jurisdiction_schema_version", "review_reasons",
    "machine_verification",
)

# Bumped only when the SHAPE changes, so a backfill can tell an already-migrated
# row from one it has never touched (that is what makes re-running it a no-op).
JURISDICTION_SCHEMA_VERSION = 1


class JurisdictionMetadataError(ValueError):
    """Raised for input that cannot be normalized. Carries every problem found,
    not just the first: an admin correcting one field at a time through a form
    is a worse experience than being told all four at once."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class NormalizedJurisdiction:
    metadata: dict[str, Any]
    # `{section_number: {field: value}}` -- only the fields a section actually
    # overrides, never a full copy of the document's values (a full copy would
    # freeze the document's values into each section and stop a later
    # document-level correction from reaching them).
    section_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def review_status(self) -> str:
        return str(self.metadata["review_status"])


def normalize_state_code(value: Any) -> str | None:
    """Accepts a code ("MH", "mh") or a full State/UT name ("Maharashtra"),
    returns the canonical code, or `None` if it is neither. Names are accepted
    because that is what an admin has in front of them on a gazette page; they
    are converted immediately so only codes are ever stored."""
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate:
        return None
    upper = candidate.upper()
    if upper in STATE_UT_CODES:
        return upper
    return _NAME_TO_CODE.get(candidate.lower())


def _parse_date(value: Any, field_name: str, errors: list[str]) -> str | None:
    """Rule 2: an absent or empty date stays `None`. Only a real ISO date is
    accepted -- a partially-known "2015" is rejected rather than silently
    completed to 2015-01-01, which would be a fabricated commencement date."""
    if value in (None, "", "unknown"):
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str) and _ISO_DATE_RE.match(value.strip()):
        try:
            return date.fromisoformat(value.strip()).isoformat()
        except ValueError:
            pass
    errors.append(f"{field_name} must be an ISO date (YYYY-MM-DD) or omitted; got {value!r}.")
    return None


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _enum(value: Any, allowed: tuple[str, ...], field_name: str, default: str, errors: list[str]) -> str:
    if value in (None, ""):
        return default
    candidate = str(value).strip().lower()
    if candidate not in allowed:
        errors.append(f"{field_name} must be one of {', '.join(allowed)}; got {value!r}.")
        return default
    return candidate


def _resolve_verification(raw: Any, provenance: str, errors: list[str]) -> str:
    """Rule 3. `PROVENANCE_INFERRED` is a hard ceiling: whatever the caller
    asked for, extractor output is `VERIFICATION_INFERRED`. Without this, an
    ingestion path that fills defaults from a regex could hand itself the
    `verified` flag and publish straight into shared retrieval."""
    status = _enum(raw, VERIFICATION_STATUSES, "verification_status", VERIFICATION_UNVERIFIED, errors)
    if provenance != PROVENANCE_MANUAL and status == VERIFICATION_VERIFIED:
        return VERIFICATION_INFERRED
    if status == VERIFICATION_MACHINE_VERIFIED and provenance != PROVENANCE_AUTOMATED_OFFICIAL:
        errors.append("machine_verified requires automated_official provenance.")
        return VERIFICATION_UNVERIFIED
    if provenance == PROVENANCE_AUTOMATED_OFFICIAL and status == VERIFICATION_VERIFIED:
        errors.append("Automated verification cannot claim human verified status.")
    return status


def _review_reasons(metadata: dict[str, Any]) -> list[str]:
    """Every reason this record is not fit for the shared corpus. Returned (and
    stored) rather than reduced to a bare status so the admin review screen can
    say *what* to fix instead of only "needs review"."""
    reasons: list[str] = []
    if metadata["issuing_level"] == ISSUING_LEVEL_UNKNOWN:
        reasons.append("Issuing level is unknown.")
    if metadata["applicability"] == APPLICABILITY_UNKNOWN:
        reasons.append("Applicability is unknown.")
    if metadata["verification_status"] not in {VERIFICATION_VERIFIED, VERIFICATION_MACHINE_VERIFIED}:
        reasons.append(f"Metadata is {metadata['verification_status']}, not human-verified.")
    if metadata["verification_status"] == VERIFICATION_MACHINE_VERIFIED:
        evidence = metadata.get("machine_verification") or {}
        required = {
            "official_url", "official_sha256", "local_sha256", "content_sha256",
            "policy_version", "verified_at", "identity_checks", "applicability_checks",
            "commencement_checks", "commencement_url", "commencement_sha256", "exact_byte_match",
        }
        if not isinstance(evidence, dict) or required - set(evidence):
            reasons.append("Machine-verification evidence is incomplete.")
        elif evidence.get("exact_byte_match") is not True or not all(
            evidence.get(name) for name in ("identity_checks", "applicability_checks", "commencement_checks")
        ):
            reasons.append("Machine-verification evidence checks did not all pass.")
    if not metadata.get("source_url"):
        reasons.append("No official source URL recorded.")
    return reasons


def normalize_jurisdiction(
    raw: dict[str, Any] | None,
    *,
    provenance: str = PROVENANCE_MANUAL,
    now: datetime | None = None,
) -> NormalizedJurisdiction:
    """Validates and canonicalizes one document's jurisdiction metadata.

    Raises `JurisdictionMetadataError` for anything structurally wrong (an
    unknown State code, a reversed date range, a `specific_states`
    applicability with no States). Anything merely *missing* is not an error:
    it becomes `unknown` and routes the record to `REVIEW_NEEDS_REVIEW`, which
    is what keeps an under-described document out of shared retrieval instead
    of out of the Knowledge Base entirely.
    """
    raw = dict(raw or {})
    errors: list[str] = []
    now = now or datetime.now(UTC)

    resolved_provenance = _enum(
        raw.get("metadata_provenance", provenance), PROVENANCES, "metadata_provenance", provenance, errors
    )
    issuing_level = _enum(raw.get("issuing_level"), ISSUING_LEVELS, "issuing_level", ISSUING_LEVEL_UNKNOWN, errors)
    # Rule 1 -- there is deliberately no `if issuing_level == central:
    # applicability = all_india` branch here, and there must never be one.
    applicability = _enum(raw.get("applicability"), APPLICABILITIES, "applicability", APPLICABILITY_UNKNOWN, errors)

    state_codes: list[str] = []
    for item in _string_list(raw.get("applicable_state_codes")):
        code = normalize_state_code(item)
        if code is None:
            errors.append(f"Unknown State/UT code or name: {item!r}.")
        elif code not in state_codes:
            state_codes.append(code)
    state_codes.sort()

    if applicability == APPLICABILITY_SPECIFIC_STATES and not state_codes:
        errors.append("applicability 'specific_states' requires at least one applicable State/UT code.")
    if applicability == APPLICABILITY_ALL_INDIA and state_codes:
        errors.append("applicability 'all_india' cannot also list specific State/UT codes.")

    effective_from = _parse_date(raw.get("effective_from"), "effective_from", errors)
    effective_to = _parse_date(raw.get("effective_to"), "effective_to", errors)
    if effective_from and effective_to and effective_to < effective_from:
        errors.append(f"effective_to ({effective_to}) is before effective_from ({effective_from}).")

    document_key = str(raw.get("document_key") or "").strip().lower()
    if document_key and not _DOCUMENT_KEY_RE.match(document_key):
        errors.append(
            "document_key must be 3-121 chars of lowercase letters, digits, '.', '_' or '-' "
            f"(e.g. 'mh-rent-control-1999'); got {raw.get('document_key')!r}."
        )

    source_type = _enum(
        raw.get("jurisdiction_source_type") or raw.get("source_type"),
        SOURCE_TYPE_VALUES, "jurisdiction_source_type", "unknown", errors,
    )
    verification_status = _resolve_verification(raw.get("verification_status"), resolved_provenance, errors)

    # `section_overrides` collects its own errors; merge them into the same
    # report so one round-trip tells the admin everything that is wrong.
    section_overrides: dict[str, dict[str, Any]] = {}
    try:
        section_overrides = _normalize_section_overrides(raw.get("section_overrides"))
    except JurisdictionMetadataError as exc:
        errors.extend(exc.errors)

    if errors:
        raise JurisdictionMetadataError(errors)

    metadata: dict[str, Any] = {
        "document_key": document_key or None,
        "issuing_level": issuing_level,
        "applicability": applicability,
        "applicable_state_codes": state_codes,
        "applicable_localities": _string_list(raw.get("applicable_localities")),
        "jurisdiction_source_type": source_type,
        "source_url": (str(raw.get("source_url")).strip() or None) if raw.get("source_url") else None,
        "source_page_reference": (
            (str(raw.get("source_page_reference")).strip() or None) if raw.get("source_page_reference") else None
        ),
        "subsection": (str(raw.get("subsection")).strip() or None) if raw.get("subsection") else None,
        "parent_reference": (
            (str(raw.get("parent_reference")).strip() or None) if raw.get("parent_reference") else None
        ),
        "version_label": (str(raw.get("version_label")).strip() or None) if raw.get("version_label") else None,
        "effective_from": effective_from,
        "effective_to": effective_to,
        "amends": _string_list(raw.get("amends")),
        "amended_by": _string_list(raw.get("amended_by")),
        "supersedes": _string_list(raw.get("supersedes")),
        "superseded_by": _string_list(raw.get("superseded_by")),
        "verification_status": verification_status,
        "metadata_provenance": resolved_provenance,
        "jurisdiction_schema_version": JURISDICTION_SCHEMA_VERSION,
        "machine_verification": raw.get("machine_verification") if isinstance(raw.get("machine_verification"), dict) else None,
    }
    # `last_verified_at` is meaningful only alongside a `verified` status --
    # stamping it on an unverified row would make an untouched document look
    # freshly checked on the review screen.
    if verification_status in {VERIFICATION_VERIFIED, VERIFICATION_MACHINE_VERIFIED}:
        supplied = raw.get("last_verified_at")
        metadata["last_verified_at"] = (
            supplied.strip() if isinstance(supplied, str) and supplied.strip() else now.isoformat()
        )
        metadata["verified_by"] = (str(raw.get("verified_by")).strip() or None) if raw.get("verified_by") else None
    else:
        metadata["last_verified_at"] = None
        metadata["verified_by"] = None

    reasons = _review_reasons(metadata)
    metadata["review_reasons"] = reasons
    metadata["review_status"] = REVIEW_NEEDS_REVIEW if reasons else REVIEW_APPROVED

    return NormalizedJurisdiction(metadata=metadata, section_overrides=section_overrides)


# Only the fields a single section can legitimately differ from its parent Act
# on. A section cannot, for instance, carry its own `document_key` (it is part
# of the same document) or its own `verification_status` (verification is an
# act of a human on the whole record).
SECTION_OVERRIDE_FIELDS: tuple[str, ...] = (
    "applicability", "applicable_state_codes", "applicable_localities",
    "effective_from", "effective_to", "in_force", "source_page_reference", "parent_reference", "subsection",
)


def _normalize_section_overrides(raw: Any) -> dict[str, dict[str, Any]]:
    """Per-section deviations, e.g. an Act in force nationwide since 2018 whose
    Section 12 was only brought into force in Maharashtra in 2021. Real, and
    common enough in Indian commencement notifications (a single Act is
    routinely commenced section by section, State by State) that a
    document-level-only model would state the wrong date for exactly the
    provision a user asked about."""
    if raw in (None, ""):
        return {}
    if not isinstance(raw, dict):
        raise JurisdictionMetadataError(["section_overrides must be an object keyed by section number."])
    normalized: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for section, override in raw.items():
        section_key = str(section).strip().upper()
        if not isinstance(override, dict):
            errors.append(f"section_overrides[{section_key}] must be an object.")
            continue
        resolved: dict[str, Any] = {}
        for key, value in override.items():
            if key not in SECTION_OVERRIDE_FIELDS:
                errors.append(
                    f"section_overrides[{section_key}] cannot override {key!r} "
                    f"(allowed: {', '.join(SECTION_OVERRIDE_FIELDS)})."
                )
                continue
            if key == "in_force":
                if not isinstance(value, bool):
                    errors.append(f"section_overrides[{section_key}].in_force must be true or false.")
                else:
                    resolved[key] = value
            elif key in ("effective_from", "effective_to"):
                resolved[key] = _parse_date(value, f"section_overrides[{section_key}].{key}", errors)
            elif key == "applicability":
                resolved[key] = _enum(
                    value, APPLICABILITIES, f"section_overrides[{section_key}].applicability",
                    APPLICABILITY_UNKNOWN, errors,
                )
            elif key == "applicable_state_codes":
                codes: list[str] = []
                for item in _string_list(value):
                    code = normalize_state_code(item)
                    if code is None:
                        errors.append(f"section_overrides[{section_key}]: unknown State/UT code {item!r}.")
                    elif code not in codes:
                        codes.append(code)
                resolved[key] = sorted(codes)
            elif key == "applicable_localities":
                resolved[key] = _string_list(value)
            else:
                resolved[key] = (str(value).strip() or None) if value else None
        if (
            resolved.get("effective_from") and resolved.get("effective_to")
            and resolved["effective_to"] < resolved["effective_from"]
        ):
            errors.append(f"section_overrides[{section_key}]: effective_to is before effective_from.")
        if resolved:
            normalized[section_key] = resolved
    if errors:
        raise JurisdictionMetadataError(errors)
    return normalized


def document_metadata_fields(normalized: NormalizedJurisdiction) -> dict[str, Any]:
    """The flat dict merged into `document.metadata` at index time. Section
    overrides ride along under their own key so they survive into the document
    record and can be re-applied without re-parsing the file."""
    fields = dict(normalized.metadata)
    if normalized.section_overrides:
        fields["section_overrides"] = normalized.section_overrides
    return fields


def apply_to_chunk(chunk_metadata: dict[str, Any], document_metadata: dict[str, Any]) -> dict[str, Any]:
    """Copies the document's jurisdiction fields onto one chunk, then applies
    that chunk's own section override if the document declared one.

    Called per chunk by `IndexingPipeline` AFTER `MetadataExtractor.extract`
    has determined the chunk's own `section_number`, because which override
    applies is a function of that number. Mutates and returns `chunk_metadata`
    (the same convention `extract` uses).
    """
    for key in JURISDICTION_FIELDS:
        if key in document_metadata:
            chunk_metadata[key] = document_metadata[key]

    overrides = document_metadata.get("section_overrides") or {}
    if not overrides:
        return chunk_metadata
    section = chunk_metadata.get("section_number") or chunk_metadata.get("article_number")
    override = overrides.get(str(section).strip().upper()) if section else None
    if not override:
        return chunk_metadata
    chunk_metadata.update(override)
    # Recorded so a reviewer (and any later backfill) can tell a chunk whose
    # dates/applicability came from a section-level override apart from one
    # that simply inherited the document's values.
    chunk_metadata["section_override_applied"] = True
    return chunk_metadata


async def propagate_jurisdiction_metadata(
    documents_collection: Any,
    chunks_collection: Any,
    *,
    document_id: str,
    source_document: str | None,
    fields: dict[str, Any],
    section_overrides: dict[str, dict[str, Any]] | None = None,
) -> int:
    """Writes `fields` (from `document_metadata_fields`) onto one KB
    document's `metadata` and onto every chunk of that same
    `source_document` -- directly, via `update_one`/`update_many`/per-chunk
    `update_one`, NEVER through `IndexingPipeline.index_file`.

    Shared by the one-shot admin jurisdiction-correction endpoint
    (`KnowledgeBaseIngestionService.update_jurisdiction_metadata`, gap 2) and
    `scripts/backfill_kb_jurisdiction.py`'s bulk migration, so both write
    metadata via the exact same rule, not two copies that could drift apart:
    a metadata-only correction is not new content, so it must never go back
    through `DocumentQualityChecker`'s duplicate-hash gate (which would
    reject it outright as a "duplicate" of the very document it is
    correcting) or trigger re-embedding -- text, chunk IDs, embeddings and
    `document_versions` history are never touched here.

    Returns the number of chunks updated.
    """
    await documents_collection.update_one(
        {"_id": document_id}, {"$set": {f"metadata.{key}": value for key, value in fields.items()}}
    )
    if not source_document:
        return 0
    if section_overrides:
        # A section-level override needs each chunk's OWN section_number to
        # resolve correctly (see `apply_to_chunk`) -- read and rewrite those
        # chunks individually rather than one blanket `update_many`.
        updated = 0
        async for chunk in chunks_collection.find({"metadata.source_document": source_document}):
            chunk_metadata = apply_to_chunk(dict(chunk.get("metadata") or {}), fields)
            await chunks_collection.update_one({"_id": chunk["_id"]}, {"$set": {"metadata": chunk_metadata}})
            updated += 1
        return updated
    result = await chunks_collection.update_many(
        {"metadata.source_document": source_document},
        {"$set": {f"metadata.{key}": value for key, value in fields.items()}},
    )
    return int(result.modified_count)


def shared_retrieval_filters() -> dict[str, Any]:
    """The review-status constraint the SHARED (unowned) branch of a
    retrieval query carries: `review_status` must equal `REVIEW_APPROVED`
    exactly.

    Deliberately strict, not `[None, REVIEW_APPROVED]` -- an earlier version
    of this function let a chunk with NO `review_status` key at all (every
    chunk indexed before this phase, and any KB document the jurisdiction
    backfill has not yet reached) pass through unreviewed, on the reasoning
    that this kept the existing corpus answering exactly as before. That
    reasoning was wrong: it silently re-opened the exact hole Phase 1 exists
    to close (an unreviewed document's jurisdiction being presented as
    settled) for every document until an admin remembered to run the
    backfill AND re-verify it. Explicit `REVIEW_APPROVED` is now required for
    ANY document to reach shared retrieval, migrated or not -- there is no
    backward-compatible default for "nobody has looked at this yet".

    This function must ONLY be applied to a query's SHARED/unowned branch,
    never as a blanket filter ANDed across a whole ownership `$or` --
    `LegalRetriever.retrieve` applies it exactly this way (only when the
    caller's filters carry no `$or` at all, i.e. there is no ownership
    concept in play), and `ChatService._prepare_rag_context` builds it
    directly into its shared branch. A private per-user chunk never carries
    `review_status` and is reached ONLY through its owner's ownership branch,
    which never requires this field -- see those two call sites' own
    comments for exactly how the split is kept structurally safe.
    """
    return {"review_status": REVIEW_APPROVED}


# ---------------------------------------------------------------------------
# Jurisdiction Routing (Phase 2): applicability + version eligibility at
# RETRIEVAL time, given a resolved `app.rag.matter_context.MatterContext`.
#
# Two enforcement layers, deliberately BOTH present:
#   1. `jurisdiction_or_branches` builds a native Mongo `$or` that
#      `ChatService._prepare_rag_context` folds into its shared/unowned
#      branch (same place `shared_retrieval_filters` lives) -- this is what
#      keeps an UNRELATED State's law out of the candidate pool in the first
#      place, for the two DB-backed retrieval legs.
#   2. `filter_by_matter_context` is a PURE post-filter, applied by
#      `LegalRetriever.retrieve` unconditionally to whatever `results` list
#      it has assembled by the time it returns -- regardless of which
#      internal path produced a candidate (vector+BM25 fusion, the named-
#      section/article exact lookups, the bare-section-number floor
#      fallback). This is what satisfies "vector, BM25, exact-section
#      lookup, reranking inputs aur fallback paths mein same eligibility
#      enforce karo": a single choke point right before `retrieve()` returns
#      means no current or future internal path can accidentally bypass it,
#      and a zero-result outcome is never "fixed" by silently dropping the
#      filter (this function never falls back to unfiltered on empty input).
# ---------------------------------------------------------------------------


def jurisdiction_or_branches(state_codes: list[str] | None, locality: str | None) -> list[dict[str, Any]]:
    """The applicability `$or` for a resolved matter -- always includes
    all-India provisions, plus one branch per resolved State (multi-State
    matters get one branch each, never forced into a single State -- see
    `MatterContext`), plus a locality branch only when a locality was
    actually resolved. `applicability=unknown` matches NONE of these
    branches on purpose (objective: "Unknown applicability ko unrestricted
    fallback mat banao") -- and per `_review_reasons`, an `unknown`-
    applicability chunk can never be `review_status=approved` anyway, so this
    is a second, independent guarantee, not the only one.

    Returns `[]` only when NEITHER a State nor a locality is resolved --
    callers must treat that as "do not add a jurisdiction constraint at all"
    (a state-INsensitive question needs no geographic filter), never as
    "match nothing". A locality resolved WITHOUT a State (a user answered
    "which city?" without also naming a State -- see
    `matter_context.resolve_matter_context`'s `awaiting_locality_clarification`)
    still produces a real constraint: all-India provisions, plus any
    State-specific provision whose OWN `applicable_localities` names that
    locality, regardless of which State it's tagged under.
    """
    if not state_codes and not locality:
        return []
    branches: list[dict[str, Any]] = [{"applicability": APPLICABILITY_ALL_INDIA}]
    for code in state_codes or []:
        branches.append({"applicability": APPLICABILITY_SPECIFIC_STATES, "applicable_state_codes": code})
    if locality:
        branches.append({"applicability": APPLICABILITY_SPECIFIC_STATES, "applicable_localities": locality})
    return branches


def chunk_matches_jurisdiction(metadata: dict[str, Any], state_codes: list[str] | None, locality: str | None) -> bool:
    """Pure-Python mirror of `jurisdiction_or_branches`, evaluated directly
    against one chunk's metadata -- the post-filter layer (2) described
    above. `applicable_state_codes`/`applicable_localities` are stored as
    LISTS on the chunk (a provision can apply to more than one State), so
    membership, not equality, is the right comparison here.
    """
    if not state_codes and not locality:
        return True
    applicability = metadata.get("applicability")
    if applicability == APPLICABILITY_ALL_INDIA:
        return True
    if applicability != APPLICABILITY_SPECIFIC_STATES:
        return False  # covers "unknown" and any unexpected value -- never an unrestricted fallback
    chunk_states = metadata.get("applicable_state_codes") or []
    if state_codes and any(code in chunk_states for code in state_codes):
        return True
    if locality:
        chunk_localities = metadata.get("applicable_localities") or []
        if locality in chunk_localities:
            return True
    return False


def chunk_temporally_eligible(metadata: dict[str, Any], as_of_date: str | None) -> bool:
    """True when this chunk's `effective_from`/`effective_to` do not
    AFFIRMATIVELY rule it out as of `as_of_date` (ISO date; `None` skips
    temporal filtering entirely -- no date resolved means no temporal claim
    is being made either way).

    Deliberately asymmetric: a chunk with NEITHER date set is always
    eligible (objective: "Dates/relationships insufficient hon toh...guess
    mat karo" -- absence of data is not evidence of ineligibility, and
    almost the entire corpus has no dates recorded yet; treating absence as
    exclusion would empty shared retrieval, and treating it as automatic
    "this is definitely current" would be the fabrication the phase forbids
    just as much). Only an EXPLICIT `effective_from` in the future, or an
    EXPLICIT `effective_to` before `as_of_date`, excludes a chunk -- both are
    affirmative facts already on record, not guesses.
    """
    if metadata.get("in_force") is False:
        return False
    if not as_of_date:
        return True
    effective_from = metadata.get("effective_from")
    if effective_from and effective_from > as_of_date:
        return False  # not yet in force as of this date (objective: no future-effective provision as "current")
    effective_to = metadata.get("effective_to")
    return not (effective_to and effective_to < as_of_date)  # excluded only if already superseded/expired


# The reserved filter key `MongoVectorStore._mongo_filter`/`_atlas_filter`
# and `BM25Index._matches_filters` each special-case, so temporal
# eligibility is enforced at CANDIDATE SELECTION (inside the Mongo/Atlas
# query itself, or before BM25's own top_k truncation) -- NOT only by the
# `filter_by_matter_context` post-filter above. Without this, an
# out-of-range chunk could occupy a top_k candidate slot ahead of an
# eligible one sitting deeper in the corpus, so a historical query could
# come back EMPTY even though a genuinely eligible version exists -- the
# post-filter alone cannot recover a candidate that was never fetched.
TEMPORAL_FILTER_KEY = "_temporal_as_of"


def temporal_filter(as_of_date: str | None) -> dict[str, Any]:
    """The filter-dict fragment for `as_of_date` -- merge into a query's
    filters (e.g. `ChatService`'s shared branch) alongside
    `jurisdiction_or_branches`'s output. `{}` when `as_of_date` is `None`."""
    return {TEMPORAL_FILTER_KEY: as_of_date} if as_of_date else {}


def mongo_temporal_and_clauses(as_of_date: str) -> list[dict[str, Any]]:
    """Native MQL clauses expressing exactly `chunk_temporally_eligible`'s
    logic -- absent OR null OR in-range, for both `effective_from` and
    `effective_to`. Valid both for a real `_local_cosine_leg`
    `collection.find()` query AND for `$vectorSearch`'s `filter` option
    (confirmed by this module's existing `_atlas_filter` output already using
    plain MQL operators -- `$eq`/`$in`/`$and`/`$or` -- for that same option,
    not Atlas Search's separate `compound`/`range` text-search DSL).
    """
    return [
        {"metadata.in_force": {"$ne": False}},
        {
            "$or": [
                {"metadata.effective_from": {"$exists": False}},
                {"metadata.effective_from": None},
                {"metadata.effective_from": {"$lte": as_of_date}},
            ]
        },
        {
            "$or": [
                {"metadata.effective_to": {"$exists": False}},
                {"metadata.effective_to": None},
                {"metadata.effective_to": {"$gte": as_of_date}},
            ]
        },
    ]


def filter_by_matter_context(
    results: list[Any], state_codes: list[str] | None, locality: str | None, as_of_date: str | None,
) -> list[Any]:
    """Applies both eligibility checks above to a list of `RetrievedChunk`-
    shaped objects (anything with a `.metadata` dict attribute), in one pass.
    Returns a NEW list -- an empty result is returned as-is, never backfilled
    with unfiltered candidates (that silent-fallback is exactly what the
    objective's "zero results par restrictions silently remove mat karo"
    forbids down at this layer; `ChatService`'s existing strict-RAG guardrail
    already turns an empty `ranked` list into an honest "no verified context"
    answer, so an empty return here is the CORRECT outcome, not a bug to work
    around).
    """
    if not state_codes and not locality and not as_of_date:
        return results
    return [
        chunk
        for chunk in results
        if chunk_matches_jurisdiction(chunk.metadata, state_codes, locality)
        and chunk_temporally_eligible(chunk.metadata, as_of_date)
    ]


def detect_jurisdiction_ambiguity(results: list[Any]) -> dict[str, Any] | None:
    """Data-driven clarification trigger, checked AFTER retrieval when no
    State was resolved from any source (`filter_by_matter_context` was a
    no-op, so `results` is whatever similarity ranking surfaced,
    unconstrained by State).

    Broader than any fixed category+keyword allowlist
    (`app.rag.matter_context.STATE_SENSITIVE_LEGAL_CATEGORIES`): it looks at
    what the retrieved candidates THEMSELVES claim. Two distinct triggers,
    both real ways an unconfirmed jurisdiction can mislead an answer:

    1. CONFLICT -- two or more different States (or localities) each have
       their own specific-applicability candidate among the results. Picking
       either arbitrarily would silently prefer one State's law over another
       the user never named.
    2. UNCONFIRMED SINGLE STATE -- exactly one State's (or locality's)
       specific-applicability candidate was found, and NO all-India provision
       is also present to fall back to. There is no "conflict" to report
       here, but presenting that one State's rule as the answer still
       silently assumes the user is IN that State -- an absence of conflict
       is not proof the jurisdiction is settled (a State this KB happens not
       to have indexed content for could differ just as much as one that
       visibly conflicts). This is the more common real case in a KB that
       has not yet indexed every State's version of a given law.

    Chunks with no `applicability` at all (untagged legacy content, case law,
    private per-user documents) are ignored entirely for this signal -- it
    only activates once at least one candidate is genuinely
    jurisdiction-tagged; an untagged corpus produces no false positives here
    (that gap is instead the fixed category+keyword pre-check's job).

    Locality is reported in preference to State when both are present (more
    specific, more decision-relevant to ask about first). Returns `None`
    only when either an all-India candidate covers the question or no
    jurisdiction-tagged candidate was found at all.
    """
    # Deliberately local: `app.rag.citation` importing this module (for
    # `RetrievedChunk`/citation building) is the natural direction, so this
    # import runs the other way to avoid a cycle rather than because the
    # logic belongs to jurisdiction specifically.
    from app.rag.citation import _looks_like_bns_family

    has_all_india = False
    states: set[str] = set()
    localities: set[str] = set()
    tagged_candidate_found = False
    for chunk in results:
        metadata = getattr(chunk, "metadata", None) or {}
        # BNS/BNSS/BSA are confirmed, uniformly all-India central codes (they
        # replaced IPC/CrPC/the Evidence Act, which applied identically
        # everywhere -- unlike, say, a "model law" needing State adoption,
        # nothing here depends on issuing_level implying applicability in
        # general, which this module's own docstring correctly warns
        # against). Confirmed live: a large share of this corpus's BNS/BNSS
        # content carries `applicability: "unknown"` (an ingestion gap, not
        # a real ambiguity about these specific Acts) or no applicability at
        # all -- left as `unknown`/`None`, such a chunk contributes nothing
        # to `has_all_india` below, so a single, genuinely unrelated
        # State-specific chunk retrieved alongside it (embedding/BM25 noise,
        # a separate known issue) was enough to wrongly ask "Which State?"
        # for an ordinary central-code lookup (e.g. "which section talks
        # about culpable homicide"). Checked before the `applicability is
        # None` skip below so it applies regardless of whether the field is
        # `unknown`, missing, or (incorrectly) `specific_states`.
        if _looks_like_bns_family(str(metadata.get("act_name") or "")):
            has_all_india = True
            continue
        applicability = metadata.get("applicability")
        if applicability is None:
            continue
        tagged_candidate_found = True
        if applicability == APPLICABILITY_ALL_INDIA:
            has_all_india = True
        elif applicability == APPLICABILITY_SPECIFIC_STATES:
            states.update(metadata.get("applicable_state_codes") or [])
            localities.update(metadata.get("applicable_localities") or [])
    if not tagged_candidate_found and not has_all_india:
        return None
    if len(localities) > 1:
        return {"ambiguity": "locality", "candidates": sorted(localities)}
    if len(states) > 1:
        return {"ambiguity": "state", "candidates": sorted(states)}
    if not has_all_india and localities:
        return {"ambiguity": "locality", "candidates": sorted(localities)}
    if not has_all_india and states:
        return {"ambiguity": "state", "candidates": sorted(states)}
    return None


def detect_version_ambiguity(results: list[Any]) -> list[tuple[str | None, str | None]] | None:
    """For a HISTORICAL query (`as_of_date` resolved), a single `as_of_date`
    does not by itself guarantee only one version of a provision survives
    `filter_by_matter_context` -- two versions with no recorded `effective_to`
    (or otherwise overlapping ranges) can both look "in force" as of the same
    date, and neither this function nor anything upstream has any transition/
    savings-clause metadata to break the tie (objective: "Incident date alone
    se har legal issue ka governing version conclusively decide mat karo").

    Groups the given (already temporally-filtered) results by
    `(source_document, section_number)` and returns the distinct
    `(effective_from, effective_to)` pairs for any group with more than one --
    i.e. genuine, KB-internal version ambiguity for that one provision, not a
    question the USER can resolve by naming a State (see
    `detect_jurisdiction_ambiguity` for that, separate, case). `None` when
    every provision has at most one surviving version.
    """
    groups: dict[tuple[str | None, str | None], set[tuple[str | None, str | None]]] = {}
    for chunk in results:
        metadata = getattr(chunk, "metadata", None) or {}
        section = metadata.get("section_number") or metadata.get("article_number")
        if not section:
            continue
        key = (metadata.get("source_document"), section)
        groups.setdefault(key, set()).add((metadata.get("effective_from"), metadata.get("effective_to")))
    for versions in groups.values():
        if len(versions) > 1:
            return sorted(versions, key=lambda pair: (pair[0] or "", pair[1] or ""))
    return None
