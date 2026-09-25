"""KB Cleanup Phase 2: Priority Laws Review and Controlled Publication.

Builds the admin review package for the 7 priority central laws (evidence
gathered from official sources, resolved against Phase 1's inventory and
MongoDB), stages the one in-scope genuinely-new file (RTI Act 2005) via a
copy-based, needs_review-only ledger entry, and -- ONLY when explicitly
invoked with an approved manifest -- propagates verification_status/
review_status metadata to already-indexed documents and their chunks.

Nothing here indexes, re-embeds, deletes, moves or renames any KB file.
`publish_approved` (never called by `main()`) is the only function that
writes review_status=approved, and only for document_keys the caller passes
explicitly, each requiring `verified_by`.

Usage:
    python scripts/kb_audit_phase2.py            # review package only (read-only + one safe copy)
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cache.redis_client import redis_client
from app.cache.response_cache import CACHE_KEY_PREFIX, response_cache
from app.core.config import settings
from app.database.mongodb import mongodb
from app.models.collections import (
    DOCUMENT_VERSIONS,
    EMBEDDINGS_METADATA,
    UPLOADED_DOCUMENTS,
)
from app.rag.bm25_index import bm25_index
from app.rag.kb_jurisdiction import (
    PROVENANCE_INFERRED,
    document_metadata_fields,
    normalize_jurisdiction,
    propagate_jurisdiction_metadata,
)
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService

AUDIT_DIR = Path(__file__).resolve().parent.parent / "storage" / "kb_audit"
KB_DIR = settings.knowledge_base_dir

# ---------------------------------------------------------------------------
# Priority law identities + official evidence, gathered from official-domain
# sources this session (India Code / MHA / issuing ministry / PIB). Every
# `*_confirmed_from_local_file` value was read directly out of the file's own
# text (title page / Section 1), not inferred. Every `official_*` value is
# from a WebSearch citing an official .gov.in domain; direct WebFetch to
# those domains returned 403/connection-refused in this environment, so
# `fetch_verified=False` throughout -- a human must open the cited URL
# directly before any of this becomes `verification_status=verified`.
# ---------------------------------------------------------------------------
PRIORITY_LAWS: list[dict[str, Any]] = [
    {
        "priority": 1,
        "law": "Bharatiya Nyaya Sanhita, 2023 (BNS)",
        "document_key": "bharatiya_nyaya_sanhita_2023",
        "canonical_filename": "The_Bhara_Tiy_A_Ny_A_Y_A_Sanhita_2023_5.pdf",
        "duplicate_filenames": ["Bharatiya_Nyaya_Sanhita_2023_Complete_Act.pdf"],
        "title_confirmed_from_local_file": "THE BHARA TIY A NY A Y A SANHITA, 2023 -- NO. 45 OF 2023 [25th December, 2023.]",
        "act_number": "45 of 2023",
        "enactment_date_local_file": "2023-12-25",
        "official_commencement_claim": (
            "1 July 2024 (except section 106(2)) per Notification No. S.O. 850(E) dated 23.02.2024 -- "
            "cited via WebSearch against mha.gov.in (250883_english_01042024.pdf); direct fetch blocked."
        ),
        "official_source_urls": [
            "https://www.mha.gov.in/sites/default/files/250883_english_01042024.pdf",
            "https://www.indiacode.nic.in/handle/123456789/20062?locale=en",
        ],
        "applicability_evidence": (
            "Local file Section 1(3)-(5): liability 'within India' and extraterritorial application to "
            "citizens of India -- application clause, not an explicit 'extends to whole of India' statement."
        ),
        "applicability_claim": "all_india",
        "classification": "needs_effective_date_confirmation",
        "notes": (
            "Title/Act number/enactment date confirmed directly from the local file's own gazette masthead. "
            "Commencement date and exact applicability wording are only WebSearch-mediated citations of an "
            "official domain, not independently fetched -- needs a human to open the MHA/India Code URL "
            "directly before verification_status=verified."
        ),
    },
    {
        "priority": 2,
        "law": "Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)",
        "document_key": "bharatiya_nagarik_suraksha_sanhita_2023",
        "canonical_filename": "Bharatiya_Nagarik_Suraksha_Sanhita_2023_Complete_Act.pdf",
        "duplicate_filenames": [],
        "title_confirmed_from_local_file": (
            "The Bharatiya Nagarik Suraksha Sanhita, 2023 (ACT NO. 46 OF 2023) [As on the 6th October, 2025]"
        ),
        "act_number": "46 of 2023",
        "enactment_date_local_file": None,
        "official_commencement_claim": (
            "1 July 2024 per MHA notification dated 24.02.2024 -- cited via WebSearch against "
            "mha.gov.in/sites/default/files/BharatiyaNagarikSurakshaSanhita_24022024.pdf; direct fetch blocked."
        ),
        "official_source_urls": [
            "https://www.mha.gov.in/sites/default/files/BharatiyaNagarikSurakshaSanhita_24022024.pdf",
        ],
        "applicability_evidence": (
            "Table of contents shows 'Short title, extent and commencement' as the Section 1 heading, but the "
            "clause's own text was not reached in this pass -- exact extent wording unconfirmed."
        ),
        "applicability_claim": "unknown",
        "classification": "needs_effective_date_confirmation",
        "notes": (
            "IMPORTANT: this local file's own title page is stamped '[As on the 6th October, 2025]' -- it is a "
            "CONSOLIDATED, brought-up-to-date India Code-style edition, NOT a copy of the original 25-Dec-2023 "
            "Gazette-as-enacted text (unlike the BNS/BSA local files, which show the as-enacted masthead). "
            "Per this phase's own instruction, an updated consolidated text must not be assumed identical to the "
            "original Gazette version -- a human must confirm which version is intended for the shared KB. Also "
            "has 3 historical document_versions rows (v1 deleted, v2 superseded, v3 active), all with the IDENTICAL "
            "document_hash -- benign repeated reindexing of unchanged content, not a content concern."
        ),
    },
    {
        "priority": 3,
        "law": "Bharatiya Sakshya Adhiniyam, 2023 (BSA)",
        "document_key": "bharatiya_sakshya_adhiniyam_2023",
        "canonical_filename": "The_Bhara_Tiy_A_Sakshy_A_Adhiniy_Am_2023_6.pdf",
        "duplicate_filenames": ["The_Bharatiya_Sakshya_Adhiniyam_2023_2.pdf"],
        "title_confirmed_from_local_file": "THE BHARA TIY A SAKSHY A ADHINIY AM, 2023 -- NO. 47 OF 2023 [25th December, 2023.]",
        "act_number": "47 of 2023",
        "enactment_date_local_file": "2023-12-25",
        "official_commencement_claim": (
            "1 July 2024 per MHA notification (same 23 Feb 2024 tranche as BNS/BNSS) -- WebSearch-cited, not "
            "directly fetched."
        ),
        "official_source_urls": [
            "https://www.indiacode.nic.in/handle/123456789/20063?locale=en",
        ],
        "applicability_evidence": (
            "Local file Section 1(2): 'It applies to all judicial proceedings in or before any Court...' -- "
            "this is SUBJECT-MATTER/procedural applicability, not a territorial-extent statement. No "
            "'extends to whole of India' text found in the pages read."
        ),
        "applicability_claim": "unknown",
        "classification": "needs_effective_date_confirmation",
        "notes": (
            "Title/Act number/enactment date confirmed directly from the local file's own gazette masthead. "
            "Territorial applicability NOT textually confirmed (only procedural-scope clause found) -- do not "
            "assume all_india without checking a later section or the official source directly. Commencement "
            "date is WebSearch-cited only."
        ),
    },
    {
        "priority": 4,
        "law": "Consumer Protection Act, 2019",
        "document_key": "consumer_protection_act_2019",
        "canonical_filename": "The_Consumer_Protection_Act_2019_5.pdf",
        "duplicate_filenames": [],
        "title_confirmed_from_local_file": "THE CONSUMER PROTECTION ACT, 2019",
        "act_number": "35 of 2019",
        "enactment_date_local_file": None,
        "official_commencement_claim": (
            "Staggered commencement: several provisions from 24 July 2020 vide Notification No. S.O. 2421(E) "
            "dated 23 July 2020; other provisions (main consumer-complaint machinery) reported effective "
            "20 July 2020 -- WebSearch-cited against consumeraffairs.nic.in (Dept. of Consumer Affairs, the "
            "issuing ministry); direct fetch returned a connection error in this environment."
        ),
        "official_source_urls": [
            "https://consumeraffairs.nic.in/theconsumerprotection/notification-regarding-consumer-protection-act-2019-coming-force",
            "https://www.indiacode.nic.in/handle/123456789/15256?view_type=browse&sam_handle=123456789/1362",
        ],
        "applicability_evidence": (
            "Local file Section 1(2), verbatim: 'It extends to the whole of India except the State of Jammu "
            "and Kashmir.'"
        ),
        "applicability_claim": "all_india_with_stated_exception",
        "classification": "needs_effective_date_confirmation",
        "notes": (
            "Commencement is NOT a single date -- multiple provisions were brought into force on different "
            "dates by different notifications; a single effective_from would be a legal oversimplification and "
            "must not be set without a human choosing which milestone the KB should record. The local file's own "
            "J&K exception clause pre-dates the 2019 J&K Reorganisation and its adaptation orders -- whether that "
            "carve-out is still operative is a transition/savings-provision question for a human reviewer, not "
            "resolved here."
        ),
    },
    {
        "priority": 5,
        "law": "Information Technology Act, 2000",
        "document_key": "information_technology_act_2000",
        "canonical_filename": None,
        "duplicate_filenames": ["The_Information_Technology_Act_2000_4.pdf"],
        "title_confirmed_from_local_file": "THE INFORMATION TECHNOLOGY ACT, 2000",
        "act_number": "21 of 2000",
        "enactment_date_local_file": None,
        "official_commencement_claim": (
            "17 October 2000 vide Notification No. G.S.R. 788(E) dated 17 October 2000 -- WebSearch-cited "
            "against meity.gov.in/indiacode.nic.in; not directly fetched."
        ),
        "official_source_urls": [
            "https://www.meity.gov.in/static/uploads/2024/03/IT-Act-Rules_2000_0.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/13116/1/it_act_2000_updated.pdf",
        ],
        "applicability_evidence": (
            "Local file Section 1(2), verbatim: 'It shall extend to the whole of India and, save as otherwise "
            "provided in this Act, it applies also to any offence or contravention thereunder committed "
            "outside India by any person.'"
        ),
        "applicability_claim": "all_india",
        "classification": "not_present",
        "notes": (
            "The local file's content is confirmed genuine (title, Act number, extent clause all read directly "
            "from the file) and READABLE, but it has NO live indexed content anywhere in the corpus right now -- "
            "its bytes only match document_versions rows that are 'superseded'/'deleted' under two different, "
            "no-longer-present UUID filenames (Phase 1 finding). This file is NOT one of the three Phase 1 "
            "'genuinely unique unindexed' candidates this phase's copy-based staging is scoped to, so no staging "
            "action was taken on it here -- flagged as a real priority-#5 gap for a follow-up phase to stage and "
            "index through the normal admin ingestion path, with its own human review."
        ),
    },
    {
        "priority": 6,
        "law": "Negotiable Instruments Act, 1881",
        "document_key": "negotiable_instruments_act_1881",
        "canonical_filename": "Negotiable_Instruments_Act_1881_Complete_Act.pdf",
        "duplicate_filenames": ["The_Negotiable_Instruments_Act_1881.pdf"],
        "title_confirmed_from_local_file": "THE NEGOTIABLE INSTRUMENTS ACT, 1881",
        "act_number": "26 of 1881",
        "enactment_date_local_file": None,
        "official_commencement_claim": (
            "Widely reported as commencing 1 March 1882, with major amendments in 2015 and 2018 -- this session's "
            "searches did not return a fetchable primary (indiacode.nic.in/legislative.gov.in) citation for "
            "either the commencement date or the amendment Acts; only secondary sites were returned."
        ),
        "official_source_urls": [],
        "applicability_evidence": (
            "No extent/territorial clause found in the pages read (may be on a later page not reached this pass)."
        ),
        "applicability_claim": "unknown",
        "classification": "needs_effective_date_confirmation",
        "notes": (
            "Title confirmed directly from the local file. This is the WEAKEST-sourced of the seven this pass: "
            "no primary-domain citation was obtained for commencement date, amendment history, or territorial "
            "extent. Needs a dedicated official-source lookup (India Code / Legislative Department) before any "
            "manifest field beyond title/Act-number/type can move past 'unverified'."
        ),
    },
    {
        "priority": 7,
        "law": "Right to Information Act, 2005",
        "document_key": "right_to_information_act_2005",
        "canonical_filename": None,
        "duplicate_filenames": ["The_Right_To_Information_Act_2005.pdf", "21rff_Mere.pdf"],
        "title_confirmed_from_local_file": "THE RIGHT TO INFORMATION ACT, 2005 (local file stamped 'Last Updated: 17-5-2021')",
        "act_number": "22 of 2005",
        "enactment_date_local_file": None,
        "official_commencement_claim": (
            "Assented 15 June 2005; certain provisions (s.4(1), s.5(1)-(2), ss.12,13,15,16,24,27,28) in force "
            "immediately, remainder in force 12 October 2005 (120th day) -- WebSearch-cited against rti.gov.in "
            "(the Department of Personnel & Training's official RTI portal); not directly fetched."
        ),
        "official_source_urls": [
            "https://rti.gov.in/rti%20act,%202005%20(amended)-english%20version.pdf",
        ],
        "applicability_evidence": (
            "Local file Section 1(2), verbatim: 'It extends to the whole of I ndia' (OCR-broken spacing for "
            "'India'). No Jammu & Kashmir exception present in this edition's text."
        ),
        "applicability_claim": "all_india",
        "classification": "not_present",
        "notes": (
            "'The_Right_To_Information_Act_2005.pdf' is genuinely new content (Phase 1: no content-hash match "
            "under any filename, ever) and is content-confirmed as the RTI Act, 2005 from its own text. It has "
            "been safely COPIED into storage/kb_staging (original untouched) and registered in kb_staging_records "
            "as status=needs_review -- see the three-unique-files section. It is NOT yet indexed (0 chunks) and "
            "indexing/embedding it is a separate admin-ingestion action outside this phase's metadata-only "
            "publication mechanism, so 'not_present' remains accurate for the corpus as it stands today. "
            "Separately, 'storage/knowledge_base/21rff_Mere.pdf' (Phase 1: unindexed, historical match to a "
            "superseded/deleted document whose metadata.act_name='Right to Information Act') may be a DIFFERENT "
            "digitization of the same Act -- different content hash, so not a byte-identical duplicate of the "
            "staged file. Flagged for human adjudication, not merged or resolved here."
        ),
    },
]

# The three Phase 1 "genuinely unique unindexed" candidates and what this
# phase's content inspection found them actually to be.
THREE_UNIQUE_FILES: list[dict[str, Any]] = [
    {
        "filename": "503gi_P65.pdf",
        "content_identified_as": "THE CENTRAL GOODS AND SERVICES TAX ACT, 2017 (NO. 12 OF 2017)",
        "in_priority_scope": False,
        "reason": (
            "Confirmed by reading the file's own title page. GST Act, 2017 is not one of the 7 priority laws "
            "and is explicitly out of this phase's scope ('nationwide state-law collection... later work'). "
            "An older GST Act edition (As_On_31_08_2021_4.pdf, 300 chunks) is already indexed separately -- "
            "this file may be a newer edition of the same Act, itself worth a future dedicated review, but not "
            "staged, copied, or indexed in this phase."
        ),
        "action_taken": "none",
    },
    {
        "filename": "Registered_No_Dl_N_04_0007_2003_25jftlv_H_La_Mh_Y_U_04_0007_2003_25_2.pdf",
        "content_identified_as": "THE INCOME-TAX ACT, 2025 (NO. 30 OF 2025) -- Gazette Extraordinary, 21 Aug 2025",
        "in_priority_scope": False,
        "reason": (
            "Confirmed by reading the file's own gazette masthead. Income-tax Act, 2025 is not one of the 7 "
            "priority laws and is out of scope this phase. Not staged, copied, or indexed."
        ),
        "action_taken": "none",
    },
    {
        "filename": "The_Right_To_Information_Act_2005.pdf",
        "content_identified_as": "THE RIGHT TO INFORMATION ACT, 2005 (matches priority #7)",
        "in_priority_scope": True,
        "reason": "See priority-law row 'right_to_information_act_2005' above for full evidence.",
        "action_taken": "copied_to_kb_staging_and_registered_needs_review",
    },
]


async def resolve_canonical_document(source_document: str) -> dict[str, Any] | None:
    """Resolves the LATEST version's own `uploaded_documents` row for a
    filename -- `documents.find_one({"filename": ...})` alone can return a
    STALE row (this codebase inserts a brand-new `documents` row on every
    reindex, never updates the old one in place; confirmed live for BNSS,
    which has 3 such rows, only the latest of which is current)."""
    versions_coll = mongodb.db[DOCUMENT_VERSIONS]
    documents_coll = mongodb.db[UPLOADED_DOCUMENTS]
    versions = [v async for v in versions_coll.find({"source_document": source_document}).sort("version_number", 1)]
    if not versions:
        return None
    latest = versions[-1]
    doc = await documents_coll.find_one({"_id": latest.get("document_id")}) if latest.get("document_id") else None
    return {
        "versions": [
            {"version_number": v["version_number"], "document_status": v["document_status"], "document_hash": v.get("document_hash")}
            for v in versions
        ],
        "canonical_document_id": str(doc["_id"]) if doc else None,
        "canonical_version_id": str(latest["_id"]),
        "document_hash": doc.get("document_hash") if doc else None,
        "review_status": ((doc or {}).get("metadata") or {}).get("review_status"),
        "verification_status": ((doc or {}).get("metadata") or {}).get("verification_status"),
        "act_name_extracted": ((doc or {}).get("metadata") or {}).get("act_name"),
        "chunk_count": doc.get("chunk_count") if doc else None,
        "owner_session_id": doc.get("owner_session_id") if doc else None,
        "owner_user_id": doc.get("owner_user_id") if doc else None,
    }


async def build_review_rows() -> list[dict[str, Any]]:
    rows = []
    for law in PRIORITY_LAWS:
        row = dict(law)
        filename = law["canonical_filename"] or (law["duplicate_filenames"][0] if law["duplicate_filenames"] else None)
        resolved = await resolve_canonical_document(filename) if filename else None
        row["mongo_resolution"] = resolved
        # Live chunk count via the vector store's own filename key.
        chunks_coll = mongodb.db[EMBEDDINGS_METADATA]
        row["live_chunk_count"] = await chunks_coll.count_documents({"metadata.source_document": filename}) if filename else 0
        rows.append(row)
    return rows


async def stage_rti_act(svc: KnowledgeBaseIngestionService) -> dict[str, Any]:
    """Copy-based staging for the one in-scope genuinely-new file.

    The original in `knowledge_base_dir` is left untouched. A COPY is placed
    in `kb_staging_dir` (never moved). `scripts/kb_reconcile.py`'s dry run
    (see the report) would classify this copy as `duplicate -> archive`,
    because its own hash-matching only checks "does this byte-content
    already exist somewhere physically under knowledge_base_dir" -- true
    here, since the original is right there, still unindexed. Applying that
    result would archive the very file we are trying to get reviewed, which
    is not the intended outcome, so `--apply` is deliberately never run;
    instead this registers a `kb_staging_records` ledger row directly,
    reusing the same `_ledger_fields` shape every other ledger row uses,
    with `status="needs_review"` and `current_path` pointing at the STAGED
    COPY (not the original) -- exactly what task 6 asks for, using the
    existing ledger schema/repository rather than a new mechanism.
    """
    source = KB_DIR / "The_Right_To_Information_Act_2005.pdf"
    staged = settings.kb_staging_dir / "The_Right_To_Information_Act_2005.pdf"
    settings.kb_staging_dir.mkdir(parents=True, exist_ok=True)

    content_hash = svc.quality.hash_file(source)
    collision = await svc.staging.find_active_by_content_hash(content_hash)
    if collision is not None:
        return {"staged": False, "blocker": f"active ledger collision on this content hash: {collision['_id']}"}

    if not staged.exists():
        shutil.copy2(source, staged)
    elif svc.quality.hash_file(staged) != content_hash:
        return {"staged": False, "blocker": f"a different file already exists at {staged} -- not overwritten"}

    existing_row = await svc.staging.find_by_content_hash(content_hash)
    if existing_row is not None:
        return {
            "staged": True,
            "ledger_id": str(existing_row["_id"]),
            "status": existing_row.get("status"),
            "note": "already registered by an earlier run of this script -- no duplicate row created",
        }

    fields = svc._ledger_fields(
        original_filename=source.name,
        status="needs_review",
        content_hash=content_hash,
        current_path=staged,
        destination_path=settings.kb_review_dir / "pending",
        reason=(
            "Phase 2 priority-law review: genuinely new, never-indexed content confirmed as the Right to "
            "Information Act, 2005 (priority #7). Original file in knowledge_base_dir preserved; this is a copy "
            "staged for admin review. NOT auto-approved and NOT auto-indexed -- indexing is a separate, later "
            "admin-ingestion decision."
        ),
        ingestion_source="phase2_priority_review",
    )
    ledger_id = await svc.staging.insert(fields)
    return {"staged": True, "ledger_id": ledger_id, "status": "needs_review", "staged_path": str(staged), "content_hash": content_hash}


async def build_conflicts() -> list[dict[str, Any]]:
    return [
        {
            "type": "duplicate_or_ambiguous",
            "document_key": "bharatiya_nyaya_sanhita_2023",
            "description": (
                "Bharatiya_Nyaya_Sanhita_2023_Complete_Act.pdf (unindexed) is content-identical to a "
                "SUPERSEDED document_versions row (cf2b925a-...pdf), not to the currently-indexed canonical "
                "copy (The_Bhara_Tiy_A_Ny_A_Y_A_Sanhita_2023_5.pdf, different hash/OCR). The canonical indexed "
                "copy is treated as authoritative; the unindexed duplicate is left untouched pending admin "
                "adjudication of why its earlier indexed twin was removed."
            ),
        },
        {
            "type": "duplicate_or_ambiguous",
            "document_key": "bharatiya_sakshya_adhiniyam_2023",
            "description": (
                "The_Bharatiya_Sakshya_Adhiniyam_2023_2.pdf (unindexed) content-matches a DELETED "
                "document_versions row (58bdf7cf-...pdf), distinct from the canonical indexed copy "
                "(The_Bhara_Tiy_A_Sakshy_A_Adhiniy_Am_2023_6.pdf). Left untouched."
            ),
        },
        {
            "type": "duplicate_or_ambiguous",
            "document_key": "negotiable_instruments_act_1881",
            "description": (
                "The_Negotiable_Instruments_Act_1881.pdf (unindexed) content-matches a superseded-then-deleted "
                "document_versions row (99ac8b0e-...pdf), distinct from the canonical indexed copy "
                "(Negotiable_Instruments_Act_1881_Complete_Act.pdf). Left untouched."
            ),
        },
        {
            "type": "duplicate_or_ambiguous",
            "document_key": "right_to_information_act_2005",
            "description": (
                "21rff_Mere.pdf (unindexed) content-matches a superseded document_versions row whose "
                "metadata.act_name='Right to Information Act' -- possibly a DIFFERENT digitization/edition of "
                "the RTI Act than the newly-staged The_Right_To_Information_Act_2005.pdf (different hash). Not "
                "merged, not staged; flagged for human adjudication."
            ),
        },
        {
            "type": "not_priority_content",
            "document_key": None,
            "description": (
                "503gi_P65.pdf is the Central Goods and Services Tax Act, 2017 (confirmed from its own title "
                "page) -- not a priority law. No action taken."
            ),
        },
        {
            "type": "not_priority_content",
            "document_key": None,
            "description": (
                "Registered_No_Dl_N_04_0007_2003_25jftlv_H_La_Mh_Y_U_04_0007_2003_25_2.pdf is the Income-tax "
                "Act, 2025 (confirmed from its own gazette masthead) -- not a priority law. No action taken."
            ),
        },
        {
            "type": "consolidated_vs_original_text",
            "document_key": "bharatiya_nagarik_suraksha_sanhita_2023",
            "description": (
                "The indexed BNSS file's own title page is stamped '[As on the 6th October, 2025]' -- a "
                "consolidated/updated-to-date India Code-style edition, not the original 25-Dec-2023 "
                "Gazette-as-enacted text. Needs a human decision on whether the consolidated text is what the "
                "shared KB should present as this Act, and whether that should be recorded in version_label."
            ),
        },
        {
            "type": "missing_official_source_confirmation",
            "document_key": "negotiable_instruments_act_1881",
            "description": (
                "No primary (.gov.in/indiacode.nic.in) citation was obtained this session for NI Act 1881's "
                "commencement date, amendment history, or territorial extent -- only secondary sites. Needs a "
                "dedicated follow-up official-source lookup before this document can move past 'unverified'."
            ),
        },
    ]


def build_publication_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every entry here is `verification_status=pending_review`, never
    `approved`/`verified` -- this manifest is the PROPOSED replacement
    metadata for human review, not something already applied to any
    document. `publish_approved` (below) is the only path that can move an
    entry from here into `review_status=approved`, and only for keys an
    authenticated operator explicitly names."""
    entries = []
    for row in rows:
        entries.append(
            {
                "document_key": row["document_key"],
                "law": row["law"],
                "canonical_filename": row["canonical_filename"],
                "issuing_level": "central",
                "applicability": (
                    "unknown" if row["applicability_claim"] in ("unknown",) else row["applicability_claim"]
                ),
                "applicable_state_codes": [],
                "applicable_localities": [],
                "jurisdiction_source_type": "bare_act",
                "source_url": (row["official_source_urls"] or [None])[0],
                "version_label": row.get("act_number"),
                "effective_from": None,  # never fabricated; see review notes for the claimed date
                "effective_to": None,
                "amends": None,
                "amended_by": None,
                "supersedes": None,
                "superseded_by": None,
                "verification_status": "pending_review",
                "review_status_target": "needs_review",  # unchanged until a human approves
                "verified_by": None,
                "metadata_provenance": PROVENANCE_INFERRED,
                "section_overrides": {},
                "review_notes": row["notes"],
                "evidence_urls": row["official_source_urls"],
                "classification": row["classification"],
            }
        )
    return {"schema_version": 1, "status": "pending_review", "entries": entries}


async def cache_generation() -> str | None:
    try:
        raw = await redis_client.client.get(f"{CACHE_KEY_PREFIX}:kb-generation")
        return raw.decode() if isinstance(raw, bytes) else raw
    except Exception:  # noqa: BLE001 - offline audit reports unavailable Redis as unknown
        return None


async def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    await mongodb.connect()
    await redis_client.connect()
    svc = KnowledgeBaseIngestionService()

    print("Resolving priority-law identities against MongoDB (read-only)...")
    rows = await build_review_rows()
    for row in rows:
        print(f"  #{row['priority']} {row['law']}: classification={row['classification']} chunks={row['live_chunk_count']}")

    print("Staging the one in-scope unique file (RTI Act 2005) via copy + ledger entry...")
    staging_result = await stage_rti_act(svc)
    print(f"  {staging_result}")

    print("Running kb_reconcile.py-equivalent dry run for transparency (not applied)...")
    from app.services.kb_ingestion_service import KnowledgeBaseIngestionService as _KBIS

    reconcile_report = await _KBIS().reconcile_staging(apply=False)
    reconcile_summary = {
        k: reconcile_report[k]
        for k in ("mode", "scanned", "archive", "needs_review", "failed_review", "ledger_missing_file", "missing_ledger_paths")
    }
    print(f"  {reconcile_summary}")

    conflicts = await build_conflicts()
    manifest = build_publication_manifest(rows)
    gen = await cache_generation()

    review_package = {
        "schema_version": 1,
        "phase1_baseline_reused": True,
        "priority_laws": rows,
        "three_unique_files": THREE_UNIQUE_FILES,
        "rti_staging_result": staging_result,
        "reconcile_staging_dry_run_note": (
            "scripts/kb_reconcile.py's own reconcile_staging(apply=False) was run for transparency. It would "
            "classify the staged RTI copy as outcome='duplicate' (destination=archive_dir), because its "
            "duplicate check is 'does this hash already exist anywhere under knowledge_base_dir' -- true here, "
            "since the original unindexed file is right there. Applying that result would archive the very "
            "file being staged for review, so --apply was never run against it; a direct kb_staging_records "
            "ledger entry was used instead (see rti_staging_result)."
        ),
        "reconcile_staging_dry_run": reconcile_summary,
        "cache_generation_current": gen,
    }
    (AUDIT_DIR / "phase2_priority_review.json").write_text(json.dumps(review_package, indent=2, default=str), encoding="utf-8")
    (AUDIT_DIR / "phase2_publication_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    (AUDIT_DIR / "phase2_conflicts.json").write_text(json.dumps({"conflicts": conflicts}, indent=2, default=str), encoding="utf-8")

    print("Wrote phase2_priority_review.json, phase2_publication_manifest.json, phase2_conflicts.json")
    print("No document_status/review_status/verification_status was changed. No file was indexed, moved, or deleted.")

    await mongodb.close()


# ---------------------------------------------------------------------------
# NOT called by main(). Only for use after explicit human/operator approval,
# invoked separately with the exact approved document_keys and a real
# operator id. Left here (implemented, not executed) so the approval step
# has no new code path to write once approval is given.
# ---------------------------------------------------------------------------
async def publish_approved(
    approved_document_keys: list[str], *, operator_id: str, manifest_path: Path | None = None
) -> dict[str, Any]:
    if not operator_id:
        raise ValueError("verified_by requires an authenticated operator id -- refusing to publish anonymously.")
    manifest = json.loads((manifest_path or (AUDIT_DIR / "phase2_publication_manifest.json")).read_text(encoding="utf-8"))
    documents_coll = mongodb.db[UPLOADED_DOCUMENTS]
    chunks_coll = mongodb.db[EMBEDDINGS_METADATA]
    results = []
    for entry in manifest["entries"]:
        if entry["document_key"] not in approved_document_keys:
            continue
        law = next(law for law in PRIORITY_LAWS if law["document_key"] == entry["document_key"])
        filename = law["canonical_filename"]
        if not filename:
            results.append({"document_key": entry["document_key"], "status": "skipped", "reason": "no canonical indexed file to publish"})
            continue
        try:
            raw = {
                "issuing_level": entry["issuing_level"],
                "applicability": entry["applicability"],
                "jurisdiction_source_type": entry["jurisdiction_source_type"],
                "source_url": entry["source_url"],
                "version_label": entry["version_label"],
                "verification_status": "verified",
                "verified_by": operator_id,
                "review_status": "approved",
            }
            normalized = normalize_jurisdiction(raw, provenance="manual")
            fields = document_metadata_fields(normalized)
            doc = await documents_coll.find_one({"filename": filename})
            chunks_updated = await propagate_jurisdiction_metadata(
                documents_coll, chunks_coll, document_id=doc["_id"], source_document=filename,
                fields=fields, section_overrides=normalized.section_overrides,
            )
            results.append({"document_key": entry["document_key"], "status": "published", "chunks_updated": chunks_updated})
        except Exception as exc:  # noqa: BLE001 - one failed publish must not abandon the rest; quarantine, don't half-apply
            results.append({"document_key": entry["document_key"], "status": "failed", "error": str(exc)})
    if any(r["status"] == "published" for r in results):
        await response_cache.bump_generation()
        await bm25_index.rebuild()
    return {"operator_id": operator_id, "results": results}


if __name__ == "__main__":
    asyncio.run(main())
