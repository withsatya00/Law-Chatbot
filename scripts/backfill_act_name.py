"""One-off backfill for `app/rag/metadata.py`'s `ACT_NAME_RE` fix.

The pre-fix regex was case-sensitive on the suffix keyword ("Act", never
"ACT") and required every word -- including ordinary lowercase connectors
like "to"/"of" -- to start capitalized, so it silently mismatched most
document title lines (usually rendered ALL-CAPS on a gazette cover) and
truncated most real multi-word titles. It also had no defense against the
extremely common generic two-word phrases "An Act"/"This Act" that appear
in nearly every Indian Act's own preamble and cross-references, so those
frequently won outright. Combined, ~30% of the corpus (987 of 3344 chunks
at the time this was written) carried a useless `act_name` of "An Act"/
"This Act", and another ~20% had none at all.

Picking a single representative name PER DOCUMENT (rather than trusting
whichever chunk happens to self-resolve) went through two failed designs
before this one, both worth recording so a future pass doesn't repeat them:
  1. Longest matched string among a document's own chunks -- wrong whenever
     the document cross-references a DIFFERENT Act whose name happens to be
     a longer string (confirmed live: the CGST Act's own document nearly
     got relabeled "Union Territory Goods and Services Tax Act", 4
     characters longer than CGST's own title).
  2. Most FREQUENT non-generic match among a document's own chunks -- also
     wrong, for the same underlying reason: an Act's operative text cites
     OTHER Acts by name repeatedly as part of routine cross-referencing
     ("...registered under the State Goods and Services Tax Act..."),
     which can easily outnumber how often the document states its OWN name
     (typically once, in Section 1's short-title clause).
This version instead searches for the standard Indian short-title clause
every Act's own Section 1 contains -- "This Act may be called the ...
Act" -- which names the document's own Act, once, unambiguously, and
never a cross-reference (a document doesn't call ANOTHER Act "this Act").
Falls back to the (still-imperfect but better-than-nothing) frequency vote
only for the minority of documents where that clause isn't found at all
(OCR noise, or a document that isn't structured as a standalone Act).

Usage:
    python scripts/backfill_act_name.py [--dry-run]
"""

import argparse
import asyncio
import re

import structlog

from app.core.logger import configure_logging
from app.database.mongodb import mongodb
from app.models.collections import EMBEDDINGS_METADATA
from app.rag.metadata import ACT_NAME_RE, _is_generic_act_name

log = structlog.get_logger(__name__)

_GENERIC_ACT_NAME_VALUES = {
    f"{leader} {suffix}"
    for leader in ("an", "this", "that", "any", "such", "every", "no", "said", "which", "whose", "same", "aforesaid")
    for suffix in ("act", "sanhita", "adhiniyam", "code")
}

# The standard Indian legal-drafting short-title clause, e.g. "This Act may
# be called the Central Goods and Services Tax Act, 2017." -- deliberately
# NOT anchored to "This Act" specifically (some Acts render it "It may be
# called..."), just the "may be called the <name>" core that's common to
# both phrasings.
_SHORT_TITLE_RE = re.compile(r"may\s+be\s+called\s+the\s+([A-Za-z][A-Za-z ,]*?(?:Act|Sanhita|Adhiniyam|Code))\b", re.IGNORECASE)


def _best_act_name(text: str) -> str | None:
    normalized = re.sub(r"\s+", " ", text)
    match = next(
        (m for m in ACT_NAME_RE.finditer(normalized) if not _is_generic_act_name(m.group(1))), None
    )
    return match.group(1).strip() if match else None


def _short_title_name(text: str) -> str | None:
    match = _SHORT_TITLE_RE.search(re.sub(r"\s+", " ", text))
    return match.group(1).strip().rstrip(",") if match else None


async def backfill(dry_run: bool) -> None:
    await mongodb.connect()
    collection = mongodb.db[EMBEDDINGS_METADATA]

    total = await collection.count_documents({})
    log.info("act_name_backfill_starting", total_chunks=total, dry_run=dry_run)

    # Single scan: for each document, remember the first short-title match
    # found (tier 1) and tally frequency of fresh per-chunk extractions
    # (tier 2, fallback only) at the same time.
    short_title_by_document: dict[str, str] = {}
    per_document_counts: dict[str, dict[str, int]] = {}
    cursor = collection.find({}, {"text": 1, "metadata.source_document": 1})
    async for doc in cursor:
        source_document = doc.get("metadata", {}).get("source_document")
        if not source_document:
            continue
        text = doc.get("text", "")
        if source_document not in short_title_by_document:
            short_name = _short_title_name(text)
            if short_name:
                short_title_by_document[source_document] = short_name
        name = _best_act_name(text)
        if name:
            bucket = per_document_counts.setdefault(source_document, {})
            bucket[name] = bucket.get(name, 0) + 1

    frequency_fallback = {
        source_document: max(counts.items(), key=lambda item: item[1])[0]
        for source_document, counts in per_document_counts.items()
    }
    resolved_name_by_document = {**frequency_fallback, **short_title_by_document}  # tier 1 wins where both exist

    updated = 0
    from_short_title = 0
    from_frequency = 0
    cursor = collection.find({}, {"metadata.source_document": 1, "metadata.act_name": 1})
    async for doc in cursor:
        source_document = doc.get("metadata", {}).get("source_document")
        resolved_name = resolved_name_by_document.get(source_document)
        if not resolved_name:
            continue
        current = doc.get("metadata", {}).get("act_name")
        if current == resolved_name:
            continue
        if not dry_run:
            await collection.update_one({"_id": doc["_id"]}, {"$set": {"metadata.act_name": resolved_name}})
        updated += 1
        if source_document in short_title_by_document:
            from_short_title += 1
        else:
            from_frequency += 1

    unresolved_documents = [
        source_document
        for source_document in await collection.distinct("metadata.source_document")
        if source_document and source_document not in resolved_name_by_document
    ]

    log.info(
        "act_name_backfill_complete",
        total_chunks=total,
        documents_resolved_via_short_title_clause=len(short_title_by_document),
        documents_resolved_via_frequency_fallback=len(frequency_fallback) - len(
            set(frequency_fallback) & set(short_title_by_document)
        ),
        documents_unresolved=len(unresolved_documents),
        unresolved_document_names=unresolved_documents[:20],
        chunks_updated=updated,
        chunks_updated_from_short_title=from_short_title,
        chunks_updated_from_frequency_fallback=from_frequency,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report counts without writing any changes.")
    args = parser.parse_args()
    configure_logging()
    asyncio.run(backfill(args.dry_run))
