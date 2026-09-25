"""Operator entry point for KB staging reconciliation.

DRY RUN by default -- prints what would move, where and why, and touches
neither the filesystem nor Mongo. `--apply` writes a JSON manifest under
`operations_output_dir` first, then performs the moves; pass `--expect N` to
refuse the apply unless the staging directory holds exactly N files, so a plan
reviewed against one state cannot be executed against a different one.

    python scripts/kb_reconcile.py
    python scripts/kb_reconcile.py --apply --expect 68

Nothing is ever deleted: already-indexed content is archived, corrupt/failed
documents go to the failed review queue, and everything unverified goes to the
pending review queue as `needs_review` for an admin to approve explicitly.
"""

import argparse
import asyncio
from collections import Counter

import structlog

from app.core.logger import configure_logging
from app.database.mongodb import mongodb
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService

log = structlog.get_logger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile storage/kb_staging against the staging ledger.")
    parser.add_argument("--apply", action="store_true", help="Perform the moves (default is a dry run).")
    parser.add_argument("--expect", type=int, default=None, help="Refuse to apply unless this many files are staged.")
    parser.add_argument("--limit", type=int, default=15, help="How many proposed moves to print.")
    args = parser.parse_args()

    configure_logging()
    await mongodb.connect()
    try:
        report = await KnowledgeBaseIngestionService().reconcile_staging(
            apply=args.apply, expected_total=args.expect
        )
    finally:
        await mongodb.close()

    print(f"mode: {report['mode']}")
    for key in (
        "scanned",
        "archive",
        "failed_review",
        "needs_review",
        "delete",
        "auto_index",
        "ledger_missing_file",
        "orphaned",
        "missing_ledger_paths",
        "stale_processing_records",
    ):
        print(f"  {key}: {report[key]}")
    if report.get("manifest_path"):
        print(f"  manifest: {report['manifest_path']}")
    if report.get("applied"):
        print(f"  moved: {report['moved']}")

    codes = Counter(item["reason_code"] for item in report["moves"])
    for code, count in sorted(codes.items()):
        print(f"  reason_code {code}: {count}")
    destinations = Counter(item["destination_dir"] for item in report["moves"])
    for destination, count in destinations.items():
        print(f"  -> {destination}: {count}")
    for item in report["moves"][: args.limit]:
        print(f"    [{item['outcome']}] {item['source']} -> {item['proposed_destination']}")
    remaining = len(report["moves"]) - args.limit
    if remaining > 0:
        print(f"    ... and {remaining} more")


if __name__ == "__main__":
    asyncio.run(main())
