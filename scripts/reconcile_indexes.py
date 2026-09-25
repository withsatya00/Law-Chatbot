"""Reconcile the BM25 index against MongoDB. Dry-run by default.

MongoDB is the source of truth. This reports what the two stores disagree
about and, with `--apply`, prunes BM25 entries whose chunk id no longer exists
in Mongo. It never writes to Mongo, never deletes an uploaded file, and never
alters ownership metadata -- see `app/rag/reconciliation.py` for why.

Usage:
    .venv\\Scripts\\python.exe scripts\\reconcile_indexes.py
    .venv\\Scripts\\python.exe scripts\\reconcile_indexes.py --apply
    .venv\\Scripts\\python.exe scripts\\reconcile_indexes.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database.mongodb import mongodb
from app.rag.reconciliation import IndexReconciler


async def run(*, apply: bool, rebuild_missing: bool) -> dict[str, Any]:
    await mongodb.connect()
    reconciler = IndexReconciler()
    report = await (reconciler.apply() if apply else reconciler.analyze())
    payload = report.as_dict()
    if rebuild_missing and report.missing_from_bm25_count:
        # A full rebuild from Mongo, which is the source of truth. Kept behind
        # its own flag rather than folded into --apply: --apply only removes
        # index entries that provably should not exist, while this re-tokenizes
        # every chunk and rewrites the whole corpus. Those are different-sized
        # actions and an operator should choose the larger one deliberately.
        from app.rag.bm25_index import bm25_index

        payload["rebuilt_chunk_count"] = await bm25_index.rebuild()
        payload["drifted"] = False
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="prune stale BM25 entries (default is dry-run)")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--rebuild-missing",
        action="store_true",
        help="after reporting, rebuild the whole BM25 corpus from MongoDB "
             "(the fix for chunks present in Mongo but absent from the index)",
    )
    args = parser.parse_args()

    report = asyncio.run(run(apply=args.apply, rebuild_missing=args.rebuild_missing))
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    for key, value in report.items():
        print(f"{key}: {value}")
    if report["stale_private_records"]:
        print(
            f"\nWARNING: {report['stale_private_records']} stale entries carry owner metadata. "
            "Those are retrievable copies of private uploads that Mongo no longer has."
        )
    if not args.apply and report["drifted"]:
        print("\nDry run. Re-run with --apply to prune the stale BM25 entries.")
    elif not report["drifted"]:
        print("\nIndexes are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
