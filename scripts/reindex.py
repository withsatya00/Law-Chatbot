"""Incrementally re-indexes the knowledge-base directory.

Detects new, updated, and deleted files since the last run and only touches those —
the full vector database is never rebuilt. Intended to be run from a deployment
pipeline or a periodic cron job, independent of the API's `/admin/reindex` endpoint
(which does the same work for ad-hoc, request-triggered runs).

Usage:
    python scripts/reindex.py [--root ./storage/knowledge_base]
"""

import argparse
import asyncio
from pathlib import Path

from app.cache.redis_client import redis_client
from app.core.config import settings
from app.database.mongodb import mongodb
from app.rag.incremental import IncrementalReindexRunner


async def main(root: Path) -> None:
    await mongodb.connect()
    await redis_client.connect()
    try:
        root.mkdir(parents=True, exist_ok=True)
        runner = IncrementalReindexRunner()
        summary = await runner.run(root)
        print(
            f"Reindex complete: {len(summary['new_indexed'])} new, "
            f"{len(summary['updated_indexed'])} updated, "
            f"{len(summary['deleted'])} deleted, "
            f"{len(summary['unchanged'])} unchanged, "
            f"{len(summary['failed'])} failed."
        )
        for failure in summary["failed"]:
            print(f"  FAILED: {failure['file']} -> {failure['error']}")
    finally:
        await redis_client.close()
        await mongodb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally re-index the legal knowledge base.")
    parser.add_argument("--root", type=Path, default=settings.knowledge_base_dir, help="Knowledge base root directory.")
    args = parser.parse_args()
    asyncio.run(main(args.root))
