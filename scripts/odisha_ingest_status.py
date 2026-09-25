r"""Show read-only progress for the Odisha Acts bulk ingestion.

Run from the ``legal_ai_assistant`` directory::

    .\.venv\Scripts\python.exe -m scripts.odisha_ingest_status

Optionally pass the manifest path when it is no longer available at the
original scratchpad location::

    .\.venv\Scripts\python.exe -m scripts.odisha_ingest_status --manifest C:\path\to\odisha_acts_manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from app.core.windows_runtime import repair_windows_host_env

repair_windows_host_env()

from app.database.mongodb import mongodb  # noqa: E402


DEFAULT_MANIFEST = Path(
    "C:/Users/og/AppData/Local/Temp/claude/c--Law-Chatbot-Law-Chatbot/"
    "4b27167c-7126-41cb-b62f-f1d59fae4356/scratchpad/odisha_acts_manifest.json"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


async def _status(manifest_path: Path) -> None:
    manifest: list[dict[str, Any]] = []
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    await mongodb.connect()
    try:
        documents = mongodb.db["uploaded_documents"]
        chunks = mongodb.db["embeddings_metadata"]

        # The batch script generates every filename with this stable prefix.
        document_filter = {"filename": {"$regex": "^odisha_acts_", "$options": "i"}}
        chunk_filter = {
            "metadata.source_document": {
                "$regex": "^odisha_acts_",
                "$options": "i",
            }
        }
        document_count = await documents.count_documents(document_filter)
        chunk_count = await chunks.count_documents(chunk_filter)
        source_count = len(await chunks.distinct("metadata.source_document", chunk_filter))
        indexed_urls = set(await documents.distinct("metadata.source_url", document_filter))

        print("Odisha Acts ingestion status")
        print(f"  Indexed documents: {document_count}")
        print(f"  Distinct chunk sources: {source_count}")
        print(f"  Indexed chunks: {chunk_count}")
        if manifest:
            total = len(manifest)
            completed = min(document_count, total)
            print(f"  Manifest total: {total}")
            print(f"  Approx. completion: {completed}/{total} ({completed / total:.1%})")
            print(f"  Approx. remaining: {max(total - completed, 0)}")
            positions = [
                index
                for index, row in enumerate(manifest, start=1)
                if row.get("pdf_url") in indexed_urls
            ]
            if positions:
                highest = max(positions)
                print(f"  Highest manifest item indexed: {highest}/{total}")
                print(f"  Manifest URLs indexed: {len(positions)}/{total}")
            print(
                "  Note: duplicates and failed downloads require the ingestion log "
                "for an exact processed-item count."
            )
        else:
            print(f"  Manifest unavailable: {manifest_path}")
            print("  Completion percentage cannot be calculated without it.")
    finally:
        await mongodb.close()


if __name__ == "__main__":
    asyncio.run(_status(_arguments().manifest))
