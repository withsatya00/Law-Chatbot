"""Manually ingest a scanned (image-only) official PDF from the source registry.

Use for registry sources marked `requires_ocr=True` (e.g. IT_ACT_2008_AMENDMENT),
which `scripts/sync_official_kb_sources.py` deliberately skips. OCRs the file,
checks it identifies itself, records per-page OCR confidence/provenance, and
indexes it as `needs_review`. It NEVER verifies or approves: a human reviewer
must compare the OCR text with the page images and use the admin review route.

    python scripts/ingest_scanned_official_pdf.py --law IT_ACT_2008_AMENDMENT --dry-run
    python scripts/ingest_scanned_official_pdf.py --law IT_ACT_2008_AMENDMENT
    python scripts/ingest_scanned_official_pdf.py --law IT_ACT_2008_AMENDMENT --file C:\\path\\to\\copy.pdf

`--dry-run` downloads/OCRs and writes the evidence manifest but touches neither
MongoDB nor the knowledge base. The source URL always comes from the registry;
`--file` only replaces the download (its SHA-256 is recorded).
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

from app.cache.redis_client import redis_client
from app.core.windows_runtime import repair_windows_host_env
from app.database.mongodb import mongodb
from app.services.kb_official_source_sync import SOURCES, fetch_official_document
from app.services.kb_scanned_ingestion import ScannedOfficialPdfIngestion

_WINDOWS_TESSERACT_DIR = Path(r"C:\Program Files\Tesseract-OCR")


def ensure_tesseract_on_path() -> None:
    """The stock Windows Tesseract installer does not add itself to PATH, and
    pytesseract (used by both the evidence pass and the indexing loader) only
    looks there. Process-local; nothing on the machine is changed."""
    if shutil.which("tesseract") is None and (_WINDOWS_TESSERACT_DIR / "tesseract.exe").is_file():
        os.environ["PATH"] = f"{_WINDOWS_TESSERACT_DIR}{os.pathsep}{os.environ.get('PATH', '')}"


async def main(law: str, file: Path | None, dry_run: bool) -> int:
    source = next((item for item in SOURCES if item.key == law), None)
    if source is None or not source.requires_ocr:
        scanned = [item.key for item in SOURCES if item.requires_ocr]
        print(json.dumps({"status": "rejected", "reason": f"--law must be one of {scanned}."}))
        return 2
    repair_windows_host_env()
    ensure_tesseract_on_path()
    if file is not None:
        body, acquisition = file.read_bytes(), "operator-supplied local file (SHA-256 recorded)"
    else:
        body, _content_type = await fetch_official_document(source.url)
        acquisition = "downloaded from the registry URL"
    if not dry_run:
        await mongodb.connect()
        await redis_client.connect()
    try:
        result = await ScannedOfficialPdfIngestion().ingest(
            source, body, acquisition=acquisition, dry_run=dry_run,
        )
    finally:
        if not dry_run:
            await redis_client.close()
            await mongodb.close()
    print(json.dumps(result, default=str, indent=2))
    return 0 if result["status"] in ("indexed_needs_review", "dry_run_ok") else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--law", required=True, help="Registry key of a requires_ocr source.")
    parser.add_argument("--file", type=Path, help="Local copy of the official PDF instead of downloading it.")
    parser.add_argument("--dry-run", action="store_true", help="OCR and write the manifest only; no DB writes.")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.law, args.file, args.dry_run)))
