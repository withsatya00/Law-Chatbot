"""Fetch and safely ingest configured official KB source replacements."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from app.cache.redis_client import redis_client
from app.core.windows_runtime import repair_windows_host_env
from app.database.mongodb import mongodb
from app.services.kb_official_source_sync import OfficialSourceSyncService


async def main(if_due_hours: int = 0) -> None:
    repair_windows_host_env()
    report = Path("storage/kb_audit/official_source_sync_latest.json")
    if if_due_hours and report.is_file():
        age_seconds = datetime.now(UTC).timestamp() - report.stat().st_mtime
        if age_seconds < if_due_hours * 3600:
            print(json.dumps({"status": "not_due", "age_seconds": int(age_seconds)}), flush=True)
            return
    await mongodb.connect()
    await redis_client.connect()
    try:
        result = await OfficialSourceSyncService().run()
        result["ran_at"] = datetime.now(UTC).isoformat()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        print(json.dumps(result, default=str), flush=True)
    finally:
        await redis_client.close()
        await mongodb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--if-due-hours", type=int, default=0, help="Skip when the latest report is newer.")
    asyncio.run(main(parser.parse_args().if_due_hours))
