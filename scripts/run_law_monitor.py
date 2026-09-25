"""Scheduled official-source checks. No indexing or automatic legal approval.

python -m scripts.run_law_monitor --once
python -m scripts.run_law_monitor
"""

import argparse
import asyncio
import json
from datetime import UTC, datetime

from app.database.mongodb import mongodb
from app.services.law_monitoring import LawMonitoringService


async def main(once: bool = False) -> None:
    from app.core.windows_runtime import repair_windows_host_env
    repair_windows_host_env()
    await mongodb.connect()
    try:
        service = LawMonitoringService()
        await service.ensure_indexes()
        while True:
            try:
                result = await service.check_due()
            except Exception as exc:
                result = {"worker_error": type(exc).__name__}
                if once:
                    raise
            print(json.dumps({"at": datetime.now(UTC).isoformat(), **result}), flush=True)
            if once:
                break
            await asyncio.sleep(60)
    finally:
        await mongodb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    asyncio.run(main(parser.parse_args().once))
