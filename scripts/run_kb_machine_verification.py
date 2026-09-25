"""Run conservative official-source verification and publication."""

import argparse
import asyncio
import json
from pathlib import Path

from app.cache.redis_client import redis_client
from app.core.windows_runtime import repair_windows_host_env
from app.database.mongodb import mongodb
from app.services.kb_machine_verification import MachineVerificationService


async def main(force: bool = False) -> None:
    repair_windows_host_env()
    await mongodb.connect()
    await redis_client.connect()
    try:
        result = await MachineVerificationService().run(force=force)
        report = Path("storage/kb_audit/machine_verification_latest.json")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        print(json.dumps(result, default=str), flush=True)
    finally:
        await redis_client.close()
        await mongodb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    asyncio.run(main(parser.parse_args().force))
