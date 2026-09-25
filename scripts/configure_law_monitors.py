"""Register reviewed URL configuration; does not publish legal documents.

python -m scripts.configure_law_monitors config/law_monitors.json --actor local-operator
"""

import argparse
import asyncio
import json
from pathlib import Path

from app.database.mongodb import mongodb
from app.schemas.law_monitoring import MonitorRequest
from app.services.law_monitoring import LawMonitoringService


async def main(path: Path, actor: str) -> None:
    requests = [MonitorRequest.model_validate(row) for row in json.loads(path.read_text(encoding="utf-8"))]
    await mongodb.connect()
    try:
        service = LawMonitoringService()
        await service.ensure_indexes()
        for request in requests:
            result = await service.register(request, actor)
            print(json.dumps({"id": result["_id"], "url": request.url, "enabled": request.enabled}), flush=True)
    finally:
        await mongodb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--actor", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.config, args.actor))
