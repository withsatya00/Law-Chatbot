"""Durable Phase 3 worker for export and large-document jobs."""

import asyncio
import socket

from app.core.config import settings
from app.database.mongodb import mongodb
from app.services.phase3 import BackgroundJobService


async def main() -> None:
    await mongodb.connect()
    worker_id = f"{socket.gethostname()}-{id(asyncio.current_task())}"
    try:
        while True:
            processed = await BackgroundJobService().process_next(worker_id)
            if not processed:
                await asyncio.sleep(settings.background_job_poll_seconds)
    finally:
        await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
