"""Dry-run by default. Stop API/indexing workers before --apply --writers-stopped."""
import argparse
import asyncio
import json

from app.cache.redis_client import redis_client
from app.database.mongodb import mongodb
from app.services.reindex_recovery import recover_interrupted_reindexes


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--writers-stopped", action="store_true")
    args = parser.parse_args()
    if args.apply and not args.writers_stopped:
        parser.error("--apply requires --writers-stopped; stop API and all indexing workers first")
    await mongodb.connect()
    await redis_client.connect()
    try:
        print(json.dumps(await recover_interrupted_reindexes(apply=args.apply, writers_stopped=args.writers_stopped), indent=2))
    finally:
        await redis_client.close()
        await mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())
