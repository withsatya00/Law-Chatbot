"""Synthetic 4,000-chunk update/event-loop probe; no database or LLM calls.

Run from the project root: python -m scripts.benchmark_bm25_update
"""
import asyncio
import tempfile
import time
from pathlib import Path

from app.rag.bm25_index import BM25Index
from app.rag.types import DocumentChunk


async def main():
    index = BM25Index(Path(tempfile.mkdtemp()) / "index.pkl")
    n = 4000
    index._set_corpus(
        [str(i) for i in range(n)],
        [
            ("legal notice agreement tenant rent deposit terms section " + str(i) + " ") * 50
            for i in range(n)
        ],
        [{"document_status": "active"} for _ in range(n)],
    )
    ticks = []

    async def probe():
        end = time.perf_counter() + 2
        while time.perf_counter() < end:
            t = time.perf_counter()
            await asyncio.sleep(0.01)
            ticks.append(time.perf_counter() - t)

    task = asyncio.create_task(probe())
    await asyncio.sleep(0.02)
    t = time.perf_counter()
    await index.add_or_update_chunks(
        [
            DocumentChunk(
                chunk_id="new",
                document_id="doc",
                text="tenant notice deposit",
                metadata={"document_status": "active"},
                embedding=[],
            )
        ]
    )
    elapsed = time.perf_counter() - t
    await task
    print({"update_seconds": round(elapsed, 3), "max_event_loop_gap_seconds": round(max(ticks), 3)})


if __name__ == "__main__":
    asyncio.run(main())
