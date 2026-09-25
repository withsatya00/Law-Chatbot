"""Approve the exact CBIC 2021 compilation; dry run unless --apply is supplied."""

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from app.cache.redis_client import redis_client
from app.core.config import settings
from app.database.mongodb import mongodb
from app.services.kb_machine_verification import MachinePolicy, MachineVerificationService

POLICY = MachinePolicy(
    key="CGST_2021_SNAPSHOT",
    filename="As_On_31_08_2021_4.pdf",
    official_url="https://cbic-gst.gov.in/pdf/CGST-Act-Updated-31082021.pdf",
    commencement_url="https://cbic-gst.gov.in/hindi/pdf/central-tax/notfctn-9-central-tax-english.pdf",
    document_key="central-goods-and-services-tax-2017",
    version_label="CBIC reference compilation as on 2021-08-31; not a current consolidation",
    identity_tokens=("centralgoodsandservicestaxact2017", "ason31082021"),
    applicability_tokens=("extendstothewholeofindia", "8thjuly2017"),
    commencement_tokens=("notificationno92017", "1stdayofjuly2017", "shallcomeintoforce"),
    # This is the consolidation's as-of date, not the Act's commencement.
    # Do not apply amended 2021 text to questions about earlier periods.
    effective_from="2021-08-31",
)


class PreviewPublisher:
    async def update_machine_verified_metadata(self, document_id, raw, **kwargs):
        return {"document_id": document_id, "preview_metadata": raw}


async def main(apply: bool) -> None:
    await mongodb.connect()
    await redis_client.connect()
    try:
        db = mongodb.db
        document = await db.uploaded_documents.find_one({"filename": POLICY.filename})
        if not document:
            raise RuntimeError("Expected document is missing.")
        if document.get("owner_user_id") or document.get("owner_session_id"):
            raise RuntimeError("Expected a shared KB document.")
        local_hash = hashlib.sha256(
            (Path(settings.knowledge_base_dir) / POLICY.filename).read_bytes()
        ).hexdigest()
        if document.get("document_hash") != local_hash:
            raise RuntimeError("Indexed document hash differs from the local PDF.")
        chunk_filter = {"metadata.source_document": POLICY.filename}
        rows = await db.embeddings_metadata.find(
            chunk_filter, {"metadata": 1}
        ).to_list(length=None)
        if len(rows) != 300 or any(
            row["metadata"].get("owner_user_id") or row["metadata"].get("owner_session_id")
            for row in rows
        ):
            raise RuntimeError("Expected exactly 300 shared chunks.")
        audit_dir = Path("storage/kb_audit") / datetime.now(UTC).strftime("cgst_approval_%Y%m%dT%H%M%S%fZ")
        audit_dir.mkdir(parents=True)
        (audit_dir / "before.json").write_text(
            json.dumps({"document": document, "chunks": rows}, default=str, indent=2),
            encoding="utf-8",
        )
        preview = await MachineVerificationService(publisher=PreviewPublisher()).verify(POLICY)
        if preview["status"] != "published":
            raise RuntimeError(f"Official-source verification failed: {preview.get('reason')}")
        (audit_dir / "preview.json").write_text(json.dumps(preview, default=str, indent=2), encoding="utf-8")
        if not apply:
            print(json.dumps({"status": "dry_run_passed", "chunks": len(rows), "audit": str(audit_dir)}))
            return
        if not await redis_client.ping():
            raise RuntimeError("Redis unavailable; cannot invalidate live caches.")
        result = await MachineVerificationService().verify(POLICY)
        (audit_dir / "result.json").write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        await db.kb_machine_verification_runs.insert_one({
            "ran_at": datetime.now(UTC), "requested_by": "workspace user: approve kr do",
            "scope": POLICY.filename, "results": [result], "backup": str(audit_dir / "before.json"),
        })
        approved = await db.embeddings_metadata.count_documents({
            **chunk_filter, "metadata.review_status": "approved",
            "metadata.verification_status": "machine_verified", "metadata.applicability": "all_india",
        })
        print(json.dumps({"result": result, "approved_chunks": approved, "audit": str(audit_dir)}, default=str))
        if result["status"] != "published" or approved != 300:
            raise RuntimeError("Publication did not complete; inspect saved audit report.")
    finally:
        await redis_client.close()
        await mongodb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    asyncio.run(main(parser.parse_args().apply))
