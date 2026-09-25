"""Audited onboarding for official catalogue adapters without code changes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from urllib.parse import urlsplit

from app.database.mongodb import mongodb
from app.models.collections import KB_SOURCE_ADAPTERS, OPERATIONAL_EVENTS
from app.schemas.law_monitoring import CatalogueSourceRequest
from app.services.kb_central_adapters import CatalogueConfig, OfficialCatalogueAdapter


def adapter_from_record(record: dict[str, Any], fetcher: Any = None) -> OfficialCatalogueAdapter:
    host = (urlsplit(record["url"]).hostname or "").lower()
    return OfficialCatalogueAdapter(CatalogueConfig(
        name=record["name"], authority=record["authority"], urls=(record["url"],),
        document_type=record["document_type"], include_pattern=record["include_pattern"],
        resolve_pdf_links=bool(record.get("resolve_pdf_links")),
        jurisdiction_code=record["jurisdiction_code"],
        applicable_state_codes=tuple(record.get("applicable_state_codes") or ()),
        expected_domain_suffixes=(host,),
    ), fetcher)


class SourceOnboardingService:
    def __init__(self, db: Any = None, fetcher: Any = None) -> None:
        self.db = db if db is not None else mongodb.db
        self.sources = self.db[KB_SOURCE_ADAPTERS]
        self.events = self.db[OPERATIONAL_EVENTS]
        self.fetcher = fetcher

    async def ensure_indexes(self) -> None:
        await self.sources.create_index("name", unique=True)
        await self.sources.create_index([("status", 1), ("jurisdiction_code", 1)])

    async def register(self, request: CatalogueSourceRequest, actor: str) -> dict[str, Any]:
        from app.services.kb_central_adapters import central_source_adapters
        from app.services.kb_high_court_adapters import high_court_source_adapters
        from app.services.kb_state_adapters import state_source_adapters
        reserved = {"official_manifest", *(
            adapter.name for adapter in [
                *central_source_adapters(), *state_source_adapters(), *high_court_source_adapters(),
            ]
        )}
        if request.name in reserved:
            raise ValueError("This adapter name is reserved by a built-in source.")
        now = datetime.now(UTC)
        record = request.model_dump(exclude={"authority_confirmed"})
        await self.sources.update_one(
            {"_id": request.name},
            {"$set": {**record, "status": "pending_probe", "updated_at": now,
                       "updated_by": actor},
             "$setOnInsert": {"created_at": now}}, upsert=True,
        )
        await self._audit("kb_source_registered", actor, request.name)
        saved = await self.sources.find_one({"_id": request.name})
        if saved is None:
            raise RuntimeError("Registered catalogue source could not be reloaded.")
        return cast(dict[str, Any], saved)

    async def probe(self, name: str, actor: str) -> dict[str, Any]:
        record = await self.sources.find_one({"_id": name})
        if record is None:
            raise ValueError("Catalogue source not found.")
        now = datetime.now(UTC)
        try:
            candidates, _ = await adapter_from_record(record, self.fetcher).discover(None)
            if not candidates:
                raise ValueError("Official catalogue returned no matching legal documents.")
            result = {"status": "probe_passed", "candidate_count": len(candidates)}
            await self.sources.update_one({"_id": name}, {"$set": {
                **result, "last_probe_at": now, "last_probe_error": None,
            }})
            await self._audit("kb_source_probe_passed", actor, name,
                              {"candidate_count": len(candidates)})
            return {"name": name, **result}
        except Exception as exc:  # noqa: BLE001 - every probe failure must be persisted for operations
            await self.sources.update_one({"_id": name}, {"$set": {
                "status": "probe_failed", "last_probe_at": now,
                "last_probe_error": type(exc).__name__,
            }})
            await self._audit("kb_source_probe_failed", actor, name,
                              {"error": type(exc).__name__})
            return {"name": name, "status": "probe_failed", "error": type(exc).__name__}

    async def activate(self, name: str, actor: str) -> dict[str, Any]:
        now = datetime.now(UTC)
        result = await self.sources.update_one(
            {"_id": name, "status": "probe_passed", "candidate_count": {"$gt": 0}},
            {"$set": {"status": "active", "activated_at": now, "activated_by": actor}},
        )
        if not result.modified_count:
            raise ValueError("Source must pass its latest live probe before activation.")
        await self._audit("kb_source_activated", actor, name)
        return {"name": name, "status": "active"}

    async def pause(self, name: str, actor: str) -> dict[str, Any]:
        result = await self.sources.update_one(
            {"_id": name, "status": "active"},
            {"$set": {"status": "paused", "paused_at": datetime.now(UTC),
                       "paused_by": actor}},
        )
        if not result.modified_count:
            raise ValueError("Only an active source can be paused.")
        await self._audit("kb_source_paused", actor, name)
        return {"name": name, "status": "paused"}

    async def list_sources(self) -> list[dict[str, Any]]:
        return [item async for item in self.sources.find({}).sort("jurisdiction_code", 1)]

    async def active_adapters(self) -> list[OfficialCatalogueAdapter]:
        return [adapter_from_record(item) async for item in self.sources.find({"status": "active"})]

    async def _audit(
        self, event_type: str, actor: str, source: str, details: dict[str, Any] | None = None,
    ) -> None:
        await self.events.insert_one({
            "event_type": event_type, "actor_user_id": actor, "source": source,
            "details": details or {}, "created_at": datetime.now(UTC),
        })
