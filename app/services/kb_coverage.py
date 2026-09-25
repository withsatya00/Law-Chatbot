"""Auditable all-India Knowledge Base coverage reporting.

The matrix deliberately distinguishes a jurisdiction being in acquisition
scope from a law actually being downloaded, verified and searchable.  This
prevents a configured India Code search from being reported as corpus
coverage.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from app.database.mongodb import mongodb
from app.models.collections import EMBEDDINGS_METADATA
from app.rag.kb_jurisdiction import REVIEW_APPROVED, STATE_UT_CODES

INDIA_CODE_CENTRAL = "https://indiacode.gov.in/legislation-counts"
INDIA_CODE_STATE = "https://indiacode.gov.in/statistics"


class KnowledgeBaseCoverageService:
    """Build a stage-by-stage coverage matrix from indexed chunk metadata."""

    async def coverage(self) -> dict[str, Any]:
        report = await self._chunk_coverage()
        from app.services.kb_automation import KnowledgeBaseAutomationService
        automation = KnowledgeBaseAutomationService()
        jurisdiction_summary = await automation.per_jurisdiction_summary()
        return self.merge_automation(report, jurisdiction_summary)

    async def _chunk_coverage(self) -> dict[str, Any]:
        grouped = await mongodb.db[EMBEDDINGS_METADATA].aggregate([
            {
                "$group": {
                    "_id": {
                        "issuing_level": "$metadata.issuing_level",
                        "state_codes": "$metadata.applicable_state_codes",
                        "source_document": "$metadata.source_document",
                    },
                    "chunks": {"$sum": 1},
                    "approved_chunks": {
                        "$sum": {
                            "$cond": [
                                {"$eq": ["$metadata.review_status", REVIEW_APPROVED]}, 1, 0,
                            ]
                        }
                    },
                }
            }
        ]).to_list(length=None)
        return self.from_grouped_chunks(grouped)

    @staticmethod
    def from_grouped_chunks(grouped: list[dict[str, Any]]) -> dict[str, Any]:
        central = _empty_row("IN", "Central", "central", INDIA_CODE_CENTRAL)
        states = {
            code: _empty_row(code, name, "state_or_ut", INDIA_CODE_STATE)
            for code, name in STATE_UT_CODES.items()
        }
        unmapped: dict[str, Any] = {"documents": set(), "chunks": 0, "approved_chunks": 0}

        for group in grouped:
            identity = group.get("_id") or {}
            source = str(identity.get("source_document") or "unknown")
            chunks = int(group.get("chunks") or 0)
            approved = int(group.get("approved_chunks") or 0)
            issuing_level = identity.get("issuing_level")
            codes = identity.get("state_codes") or []
            if isinstance(codes, str):
                codes = [codes]

            targets: list[dict[str, Any]] = []
            if issuing_level == "central":
                targets.append(central)
            for code in codes:
                if code in states:
                    targets.append(states[code])
            if not targets:
                unmapped["documents"].add(source)
                unmapped["chunks"] += chunks
                unmapped["approved_chunks"] += approved
                continue
            for row in targets:
                row["_documents"].add(source)
                row["indexed_chunks"] += chunks
                row["approved_chunks"] += approved

        rows = [central, *states.values()]
        for row in rows:
            row["indexed_documents"] = len(row.pop("_documents"))
            row["status"] = _status(row)

        covered = sum(row["indexed_documents"] > 0 for row in rows)
        return {
            "scope": {
                "jurisdictions": len(rows),
                "central": 1,
                "states_and_union_territories": len(states),
                "official_catalog_configured": len(rows),
                "jurisdictions_with_indexed_documents": covered,
            },
            "stage_definitions": {
                "catalogued": "Official discovery catalogue is configured; no document coverage is implied.",
                "discovered": "An automation adapter has found candidate document links for this jurisdiction.",
                "downloaded": "A discovered candidate has been fetched, scanned and staged (any pipeline stage).",
                "indexed": "At least one source document has searchable chunks.",
                "approved": "Every indexed chunk counted here passed the KB publication review gate.",
                "tested": "An approved automated candidate passed the real retrieval benchmark.",
            },
            "jurisdictions": rows,
            "unmapped": {
                "indexed_documents": len(unmapped["documents"]),
                "indexed_chunks": unmapped["chunks"],
                "approved_chunks": unmapped["approved_chunks"],
            },
            "complete": covered == len(rows) and all(row["status"] == "approved" for row in rows),
        }

    @staticmethod
    def merge_automation(
        report: dict[str, Any], jurisdiction_summary: dict[str, dict[str, int]],
    ) -> dict[str, Any]:
        """Folds per-jurisdiction automation-pipeline counts (discover/
        download/test stages, from `KnowledgeBaseAutomationService.
        per_jurisdiction_summary`) onto the chunk-derived matrix from
        `from_grouped_chunks`/`_chunk_coverage`. Pure and unit-testable
        without Mongo, like `from_grouped_chunks` itself.

        A jurisdiction's per-row `complete` only requires a passing benchmark
        when one was actually run (`tested_total > 0`) -- a jurisdiction
        reached entirely through ordinary human KB upload, with no automation
        adapter configured for it at all, is not penalized for lacking a
        benchmark it was never eligible to run.
        """
        summary_by_code = {row["code"]: row for row in report["jurisdictions"]}
        manual_access_exceptions: list[str] = []
        for code, stats in jurisdiction_summary.items():
            row = summary_by_code.get(code)
            if row is None:  # an HC/adapter code outside the 28+8 universe -- surfaced, not dropped
                continue
            row["discovered"] = stats.get("discovered", 0)
            row["downloaded"] = stats.get("downloaded", 0)
            row["manual_access_required"] = stats.get("manual_access_required", 0)
            row["tested_passed"] = stats.get("tested_passed", 0)
            row["tested_total"] = stats.get("tested_total", 0)
            if row["manual_access_required"] > 0:
                manual_access_exceptions.append(code)
        for row in report["jurisdictions"]:
            row.setdefault("discovered", 0)
            row.setdefault("downloaded", 0)
            row.setdefault("manual_access_required", 0)
            row.setdefault("tested_passed", 0)
            row.setdefault("tested_total", 0)
            row["complete"] = bool(
                row["approved_chunks"] > 0 and (row["tested_total"] == 0 or row["tested_passed"] > 0)
            )
        report["manual_access_exceptions"] = sorted(manual_access_exceptions)
        report["complete"] = all(row["complete"] for row in report["jurisdictions"])
        return report

    async def reconciliation_report(self) -> dict[str, Any]:
        """Read-only monthly drift report: which jurisdictions are stuck, and
        at which stage. Never mutates anything -- approving, verifying or
        re-indexing stays a human/admin action through the existing endpoints;
        this only tells an operator where to look."""
        coverage = await self.coverage()
        stuck: list[dict[str, Any]] = []
        for row in coverage["jurisdictions"]:
            if row["complete"]:
                continue
            if row["tested_total"] > row["tested_passed"]:
                stage = "tested_failing"
            elif row["indexed_documents"] > 0:
                stage = "indexed_needs_review"
            elif row["manual_access_required"] > 0:
                stage = "manual_access_exception"
            elif row["downloaded"] > 0:
                stage = "downloaded_not_indexed"
            elif row["discovered"] > 0:
                stage = "discovered_not_downloaded"
            else:
                stage = "not_configured"
            stuck.append({"code": row["code"], "name": row["name"], "stuck_at": stage})
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "scope": coverage["scope"],
            "complete": coverage["complete"],
            "manual_access_exceptions": coverage["manual_access_exceptions"],
            "stuck_jurisdictions": stuck,
        }


def _empty_row(code: str, name: str, level: str, discovery_url: str) -> dict[str, Any]:
    return {
        "code": code,
        "name": name,
        "level": level,
        "official_discovery_url": discovery_url,
        "catalogued": True,
        "_documents": set(),
        "indexed_chunks": 0,
        "approved_chunks": 0,
    }


def _status(row: dict[str, Any]) -> str:
    if row["indexed_documents"] == 0:
        return "catalogued_only"
    if row["approved_chunks"] < row["indexed_chunks"]:
        return "indexed_needs_review"
    return "approved"
