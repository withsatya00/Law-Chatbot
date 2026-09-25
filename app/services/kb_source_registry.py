"""Phase-4 source rollout registry and production-readiness reporting.

The registry separates jurisdiction scope from a verified, enabled adapter.
That distinction prevents a dashboard from presenting all-India scope as
all-India corpus coverage.
"""

from __future__ import annotations

from typing import Any

from app.rag.kb_jurisdiction import STATE_UT_CODES

VERIFIED_SOURCES: dict[str, tuple[str, ...]] = {
    "UP": ("up_acts", "up_ordinances"),
    "DL": ("delhi_high_court_judgments", "delhi_law_dept_centralized_cos"),
    "MH": ("bombay_high_court_judgments", "mh_bombay_hc_acts_library"),
    # GA deliberately does NOT list an Acts adapter: `www.goa.gov.in/government/
    # acts-and-rules/` and `goaprintingpress.gov.in/state-acts/` were both
    # investigated 2026-09-22 and are genuinely blocked (see
    # `kb_state_adapters.py`'s module docstring) -- listing one here would be
    # untruthful, not a correction.
    "GA": ("bombay_high_court_judgments",),
    "DH": ("bombay_high_court_judgments", "ddd_acts_rules"),
    "AS": ("assam_acts",),
    "JK": ("jk_law_acts",),
    "JH": ("jharkhand_acts_rules_policies",),
    "ML": ("meghalaya_acts",),
    "PY": ("py_law_publications",),
    "TR": ("tripura_hc_acts_library",),
    "UT": ("uttarakhand_hc_acts",),
    "MN": ("manipur_assembly_acts",),
    "NL": ("nagaland_act_rules",),
    "LA": ("ladakh_acts_rules",),
    "OR": ("odisha_law_dept_acts",),
    "CT": ("chhattisgarh_law_acts",),
    "MP": ("mp_code_state_acts",),
}


def rollout_registry() -> list[dict[str, Any]]:
    """Return every State/UT with a truthful onboarding state."""
    return [
        {
            "code": code,
            "name": name,
            "source_status": "verified" if code in VERIFIED_SOURCES else "verification_required",
            "adapters": list(VERIFIED_SOURCES.get(code, ())),
            "manual_download_expected": False,
        }
        for code, name in STATE_UT_CODES.items()
    ]


def rollout_readiness(
    coverage: dict[str, Any], automation: dict[str, Any],
    onboarded_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply fail-closed Phase-4 launch gates to live reports."""
    registry = rollout_registry()
    by_code = {row["code"]: row for row in registry}
    for source in onboarded_sources or []:
        if source.get("status") != "active":
            continue
        codes = {source.get("jurisdiction_code"), *(source.get("applicable_state_codes") or [])}
        for code in codes:
            if code not in by_code:
                continue
            row = by_code[code]
            row["source_status"] = "verified"
            if source["name"] not in row["adapters"]:
                row["adapters"].append(source["name"])
    adapter_health = {row["adapter"]: row["status"] for row in automation.get("adapters", [])}
    coverage_by_code = {row["code"]: row for row in coverage.get("jurisdictions", [])}
    rows: list[dict[str, Any]] = []
    for source in registry:
        code = source["code"]
        states = [adapter_health.get(name, "never_run") for name in source["adapters"]]
        corpus = coverage_by_code.get(code, {})
        gates = {
            "official_source_verified": source["source_status"] == "verified",
            "adapter_healthy": bool(states) and all(state == "healthy" for state in states),
            "approved_corpus_present": int(corpus.get("approved_chunks", 0)) > 0,
            "retrieval_benchmark_passed": int(corpus.get("tested_passed", 0)) > 0,
            "no_manual_access_exception": int(corpus.get("manual_access_required", 0)) == 0,
        }
        rows.append({**source, "adapter_health": states, "gates": gates,
                     "ready": all(gates.values())})
    ready = sum(row["ready"] for row in rows)
    return {
        "phase": 4,
        "mode": "controlled_rollout",
        "production_ready": ready == len(rows),
        "ready_jurisdictions": ready,
        "total_jurisdictions": len(rows),
        "manual_downloads_required": int(automation.get("routine_manual_downloads_required", 0)),
        "jurisdictions": rows,
    }
