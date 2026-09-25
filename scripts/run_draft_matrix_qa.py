"""Run all-template x all-language deterministic draft QA.

Usage:
    python scripts/run_draft_matrix_qa.py
    python scripts/run_draft_matrix_qa.py --write-drafts
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.drafting.engine import LegalDraftEngine
from app.drafting.matrix_qa import run_export_smoke, run_matrix_sync, write_draft_artifacts


def _markdown(results: list, generated_at: str, export_results: list | None = None) -> str:
    counts = Counter(result.status for result in results)
    support_counts = Counter(result.support_level for result in results)
    failures = [result for result in results if result.status != "PASS"]
    lines = [
        "# Draft matrix QA",
        "",
        f"Generated: {generated_at}",
        f"Cases: {len(results)}",
        f"PASS: {counts['PASS']} | FAIL: {counts['FAIL']} | ERROR: {counts['ERROR']}",
        (
            f"Fully-localized cases: {support_counts['full']} | "
            f"Body-only localization cases: {support_counts['body_only']}"
        ),
        "",
        (
            "> PASS means generation/grammar/Unicode structure passed. Body-only languages intentionally retain "
            "English structural chrome and are not claimed as native-language complete."
        ),
        "",
        "## Failures",
        "",
    ]
    if not failures:
        lines.append("None.")
    else:
        lines.extend(
            f"- `{item.template_id}` / `{item.language}`: {', '.join(item.issues)}"
            for item in failures
        )
    if export_results is not None:
        export_counts = Counter(result.status for result in export_results)
        lines.extend([
            "", "## Export smoke", "",
            (
                f"Cases: {len(export_results)} | PASS: {export_counts['PASS']} | "
                f"WARNING: {export_counts['WARNING']} | FAIL: {export_counts['FAIL']} | "
                f"ERROR: {export_counts['ERROR']} | "
                f"SKIP: {export_counts['SKIP']}"
            ),
            "",
        ])
        export_failures = [result for result in export_results if result.status in {"FAIL", "ERROR"}]
        if not export_failures:
            lines.append("No unexpected export failures.")
        else:
            lines.extend(
                f"- `{item.template_id}` / `{item.language}` / `{item.format}`: {', '.join(item.issues)}"
                for item in export_failures
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/draft-matrix/latest"))
    parser.add_argument("--write-drafts", action="store_true")
    parser.add_argument("--export-smoke", action="store_true")
    parser.add_argument("--full-visual-matrix", action="store_true")
    args = parser.parse_args()

    generated_at = datetime.now(UTC).isoformat()
    results = run_matrix_sync(LegalDraftEngine())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_results = (
        run_export_smoke(results, args.output_dir, full_visual_matrix=args.full_visual_matrix)
        if args.export_smoke or args.full_visual_matrix else None
    )
    payload = {
        "generated_at": generated_at,
        "case_count": len(results),
        "summary": dict(Counter(result.status for result in results)),
        "support_summary": dict(Counter(result.support_level for result in results)),
        "results": [result.as_dict(include_text=False) for result in results],
    }
    if export_results is not None:
        payload["export_summary"] = dict(Counter(result.status for result in export_results))
        payload["export_results"] = [result.as_dict() for result in export_results]
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        _markdown(results, generated_at, export_results), encoding="utf-8"
    )
    if args.write_drafts:
        write_draft_artifacts(results, args.output_dir)

    failures = sum(result.status != "PASS" for result in results)
    if export_results is not None:
        failures += sum(result.status in {"FAIL", "ERROR"} for result in export_results)
    print(f"Draft matrix: {len(results)} cases, {failures} failures")
    print(f"Report: {(args.output_dir / 'report.md').resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
