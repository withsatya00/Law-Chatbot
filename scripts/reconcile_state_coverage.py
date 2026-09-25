"""Monthly State/UT coverage reconciliation. Read-only -- reports drift, never
mutates anything (approval, verification and re-indexing stay admin actions
through the existing `/admin/phase3/*` endpoints).

Usage:
    .venv\\Scripts\\python.exe scripts\\reconcile_state_coverage.py
    .venv\\Scripts\\python.exe scripts\\reconcile_state_coverage.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.database.mongodb import mongodb
from app.services.kb_coverage import KnowledgeBaseCoverageService


async def run() -> dict[str, Any]:
    await mongodb.connect()
    try:
        report = await KnowledgeBaseCoverageService().reconciliation_report()
    finally:
        await mongodb.close()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = settings.operations_output_dir / "state_coverage_reconciliation"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"reconciliation_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args()

    report = asyncio.run(run())
    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    print(f"generated_at: {report['generated_at']}")
    print(f"complete: {report['complete']}")
    print(f"manual_access_exceptions: {report['manual_access_exceptions']}")
    print(f"report_path: {report['report_path']}")
    if report["stuck_jurisdictions"]:
        print("\nstuck_jurisdictions:")
        for item in report["stuck_jurisdictions"]:
            print(f"  {item['code']} ({item['name']}): {item['stuck_at']}")
    else:
        print("\nNo stuck jurisdictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
