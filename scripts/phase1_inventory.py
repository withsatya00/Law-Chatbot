"""Read-only file inventory; static references are evidence, never deletion authority.

Run: python -m scripts.phase1_inventory --output output/phase1-inventory.json
Private/runtime files are counted without reading or listing their contents.
"""
import argparse
import ast
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", ".venv.broken-satya", "__pycache__", "node_modules"}
PRIVATE = {"storage", "output", "tmp"}


def inventory():
    totals = Counter()
    sizes = Counter()
    modules = []
    imports = Counter()
    unreadable = []
    for directory, children, filenames in os.walk(ROOT, onerror=lambda exc: unreadable.append(type(exc).__name__)):
        children[:] = [name for name in children if name not in EXCLUDED and not name.startswith(".pytest")
                       and name not in {".mypy_cache", ".ruff_cache"}]
        for name in filenames:
            path = Path(directory) / name
            relative = path.relative_to(ROOT)
            top = relative.parts[0]
            category = "private_or_runtime" if top in PRIVATE else (
                "secrets_configuration" if name.startswith(".env") else top if len(relative.parts) > 1 else "root_configuration")
            try:
                totals[category] += 1
                sizes[category] += path.stat().st_size
                if top in PRIVATE or name.startswith(".env") or path.suffix != ".py":
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8-sig"))
                modules.append(relative.as_posix())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imports[node.module] += 1
                        imports.update(f"{node.module}.{alias.name}" for alias in node.names)
            except (OSError, SyntaxError, UnicodeError) as exc:
                unreadable.append(f"{relative.as_posix()}: {type(exc).__name__}")
    return {
        "scope": "Filesystem and Python AST only; no live DB, secrets or private document contents read",
        "counts": dict(totals), "bytes": dict(sizes), "unreadable": unreadable,
        "python_modules": [{"path": path, "static_import_references": imports[path[:-3].replace("/", ".")]}
                           for path in sorted(modules)],
        "deletion_rule": "Zero static imports does NOT prove unused: check routes, entrypoints, dynamic imports, templates and deployment references.",
        "excluded": sorted(EXCLUDED | {".pytest*", ".mypy_cache", ".ruff_cache"}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "phase1-inventory.json")
    args = parser.parse_args()
    result = inventory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"counts": result["counts"], "unreadable_count": len(result["unreadable"])}))


if __name__ == "__main__":
    main()
