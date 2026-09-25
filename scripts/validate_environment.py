"""Feature-level environment validation, without starting the API.

Answers one question per feature: *can this host actually do it?* -- and, when
it cannot, names the exact thing to install. Every check probes the real
capability rather than inferring it:

  * a Python package is imported, not just found on disk;
  * Tesseract and Poppler are located as EXECUTABLES and asked for a version;
  * WeasyPrint's GTK/Pango/Cairo native libraries are exercised by rendering a
    one-line PDF, because `import weasyprint` succeeds on a host whose native
    stack is broken and fails only at render time;
  * MongoDB and Redis are pinged;
  * the configured LLM provider is constructed and health-checked;
  * every storage directory is written to and read back.

Secrets are never printed. API keys are reported only as `configured` /
`missing`, connection URIs only as scheme + host + port with any embedded
credentials dropped -- see `_safe_endpoint`. Nothing in the output can be
pasted into a bug report and leak a password.

Exit codes:
    0  every REQUIRED check passed (optional features may be unavailable)
    1  at least one required check failed
    2  the validator itself could not run (bad configuration)

Usage:
    .venv\\Scripts\\python.exe scripts\\validate_environment.py
    .venv\\Scripts\\python.exe scripts\\validate_environment.py --json
    .venv\\Scripts\\python.exe scripts\\validate_environment.py --strict
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import Settings
from app.core.optional_deps import _OPTIONAL
from app.core.windows_runtime import repair_windows_host_env

OK = "ok"
DEGRADED = "degraded"
FAILED = "failed"


@dataclass
class Check:
    """One feature-level verdict.

    `required` distinguishes "this host cannot serve traffic" from "this host
    cannot do OCR" -- both are worth reporting, only the first should fail a
    deployment gate.
    """

    name: str
    status: str
    detail: str
    required: bool = True
    fix: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "required": self.required,
            "detail": self.detail,
        }
        if self.fix:
            payload["fix"] = self.fix
        if self.extra:
            payload["extra"] = self.extra
        return payload


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def _safe_endpoint(uri: str) -> str:
    """`mongodb://user:pass@host:27017/db` -> `mongodb://host:27017`.

    Connection strings are the single most common way a credential ends up in
    a pasted diagnostic. Userinfo, path and query are all dropped rather than
    masked, so there is nothing left to reconstruct.
    """
    try:
        parts = urlsplit(uri)
    except ValueError:
        return "<unparseable>"
    host = parts.hostname or "?"
    port = f":{parts.port}" if parts.port else ""
    scheme = parts.scheme or "?"
    return f"{scheme}://{host}{port}"


def _key_state(value: str) -> str:
    """Never the key. Only whether one is present."""
    return "configured" if value.strip() else "missing"


# ---------------------------------------------------------------------------
# Python packages
# ---------------------------------------------------------------------------

# Packages without which the API cannot serve a request at all.
_REQUIRED_PACKAGES: dict[str, str] = {
    "fastapi": "pip install -e .",
    "uvicorn": "pip install -e .",
    "pydantic": "pip install -e .",
    "pydantic_settings": "pip install -e .",
    "motor": "pip install -e .",
    "pymongo": "pip install -e .",
    "redis": "pip install -e .",
    "httpx": "pip install -e .",
    "structlog": "pip install -e .",
    "orjson": "pip install -e .",
    "jose": 'pip install "python-jose[cryptography]"',
    "passlib": 'pip install "passlib[bcrypt]"',
    "bcrypt": 'pip install "bcrypt<4.1"',
    "tenacity": "pip install -e .",
    "numpy": "pip install -e .",
    "sentence_transformers": "pip install -e .",
    "rank_bm25": "pip install rank-bm25",
    "yaml": "pip install pyyaml",
    "tzdata": "pip install tzdata",
}




def _repair_windows_host_env() -> list[str]:
    """Apply the same host repairs as `scripts/_win_env.ps1`, in-process.

    A diagnostic has to run from whatever shell the operator actually opened.
    Run from a plain PowerShell -- without the launcher that dot-sources
    `_win_env.ps1` -- this script previously died with

        OPENSSL_Uplink(0x...,08): no OPENSSL_Applink

    before printing a single line (Avast injects `SSLKEYLOGFILE` as a device
    path OpenSSL cannot open; see WINDOWS_SETUP.md section 2a), and reported
    Poppler as missing because its winget directory is deliberately kept off
    the global PATH (section 4). Neither is a real environment fault, and both
    told the operator the wrong thing.

    Returns the human-readable list of repairs applied, so the report states
    what it changed rather than silently papering over the host.
    """
    # The implementation lives in `app/core/windows_runtime.py` so the API and
    # pytest get the same repair -- fixing it only here left every other
    # process still aborting.
    return repair_windows_host_env()


def _register_native_library_paths() -> None:
    """Import `app.drafting.export` before any WeasyPrint probe.

    On Windows, that module registers the GTK3 bin directory with
    `os.add_dll_directory()` at import time -- Python 3.8+ no longer searches
    PATH when resolving a loaded DLL's own transitive dependencies, so
    `import weasyprint` fails with `error 0x7e` until it has run. Probing
    WeasyPrint without it reports PDF export as unavailable on a host where it
    works perfectly, which is a worse answer than no answer: it sends an
    operator to reinstall a stack that is already correct.

    The validator must therefore load the app's own native-library setup
    first, exactly as the API does at startup.
    """
    try:
        importlib.import_module("app.drafting.export")
    except Exception:  # noqa: BLE001 - the individual checks below report the real reason
        return


def _check_required_packages() -> list[Check]:
    checks: list[Check] = []
    missing: list[str] = []
    for module, hint in _REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report ANY import failure, not just ImportError
            missing.append(f"{module} ({type(exc).__name__}: {exc})")
            checks.append(
                Check(f"package:{module}", FAILED, f"import failed: {exc}", required=True, fix=hint)
            )
    if not missing:
        checks.append(
            Check(
                "python_packages",
                OK,
                f"all {len(_REQUIRED_PACKAGES)} required packages import cleanly",
            )
        )
    return checks


def _check_optional_packages() -> list[Check]:
    """The format/feature dependencies `app.core.optional_deps` gates on.

    Reported per feature, because losing one of these costs exactly one
    capability -- PDF upload, OCR, DOCX export -- and never the whole app.
    """
    checks: list[Check] = []
    for module, (feature, hint) in sorted(_OPTIONAL.items()):
        try:
            importlib.import_module(module)
        except (ImportError, OSError) as exc:
            checks.append(Check(f"feature:{feature}", DEGRADED, str(exc), required=False, fix=hint))
        else:
            checks.append(Check(f"feature:{feature}", OK, f"{module} imports cleanly", required=False))
    return checks


# ---------------------------------------------------------------------------
# Native binaries
# ---------------------------------------------------------------------------


def _run_version(executable: str, *args: str) -> tuple[bool, str]:
    path = shutil.which(executable)
    if path is None:
        return False, f"'{executable}' is not on PATH"
    try:
        # Fixed argv, no shell, and `path` came from `shutil.which` -- no
        # user-controlled input reaches this call.
        completed = subprocess.run(
            [path, *args], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{path}: {exc}"
    output = (completed.stdout or completed.stderr or "").strip().splitlines()
    return True, (output[0] if output else path)


def _check_tesseract() -> Check:
    found, detail = _run_version("tesseract", "--version")
    if not found:
        return Check(
            "binary:tesseract",
            DEGRADED,
            detail,
            required=False,
            fix="Install Tesseract-OCR and put its directory on PATH "
            "(Windows: see WINDOWS_SETUP.md section 4; Debian: apt install tesseract-ocr)",
        )
    return Check("binary:tesseract", OK, detail, required=False)


def _check_poppler() -> Check:
    """`pdf2image` shells out to pdftoppm/pdfinfo; a missing Poppler makes
    scanned-PDF OCR return empty text rather than raise, so it is checked as a
    binary rather than inferred from the Python package."""
    found, detail = _run_version("pdftoppm", "-v")
    if not found:
        return Check(
            "binary:poppler",
            DEGRADED,
            detail,
            required=False,
            fix="Install Poppler and put its bin directory on PATH "
            "(Windows: winget install oschwartz10612.Poppler; Debian: apt install poppler-utils)",
        )
    return Check("binary:poppler", OK, detail, required=False)


def _check_weasyprint_native() -> Check:
    """Renders a real one-line PDF.

    `import weasyprint` succeeds on a host with a broken or mismatched
    GTK/Pango/Cairo stack; the failure only appears at render time, as an
    OSError from the loader (on Windows, `error 0x7f` when Tesseract's mingw64
    GTK copies are picked up alongside MSYS2's -- see `scripts/_win_env.ps1`).
    Only an actual render distinguishes the two.
    """
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as exc:
        return Check(
            "feature:PDF export (WeasyPrint native libraries)",
            DEGRADED,
            f"import failed: {exc}",
            required=False,
            fix="pip install weasyprint, plus the GTK3/Pango/Cairo native libraries "
            "(Windows: MSYS2 ucrt64, see WINDOWS_SETUP.md section 3; Debian: "
            "apt install libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0)",
        )
    try:
        pdf = HTML(string="<p>environment validation</p>").write_pdf()
    except Exception as exc:  # noqa: BLE001 - the native stack can fail in many ways
        return Check(
            "feature:PDF export (WeasyPrint native libraries)",
            DEGRADED,
            f"render failed: {type(exc).__name__}: {exc}",
            required=False,
            fix="The GTK/Pango/Cairo native libraries are missing or mismatched. "
            "On Windows this is usually PATH order -- run through scripts\\run_api.ps1, "
            "which applies scripts\\_win_env.ps1.",
        )
    if not pdf or not pdf.startswith(b"%PDF-"):
        return Check(
            "feature:PDF export (WeasyPrint native libraries)",
            DEGRADED,
            "renderer produced no PDF header",
            required=False,
        )
    return Check(
        "feature:PDF export (WeasyPrint native libraries)",
        OK,
        f"rendered a {len(pdf)}-byte PDF",
        required=False,
    )


def _check_qr_generation() -> Check:
    """The notarization verification QR. Generated for a notarized status only,
    so the probe uses that status deliberately -- see `app/notarization/qr.py`.
    """
    try:
        from app.notarization import states
        from app.notarization.qr import build_verification_qr_svg
    except Exception as exc:  # noqa: BLE001 - an import failure here is still "QR unavailable"
        return Check("feature:notarization QR", FAILED, f"import failed: {exc}", required=True)
    try:
        svg = build_verification_qr_svg(states.NOTARIZED, "https://example.invalid/verify/probe")
    except Exception as exc:  # noqa: BLE001
        return Check(
            "feature:notarization QR",
            FAILED,
            f"{type(exc).__name__}: {exc}",
            required=True,
            fix="pip install segno",
        )
    if not svg.startswith("<svg"):
        return Check("feature:notarization QR", FAILED, "generator returned non-SVG output")
    return Check("feature:notarization QR", OK, f"generated a {len(svg)}-byte SVG")


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


async def _check_mongodb(settings: Settings) -> Check:
    endpoint = _safe_endpoint(settings.mongodb_uri)
    try:
        from motor.motor_asyncio import AsyncIOMotorClient
    except ImportError as exc:
        return Check("service:mongodb", FAILED, f"motor is not installed: {exc}", fix="pip install -e .")
    client = AsyncIOMotorClient(settings.mongodb_uri, serverSelectionTimeoutMS=4000)
    try:
        await client.admin.command("ping")
    except Exception as exc:  # noqa: BLE001 - pymongo raises a wide family for an unreachable server
        return Check(
            "service:mongodb",
            FAILED,
            f"{endpoint} unreachable: {type(exc).__name__}",
            fix="Start MongoDB, or correct MONGODB_URI in .env",
            extra={"endpoint": endpoint, "database": settings.mongodb_database},
        )
    finally:
        client.close()
    return Check(
        "service:mongodb", OK, f"{endpoint} reachable",
        extra={"endpoint": endpoint, "database": settings.mongodb_database},
    )


async def _check_redis(settings: Settings) -> Check:
    endpoint = _safe_endpoint(settings.redis_url)
    try:
        from redis.asyncio import Redis
    except ImportError as exc:
        return Check("service:redis", FAILED, f"redis is not installed: {exc}", fix="pip install -e .")
    client = Redis.from_url(settings.redis_url, socket_connect_timeout=4)
    try:
        await client.ping()
    except Exception as exc:  # noqa: BLE001 - redis-py and the socket layer both raise here
        return Check(
            "service:redis",
            FAILED,
            f"{endpoint} unreachable: {type(exc).__name__}",
            fix="Start Redis/Memurai, or correct REDIS_URL in .env",
            extra={"endpoint": endpoint},
        )
    finally:
        await client.aclose()
    return Check("service:redis", OK, f"{endpoint} reachable", extra={"endpoint": endpoint})


async def _check_llm_provider(settings: Settings) -> Check:
    """Constructs the CONFIGURED provider and health-checks it.

    Reports the provider name and whether a key is configured -- never the key.
    """
    provider_name = (settings.llm_provider or "").lower()
    key_states = {
        "groq": _key_state(settings.groq_api_key),
        "openai": _key_state(settings.openai_api_key),
        "gemini": _key_state(settings.gemini_api_key),
        "claude": _key_state(settings.claude_api_key),
        "deepseek": _key_state(settings.deepseek_api_key),
        "openrouter": _key_state(settings.openrouter_api_key),
        "ollama": "not required",
    }
    extra: dict[str, Any] = {
        "provider": provider_name,
        "credential": key_states.get(provider_name, "unknown"),
        "fallbacks": [name.strip() for name in settings.llm_fallback_providers.split(",") if name.strip()],
    }
    if provider_name == "ollama":
        extra["endpoint"] = _safe_endpoint(settings.ollama_base_url)

    try:
        from app.llm.factory import LLMFactory

        provider = LLMFactory.create(provider_name)
    except Exception as exc:  # noqa: BLE001 - an unknown provider name raises ValueError; imports can raise anything
        return Check(
            "service:llm",
            FAILED,
            f"cannot construct provider '{provider_name}': {type(exc).__name__}: {exc}",
            fix="Set LLM_PROVIDER in .env to one of: groq, openai, ollama, deepseek, openrouter, gemini, claude",
            extra=extra,
        )
    try:
        healthy = await provider.health()
    except Exception as exc:  # noqa: BLE001 - provider health checks reach the network
        return Check(
            "service:llm",
            FAILED,
            f"health check raised {type(exc).__name__}",
            fix="Check the provider's credentials and network reachability.",
            extra=extra,
        )
    if not healthy:
        return Check(
            "service:llm",
            FAILED,
            f"provider '{provider_name}' reported unhealthy",
            fix=(
                "Start Ollama and pull the configured model"
                if provider_name == "ollama"
                else f"Set the API key for '{provider_name}' in .env"
            ),
            extra=extra,
        )
    return Check("service:llm", OK, f"provider '{provider_name}' is healthy", extra=extra)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _check_storage(settings: Settings) -> list[Check]:
    """Each directory is created, WRITTEN to, read back and cleaned up.

    `mkdir` succeeding proves nothing on a read-only mount or a path whose ACL
    denies writes to the service account -- both of which fail later, on the
    first upload, as a 500 in the middle of a user's request.
    """
    directories = {
        "uploads": settings.upload_storage_dir,
        "knowledge_base": settings.knowledge_base_dir,
        "drafts": settings.draft_output_dir,
        "backups": settings.backup_dir,
    }
    checks: list[Check] = []
    for label, directory in directories.items():
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            checks.append(
                Check(f"storage:{label}", FAILED, f"cannot create {directory}: {exc}",
                      fix="Create the directory or point the setting at a writable path")
            )
            continue
        probe = directory / f".envcheck-{os.getpid()}"
        try:
            probe.write_text("ok", encoding="utf-8")
            readback = probe.read_text(encoding="utf-8")
        except OSError as exc:
            checks.append(
                Check(f"storage:{label}", FAILED, f"{directory} is not writable: {exc}",
                      fix="Grant write permission to the account running the API")
            )
            continue
        finally:
            probe.unlink(missing_ok=True)
        if readback != "ok":
            checks.append(Check(f"storage:{label}", FAILED, f"{directory} read-back mismatch"))
            continue
        checks.append(Check(f"storage:{label}", OK, f"{directory.resolve()} is writable"))
    return checks


def _check_pdf_and_docx_export() -> list[Check]:
    """Produces a real file with each exporter, into a temp directory.

    This is the closest thing to "can a user download their draft?" that can
    be answered without a database.
    """
    from app.drafting.export import (
        DocxDraftExporter,
        DraftExporter,
        PdfDraftExporter,
        RtfDraftExporter,
        TxtDraftExporter,
    )

    exporters: dict[str, tuple[DraftExporter, bool]] = {
        # (exporter, required) -- TXT and RTF are pure Python and must work
        # anywhere; PDF and DOCX depend on optional native/third-party stacks.
        "txt": (TxtDraftExporter(), True),
        "rtf": (RtfDraftExporter(), True),
        "docx": (DocxDraftExporter(), False),
        "pdf": (PdfDraftExporter(), False),
    }
    sections = {"Subject": "Environment validation", "Body": "This file is a generated probe."}
    checks: list[Check] = []
    with tempfile.TemporaryDirectory(prefix="legalai-envcheck-") as tmp:
        for fmt, (exporter, required) in exporters.items():
            target = Path(tmp) / f"probe.{fmt}"
            try:
                written = exporter.export("Validation Probe", sections, target, "english", None)
            except Exception as exc:  # noqa: BLE001 - each exporter has its own failure family
                checks.append(
                    Check(
                        f"export:{fmt}",
                        FAILED if required else DEGRADED,
                        f"{type(exc).__name__}: {exc}",
                        required=required,
                        fix=_OPTIONAL.get("weasyprint", ("", ""))[1] if fmt == "pdf" else
                            _OPTIONAL.get("docx", ("", ""))[1] if fmt == "docx" else "",
                    )
                )
                continue
            size = Path(written).stat().st_size
            checks.append(
                Check(f"export:{fmt}", OK, f"wrote {size} bytes", required=required)
            )
    return checks


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_checks() -> list[Check]:
    repairs = _repair_windows_host_env()
    settings = Settings()
    _register_native_library_paths()
    checks: list[Check] = []
    if repairs:
        checks.append(
            Check("host_repairs", OK, "; ".join(repairs), required=False, extra={"applied": repairs})
        )
    checks.append(
        Check(
            "configuration",
            OK,
            f"environment={settings.environment}",
            extra={"environment": settings.environment, "app_name": settings.app_name},
        )
    )
    checks.extend(_check_required_packages())
    checks.extend(_check_optional_packages())
    checks.append(_check_tesseract())
    checks.append(_check_poppler())
    checks.append(_check_weasyprint_native())
    checks.append(_check_qr_generation())
    checks.extend(_check_storage(settings))
    checks.extend(_check_pdf_and_docx_export())
    checks.append(await _check_mongodb(settings))
    checks.append(await _check_redis(settings))
    checks.append(await _check_llm_provider(settings))
    return checks


def validate() -> dict[str, Any]:
    """Programmatic entry point. Returns the full report as plain data."""
    checks = asyncio.run(run_checks())
    failures = [check for check in checks if check.status == FAILED and check.required]
    degraded = [check for check in checks if check.status == DEGRADED]
    return {
        "ok": not failures,
        "failed": [check.name for check in failures],
        "degraded": [check.name for check in degraded],
        "checks": [check.as_dict() for check in checks],
    }


_ICONS = {OK: "[ ok ]", DEGRADED: "[warn]", FAILED: "[FAIL]"}


def _render(report: dict[str, Any], *, strict: bool) -> None:
    for check in report["checks"]:
        icon = _ICONS.get(check["status"], "[ ?? ]")
        print(f"{icon} {check['name']}: {check['detail']}")
        if check.get("fix") and check["status"] != OK:
            print(f"        fix: {check['fix']}")
    print()
    if report["failed"]:
        print(f"FAILED ({len(report['failed'])}): {', '.join(report['failed'])}")
    if report["degraded"]:
        label = "FAILED (--strict)" if strict else "unavailable features"
        print(f"{label} ({len(report['degraded'])}): {', '.join(report['degraded'])}")
    if not report["failed"] and not report["degraded"]:
        print("All checks passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also exit non-zero when an OPTIONAL feature is unavailable",
    )
    args = parser.parse_args()

    try:
        report = validate()
    except Exception as exc:  # noqa: BLE001 - a bad .env must produce a diagnosis, not a traceback
        print(f"Environment validation could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _render(report, strict=args.strict)

    if report["failed"]:
        return 1
    if args.strict and report["degraded"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
