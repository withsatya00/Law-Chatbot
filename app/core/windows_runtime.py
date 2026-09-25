"""Windows host repairs that must run before any native (GTK/OpenSSL) import.

Avast injects `SSLKEYLOGFILE=\\\\.\\aswMonFltProxy\\<id>` into every process it
watches. OpenSSL cannot open a raw device path, and instead of failing the TLS
handshake it ABORTS the process from C with

    OPENSSL_Uplink(0x...,08): no OPENSSL_Applink

with no Python traceback and no output at all. Because the injection happens
per process, clearing the variable in the parent shell does not help: the next
`python.exe` gets it again. The repair therefore has to run inside the process
that is about to load the native stack -- which is why this lives in `app/`
rather than only in `scripts/validate_environment.py`, whose in-process fix
never reached the API or a pytest worker.

`scripts/_win_env.ps1` is the shell-level equivalent; the PATH order here is
deliberately the same and is load-bearing (MSYS2's GTK stack must be found
before Tesseract's own mingw64 copies of the same library names).
"""

import os
import sys
from pathlib import Path

# Device-path prefixes a normal user-configured key-log file never has. Avast's
# proxy is the one seen in the wild; the generic `\\.\` prefix covers the same
# class of injection from any other filter driver.
_UNSAFE_SSLKEYLOG_PREFIXES = ("\\\\.\\", "//./", "\\\\?\\aswMonFltProxy")


def _accessible_directory(path: Path) -> bool:
    """Optional host tools must not prevent application startup under restricted ACLs."""
    try:
        return path.is_dir()
    except OSError:
        return False


def is_unsafe_ssl_key_log(value: str | None) -> bool:
    """Whether `value` is an injected device path rather than a real file.

    Deliberately narrow: an operator who genuinely configured
    `SSLKEYLOGFILE=C:\\tmp\\keys.log` to debug TLS keeps it. Only a path OpenSSL
    cannot open as a file -- a device path -- is treated as hostile.
    """
    if not value:
        return False
    normalized = value.strip()
    if not normalized:
        return False
    if any(normalized.startswith(prefix) for prefix in _UNSAFE_SSLKEYLOG_PREFIXES):
        return True
    return "aswMonFltProxy" in normalized


def repair_windows_host_env() -> list[str]:
    """Apply the host repairs, in-process. Idempotent; a no-op off Windows.

    Returns the human-readable list of repairs applied, so a caller can state
    what it changed rather than silently papering over the host.
    """
    if sys.platform != "win32":
        return []
    applied: list[str] = []

    if is_unsafe_ssl_key_log(os.environ.get("SSLKEYLOGFILE")):
        os.environ.pop("SSLKEYLOGFILE", None)
        applied.append("cleared SSLKEYLOGFILE (an injected device path OpenSSL cannot open)")

    # Order matches `_win_env.ps1` and is load-bearing.
    candidates = [
        Path(r"C:\msys64\ucrt64\bin"),
        Path(r"C:\Program Files\Tesseract-OCR"),
    ]
    # Poppler lives under a version-stamped winget directory; resolve it by
    # pattern so an upgrade does not silently strip it.
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if _accessible_directory(packages):
        try:
            for package in sorted(packages.glob("oschwartz10612.Poppler*")):
                for version in sorted(package.glob("poppler-*"), reverse=True):
                    bin_dir = version / "Library" / "bin"
                    if _accessible_directory(bin_dir):
                        candidates.append(bin_dir)
                        break
        except OSError:
            # Discovery is optional; explicitly configured tools on PATH still work.
            pass

    existing = os.environ.get("PATH", "").split(os.pathsep)
    added = [str(d) for d in candidates if _accessible_directory(d) and str(d) not in existing]
    if added:
        os.environ["PATH"] = os.pathsep.join(added + existing)
        applied.append(f"prepended {len(added)} native-tool directories to PATH")
    return applied
