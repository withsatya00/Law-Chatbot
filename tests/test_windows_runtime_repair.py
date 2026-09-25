"""The Windows host repair that keeps native (GTK/OpenSSL) imports alive.

Avast injects `SSLKEYLOGFILE` as a device path into every watched process;
OpenSSL aborts the interpreter from C rather than raising, which killed the API
and any pytest run that rendered a PDF. These tests pin the two halves that
matter: the injected value is removed, and a real one an operator configured
themselves is not.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.exceptions import UnsupportedExportError
from app.core.windows_runtime import is_unsafe_ssl_key_log, repair_windows_host_env


def test_injected_avast_device_path_is_recognised_as_unsafe() -> None:
    assert is_unsafe_ssl_key_log(r"\\.\aswMonFltProxy\115ed23d4320f3d3")
    assert is_unsafe_ssl_key_log(r"\\.\pipe\anything")
    assert is_unsafe_ssl_key_log("//./aswMonFltProxy/abc")


def test_a_real_user_configured_key_log_path_is_not_unsafe() -> None:
    assert not is_unsafe_ssl_key_log(r"C:\tmp\tls-keys.log")
    assert not is_unsafe_ssl_key_log("/home/dev/keys.log")
    assert not is_unsafe_ssl_key_log("")
    assert not is_unsafe_ssl_key_log(None)


def test_repair_removes_the_injected_value(monkeypatch: pytest.MonkeyPatch) -> None:
    if sys.platform != "win32":
        pytest.skip("Windows-only host repair")
    monkeypatch.setenv("SSLKEYLOGFILE", r"\\.\aswMonFltProxy\deadbeef")

    applied = repair_windows_host_env()

    assert "SSLKEYLOGFILE" not in os.environ
    assert any("SSLKEYLOGFILE" in line for line in applied)


def test_repair_preserves_a_valid_key_log_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Debugging TLS is a legitimate thing to have set up; the repair must not
    silently disable it."""
    if sys.platform != "win32":
        pytest.skip("Windows-only host repair")
    monkeypatch.setenv("SSLKEYLOGFILE", r"C:\tmp\tls-keys.log")

    repair_windows_host_env()

    assert os.environ["SSLKEYLOGFILE"] == r"C:\tmp\tls-keys.log"


def test_importing_the_export_module_cannot_abort_collection() -> None:
    """Imported in a fresh child process WITH the hostile value injected: the
    import must return normally rather than take the interpreter down."""
    env = {**os.environ, "SSLKEYLOGFILE": r"\\.\aswMonFltProxy\115ed23d4320f3d3"}
    probe = subprocess.run(
        [sys.executable, "-c", "import app.drafting.export; print('ok')"],
        capture_output=True,
        env=env,
        check=False,
    )

    assert probe.returncode == 0, probe.stderr.decode(errors="replace")
    assert b"ok" in probe.stdout
    assert b"OPENSSL_Applink" not in probe.stderr


def test_pdf_export_produces_a_readable_pdf_or_fails_explicitly(tmp_path: Path) -> None:
    """When the renderer is available this must produce a real PDF -- skipping
    every PDF test is not a passing state. When it is genuinely unavailable the
    failure must be an explicit `UnsupportedExportError`, not a dead process."""
    from app.drafting import export

    try:
        html = export._html_renderer()
    except UnsupportedExportError as exc:
        assert "PDF export is unavailable" in str(exc)
        return

    destination = tmp_path / "probe.pdf"
    html(string="<html><body><h1>Reconciliation probe</h1></body></html>").write_pdf(str(destination))

    assert destination.exists()
    assert destination.read_bytes().startswith(b"%PDF-")
