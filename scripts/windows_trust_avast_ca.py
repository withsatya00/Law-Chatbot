r"""Environment-only fix: teach this venv's certifi bundle about the local
TLS-interception root CA (Avast Web/Mail Shield).

Nothing in `app/` is touched or imported by this script -- it only patches the
CA bundle that `httpx` / `requests` / `huggingface_hub` read at runtime.

WHY THIS IS NEEDED ON THIS MACHINE
----------------------------------
Avast Antivirus' "Web Shield" terminates and re-signs every outbound HTTPS
connection with its own root CA ("CN=Avast Web/Mail Shield Root"). That root is
installed in the Windows LocalMachine\Root store, so anything using the OS
trust store (urllib, PowerShell, browsers) works fine. But `httpx` -- which is
how every LLM provider in `app/llm/` talks to its API -- and `requests` (used by
`huggingface_hub` to download the embedding/reranker models) both verify
against `certifi`'s *bundled* PEM file instead, which does not and will never
contain a locally generated interception root. Result:

    httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify
    failed: unable to get local issuer certificate

This script copies that already-OS-trusted root into the venv's certifi bundle
so Python agrees with the rest of the machine. It adds nothing that Windows is
not already trusting.

Idempotent: re-running is a no-op. Re-run it after any `pip install -U certifi`,
which replaces cacert.pem wholesale.

THE ALTERNATIVE, IF YOU PREFER: turn off Avast -> Protection -> Core Shields ->
Web Shield -> "Enable HTTPS scanning". Then no interception happens, stock
certifi verifies real certificates, and this script is unnecessary.

Usage:  .venv\Scripts\python.exe scripts\windows_trust_avast_ca.py
"""

import ssl
import sys
from pathlib import Path

MARKER = "# --- locally added: Windows-store TLS interception root(s) ---"
WANTED_SUBSTRINGS = ("Avast Web/Mail Shield", "AVG Web/Mail Shield", "Kaspersky Anti-Virus Personal Root")


def windows_root_pems() -> list[tuple[str, str]]:
    """Return (label, PEM) for interception roots found in the Windows ROOT store."""
    if not hasattr(ssl, "enum_certificates"):
        sys.exit("ssl.enum_certificates() is unavailable -- this script is Windows-only.")
    found = []
    for der, _enc, _trust in ssl.enum_certificates("ROOT"):
        pem = ssl.DER_cert_to_PEM_cert(der)
        try:
            from cryptography import x509

            subject = x509.load_pem_x509_certificate(pem.encode()).subject.rfc4514_string()
        except Exception:  # noqa: BLE001 - an unparseable certificate in the Windows store is skipped, not fatal to the scan
            subject = "<unparsed>"
        if any(s in subject for s in WANTED_SUBSTRINGS):
            found.append((subject, pem))
    return found


def main() -> int:
    import certifi

    bundle = Path(certifi.where())
    existing = bundle.read_text(encoding="utf-8")

    roots = windows_root_pems()
    if not roots:
        print("No TLS-interception root found in the Windows ROOT store -- nothing to do.")
        print("If httpx still fails to verify certificates, HTTPS scanning may be off already.")
        return 0

    additions = [(label, pem) for label, pem in roots if pem.strip() not in existing]
    if not additions:
        print(f"Already present in {bundle} -- no change.")
        return 0

    with bundle.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{MARKER}\n")
        for label, pem in additions:
            handle.write(f"# {label}\n{pem}")

    for label, _ in additions:
        print(f"Appended to certifi bundle: {label}")
    print(f"Bundle: {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
