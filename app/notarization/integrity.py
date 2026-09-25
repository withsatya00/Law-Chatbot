"""Document hashing and verification-token generation.

Two separate jobs that both must be got right, kept together because they are
the integrity primitives the rest of the package builds on:

* `document_hash` -- the SHA-256 that identifies exactly which bytes a
  signature or a notarization applies to. A notarization is only ever
  recorded against a hash, never against a mutable document id, so altering
  the content necessarily invalidates it.

* `new_verification_token` -- the random, unguessable handle that appears in
  a notarized document's QR code. It carries NO information: it is not
  derived from the document, the hash, the user, or the notary, so possessing
  or brute-forcing one reveals nothing and cannot be reversed into personal
  data. The public verification endpoint looks it up and returns only the
  small, fixed set of fields in `PublicVerificationResult`.
"""

import hashlib
import hmac
import secrets
from pathlib import Path

# 32 bytes of `secrets` entropy, URL-safe. Long enough that enumeration is
# not a concern even without the rate limit that also guards the endpoint.
_VERIFICATION_TOKEN_BYTES = 32


def document_hash(content: bytes) -> str:
    """The lowercase hex SHA-256 of `content`."""
    return hashlib.sha256(content).hexdigest()


def hash_file(path: str | Path) -> str:
    """SHA-256 of a file, read in chunks so a large export is not loaded whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_sections(sections: dict[str, str]) -> str:
    """SHA-256 over a draft's rendered sections.

    Serialized deterministically -- headings in their stored order, separated
    by characters that cannot occur in a heading -- so the same document
    always hashes identically, and so no reordering or heading/body boundary
    shift can collide with a different document.
    """
    payload = "\x1e".join(f"{heading}\x1f{text}" for heading, text in sections.items())
    return document_hash(payload.encode("utf-8"))


def hashes_match(left: str, right: str) -> bool:
    """Constant-time hash comparison.

    Timing-safe because this decides whether a document is presented to the
    public as verified: a comparison that leaks position information is a
    comparison worth removing, even where an attack is impractical.
    """
    return hmac.compare_digest((left or "").lower(), (right or "").lower())


def new_verification_token() -> str:
    return secrets.token_urlsafe(_VERIFICATION_TOKEN_BYTES)
