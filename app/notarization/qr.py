"""QR code generation for NOTARIZED documents only.

`build_verification_qr_svg` is deliberately the only entry point, and it
refuses to produce anything unless the document's status is exactly
`notarized`. A QR code on this platform is a claim that a verified notary
attested the document; putting one on a draft, on a signed-but-not-notarized
document, or on a revoked one would make that claim falsely. That gate -- not
the encoding -- is what this module exists to enforce.

The encoded payload is the public verification URL and nothing else: a random
token carrying no information about the document, its content, its signer, or
its owner (see `integrity.new_verification_token`).

On the encoder: an earlier revision of this file hand-rolled QR encoding to
avoid a dependency. It produced structurally plausible output that OpenCV's
decoder could not read at all -- the mask evaluation, format/version BCH bits
and block interleaving are a large spec surface to get exactly right, and a
verification QR that does not scan is worse than no QR, because it looks like
a working one. Encoding is now delegated to `segno` (pure Python, no native
dependencies), and the module's own test decodes its output to prove it
scans.

`segno` is bound at call time through `optional_deps` rather than imported at
module scope, matching how this codebase treats every other format
dependency: a deployment without it loses the QR on the notarized PDF and is
told what to install, instead of failing to import the notarization package.
"""

import io

from app.core.optional_deps import load
from app.notarization import states


class QRUnavailableError(RuntimeError):
    """Raised when a QR cannot be produced.

    Never catch this and substitute a placeholder image: a document either
    carries a real, scannable verification QR or it carries none at all.
    """


def build_verification_qr_svg(status: str, verification_url: str, *, size_px: int = 132) -> str:
    """The verification QR for a notarized document, as an inline SVG string.

    Inline SVG (rather than a linked file or a raster) so the exporter can
    embed it directly with no external asset and no loss of sharpness at
    print resolution.
    """
    if not states.is_notarized(status):
        raise QRUnavailableError(
            f"A verification QR code may only be placed on a notarized document (status was '{status}')."
        )
    if not verification_url:
        raise QRUnavailableError("A verification URL is required to build a QR code.")

    segno = load("segno", feature="notarization verification QR codes")
    # Error correction M: the standard trade-off for a printed document that
    # may be photocopied or scanned from paper.
    code = segno.make(verification_url, error="m")
    # segno writes SVG as bytes, so the buffer is binary and decoded here.
    buffer = io.BytesIO()
    code.save(buffer, kind="svg", scale=1, border=4, xmldecl=False, svgns=True, omitsize=True, svgclass=None)
    svg = buffer.getvalue().decode("utf-8")
    # Size is applied here rather than through segno's own scaling so the
    # markup stays resolution-independent and the caller controls print size.
    return svg.replace(
        "<svg",
        f"<svg width='{size_px}' height='{size_px}' role='img' "
        f"aria-label='Notarization verification QR code'",
        1,
    )


def verification_caption(notary_name: str, registration_number: str) -> str:
    """The line printed beneath the QR on a notarized document.

    States what the QR proves and, just as importantly, what it does not: it
    confirms this platform's record of the notarization, and is not itself a
    substitute for the notary's own attestation on the document.
    """
    attribution = f"{notary_name} (Reg. No. {registration_number})" if notary_name else "the attesting notary"
    return (
        f"Scan to verify this document's notarization record, attested by {attribution}. "
        "This code confirms the platform's record of notarization; it does not replace the notary's "
        "own seal and signature on the document."
    )
