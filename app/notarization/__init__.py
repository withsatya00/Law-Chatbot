"""E-Notarization preparation and verification.

READ THIS BEFORE CHANGING ANYTHING IN THIS PACKAGE.

This module prepares documents FOR notarization and records the outcome of a
notarization performed by a licensed human notary. It does not, and must
never, notarize anything itself.

The single rule the whole package is built around: a document is "notarized"
if and only if a verified, active notary account has explicitly approved a
notarization request for that exact document hash. Not because it was
generated here. Not because it was digitally signed. Not because it was
uploaded, exported, or stamped. Every state transition, permission check and
label in this package exists to keep that statement true.

Six distinct things are deliberately kept distinguishable end to end, because
conflating any two of them would misrepresent a document's legal status:

    1. AI-generated draft         -- produced by `app.drafting`
    2. User-signed document       -- the user applied their own signature
    3. E-signed document          -- an e-sign provider completed a signature
    4. Notary-ready document      -- checklist complete, nothing verified yet
    5. Notarization requested     -- submitted to a notary, pending review
    6. Notarized document         -- a verified notary approved it

Explicitly NOT implemented here, and not to be added: any generation of
notary stamps, notary signatures, registration numbers, certificates, or
notarization records. There is no code path that can mint a notarization
without a verified notary's own approval action, and requests to add one
should be refused.
"""
