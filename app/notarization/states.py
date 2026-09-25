"""The notarization state machine.

Kept as pure data + pure functions, with no database or framework imports, so
the legal correctness of the lifecycle can be read (and tested) in one place
rather than inferred from the handlers that drive it.
"""

from typing import Literal

DocumentStatus = Literal[
    "draft",
    "ready_for_signature",
    "signing_in_progress",
    "signed",
    "notary_review_requested",
    "notary_review_in_progress",
    "notarized",
    "revoked",
]

DRAFT: DocumentStatus = "draft"
READY_FOR_SIGNATURE: DocumentStatus = "ready_for_signature"
SIGNING_IN_PROGRESS: DocumentStatus = "signing_in_progress"
SIGNED: DocumentStatus = "signed"
NOTARY_REVIEW_REQUESTED: DocumentStatus = "notary_review_requested"
NOTARY_REVIEW_IN_PROGRESS: DocumentStatus = "notary_review_in_progress"
NOTARIZED: DocumentStatus = "notarized"
REVOKED: DocumentStatus = "revoked"

ALL_STATUSES: tuple[DocumentStatus, ...] = (
    DRAFT,
    READY_FOR_SIGNATURE,
    SIGNING_IN_PROGRESS,
    SIGNED,
    NOTARY_REVIEW_REQUESTED,
    NOTARY_REVIEW_IN_PROGRESS,
    NOTARIZED,
    REVOKED,
)

# The ONLY status that may ever be presented to a user as "Notarized".
NOTARIZED_STATUSES: frozenset[str] = frozenset({NOTARIZED})

# Allowed transitions. Anything not listed here is rejected.
#
# Note the deliberate asymmetries:
#   * `signing_in_progress` can fall BACK to `ready_for_signature`. A failed,
#     expired or cancelled e-sign attempt must leave the document
#     non-signed -- never parked in a state that reads as further along than
#     it is.
#   * `notary_review_in_progress` can fall back to `signed` (rejection). A
#     rejected request must leave the document NON-notarized.
#   * `notarized` leads only to `revoked`. It is never an input to signing or
#     review again: any change to a notarized document creates a NEW version
#     that starts at `draft` (see `NotarizationService.create_version`), which
#     is what keeps a notarized artefact immutable.
#   * `revoked` is terminal. A revoked document is never resurrected; a
#     replacement is a new version.
_TRANSITIONS: dict[DocumentStatus, frozenset[DocumentStatus]] = {
    DRAFT: frozenset({READY_FOR_SIGNATURE}),
    READY_FOR_SIGNATURE: frozenset({SIGNING_IN_PROGRESS, DRAFT}),
    SIGNING_IN_PROGRESS: frozenset({SIGNED, READY_FOR_SIGNATURE}),
    SIGNED: frozenset({NOTARY_REVIEW_REQUESTED, DRAFT}),
    NOTARY_REVIEW_REQUESTED: frozenset({NOTARY_REVIEW_IN_PROGRESS, SIGNED}),
    NOTARY_REVIEW_IN_PROGRESS: frozenset({NOTARIZED, SIGNED}),
    NOTARIZED: frozenset({REVOKED}),
    REVOKED: frozenset(),
}


def can_transition(current: str, target: str) -> bool:
    return target in allowed_transitions(current)


def allowed_transitions(current: str) -> frozenset[str]:
    """The statuses reachable from `current`; empty for an unknown status.

    `current` is `str`, not `DocumentStatus`, because callers pass a status
    read back out of the database -- including one written by an older build.
    The narrowing is done here, once, so `_TRANSITIONS` itself stays keyed by
    the closed literal and a typo in the table is still a type error.
    """
    if current not in _TRANSITIONS:
        return frozenset()
    return _TRANSITIONS[current]


def is_notarized(status: str) -> bool:
    """The only sanctioned way to ask "may this be shown as notarized?"."""
    return status in NOTARIZED_STATUSES


def is_editable(status: str) -> bool:
    """Whether the document's CONTENT may still be changed in place.

    False for everything from `signed` onward: once a signature exists over a
    hash, editing the bytes under it would silently invalidate the signature.
    Callers must create a new version instead.
    """
    return status in {DRAFT, READY_FOR_SIGNATURE}


# Terminal e-sign outcomes that must NOT advance the document.
FAILED_SIGNING_STATES: frozenset[str] = frozenset({"failed", "expired", "cancelled"})
