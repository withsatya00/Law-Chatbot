"""Notary-readiness validation.

Produces the checklist a user sees before export, and decides whether a
document may be labelled "Notary-ready draft". It never decides that anything
is notarized -- that requires a verified notary's approval and nothing else.

The distinction this file exists to hold: "notary-ready" means *we have
collected the details a notary will ask for*. It is a completeness check on
our own form, not a legal verification of identity, and the wording returned
here is careful never to imply otherwise.
"""

from dataclasses import dataclass, field
from typing import Literal

# Document categories eligible for notarization preparation. A police
# complaint or an RTI application is filed, not notarized, so offering the
# action for them would teach users something false about the process.
ELIGIBLE_CATEGORIES: frozenset[str] = frozenset({"Affidavit", "Contract", "Declaration", "NOC", "PowerOfAttorney"})

# Draft template categories in `app/drafting/templates` map onto the above.
# "Contract" covers agreements; affidavits and declarations are their own.
_CATEGORY_ALIASES: dict[str, str] = {
    "affidavit": "Affidavit",
    "agreement": "Contract",
    "contract": "Contract",
    "declaration": "Declaration",
    "noc": "NOC",
    "power_of_attorney": "PowerOfAttorney",
    "powerofattorney": "PowerOfAttorney",
}

Severity = Literal["required", "recommended"]


@dataclass(frozen=True)
class ChecklistItem:
    key: str
    label: str
    severity: Severity
    satisfied: bool
    detail: str = ""


@dataclass(frozen=True)
class NotaryReadinessResult:
    eligible: bool
    complete: bool
    items: list[ChecklistItem] = field(default_factory=list)
    # Always "Notary-ready draft" or "Draft"; never "Notarized". Present so
    # every caller shows the same words rather than composing their own.
    label: str = "Draft"
    ineligible_reason: str = ""

    def missing_required(self) -> list[str]:
        return [item.key for item in self.items if item.severity == "required" and not item.satisfied]

    def as_dicts(self) -> list[dict[str, object]]:
        return [
            {
                "key": item.key,
                "label": item.label,
                "severity": item.severity,
                "satisfied": item.satisfied,
                "detail": item.detail,
            }
            for item in self.items
        ]


def normalize_category(category: str) -> str:
    return _CATEGORY_ALIASES.get((category or "").strip().lower(), (category or "").strip())


def is_eligible(category: str) -> bool:
    return normalize_category(category) in ELIGIBLE_CATEGORIES


# Signer details a notary will require before attesting. `witness_details` is
# conditional: required only for categories where Indian practice expects
# attesting witnesses.
_WITNESS_REQUIRED_CATEGORIES: frozenset[str] = frozenset({"PowerOfAttorney", "Declaration"})

_REQUIRED_SIGNER_FIELDS: tuple[tuple[str, str], ...] = (
    ("full_name", "Signer's full name"),
    ("address", "Signer's address"),
    ("identity_document_type", "Identity document type"),
    ("place", "Place of execution"),
    ("date", "Date of execution"),
)


def evaluate(category: str, signer: dict[str, str], witnesses: list[dict[str, str]] | None = None) -> NotaryReadinessResult:
    """The checklist for one document.

    `signer` holds only the fields a notary needs. Note what is NOT asked
    for: the identity document's NUMBER is deliberately not a required field
    here -- the notary inspects the physical document at attestation. We
    record only its TYPE, so a full identity number never has to enter this
    system at all.
    """
    normalized = normalize_category(category)
    if normalized not in ELIGIBLE_CATEGORIES:
        return NotaryReadinessResult(
            eligible=False,
            complete=False,
            label="Draft",
            ineligible_reason=(
                f"'{category}' documents are not prepared for notarization. Notarization preparation is "
                f"available for: {', '.join(sorted(ELIGIBLE_CATEGORIES))}."
            ),
        )

    items: list[ChecklistItem] = []
    for key, label in _REQUIRED_SIGNER_FIELDS:
        value = (signer or {}).get(key, "")
        items.append(
            ChecklistItem(
                key=key,
                label=label,
                severity="required",
                satisfied=bool(value and str(value).strip()),
            )
        )

    witness_list = witnesses or []
    witness_required = normalized in _WITNESS_REQUIRED_CATEGORIES
    complete_witnesses = [
        witness for witness in witness_list if (witness.get("full_name") or "").strip() and (witness.get("address") or "").strip()
    ]
    items.append(
        ChecklistItem(
            key="witness_details",
            label="Witness name and address",
            severity="required" if witness_required else "recommended",
            satisfied=bool(complete_witnesses),
            detail=(
                "Two attesting witnesses are conventionally required for this document type."
                if witness_required
                else "Optional for this document type, but often requested by the notary."
            ),
        )
    )
    items.append(
        ChecklistItem(
            key="original_identity_document",
            label="Carry the original identity document to the notary",
            severity="recommended",
            satisfied=False,
            detail=(
                "This platform records only the TYPE of identity document. The notary inspects the original "
                "in person; no identity number is stored here."
            ),
        )
    )

    complete = not [item for item in items if item.severity == "required" and not item.satisfied]
    return NotaryReadinessResult(
        eligible=True,
        complete=complete,
        items=items,
        # Even a fully complete checklist is only ever "Notary-ready draft".
        # Nothing in this module can produce the word "Notarized".
        label="Notary-ready draft" if complete else "Draft",
    )
