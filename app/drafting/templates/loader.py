"""Loads draft template definitions from the YAML files in this directory.

Adding a new draft type is a pure data change: drop a new `*.yaml` file here
following the shape of any existing template (see `_REQUIRED_TOP_LEVEL_KEYS`
below for the minimum required keys) — no Python code changes needed
anywhere in the drafting engine, API, or conversation flow.
"""

import functools
from pathlib import Path
from typing import Any

import yaml

from app.drafting.templates.base import (
    CASE_STAGES,
    DOCUMENT_FAMILIES,
    TEMPLATE_STATUSES,
    DraftField,
    DraftTemplateDefinition,
)

_TEMPLATES_DIR = Path(__file__).parent

_REQUIRED_TOP_LEVEL_KEYS = {
    "draft_id",
    "name",
    "hindi_name",
    "category",
    "description",
    "authority_label",
    "required_fields",
}
_REQUIRED_FIELD_KEYS = {"key", "label", "hindi_label"}


def _build_field(raw: dict[str, Any], *, template_id: str) -> DraftField:
    missing = _REQUIRED_FIELD_KEYS - raw.keys()
    if missing:
        raise ValueError(f"Field in template '{template_id}' is missing keys: {sorted(missing)}")
    return DraftField(
        key=raw["key"],
        label=raw["label"],
        hindi_label=raw["hindi_label"],
        field_type=raw.get("field_type", "text"),
        required=raw.get("required", True),
        help_text=raw.get("help_text", ""),
        must_appear_verbatim=raw.get("must_appear_verbatim", False),
    )


def _build_template(raw: dict[str, Any], source: Path) -> DraftTemplateDefinition:
    missing = _REQUIRED_TOP_LEVEL_KEYS - raw.keys()
    if missing:
        raise ValueError(f"Draft template file {source.name} is missing required keys: {sorted(missing)}")
    draft_id = raw["draft_id"]
    # A stated reply/payment period must name the provision that fixes it.
    # Post-Phase-3 hardening milestone C removed a hardcoded "or 15 days"
    # default from the notice builder: an arbitrary deadline in a legal notice
    # is a demand the sender has no authority to make, and a template that
    # wants to state one has to record where it comes from.
    period_days_raw = raw.get("statutory_response_period_days")
    period_basis = str(raw.get("statutory_response_period_basis", "")).strip()
    period_days = int(period_days_raw) if period_days_raw is not None else None
    if (period_days is None) != (not period_basis):
        raise ValueError(
            f"Draft template file {source.name} must declare "
            "statutory_response_period_days and statutory_response_period_basis together, or neither."
        )
    unknown_particulars = [
        key
        for key in raw.get("particulars_fields", [])
        if key not in {f["key"] for f in [*raw["required_fields"], *raw.get("optional_fields", [])]}
    ]
    if unknown_particulars:
        raise ValueError(
            f"Draft template file {source.name} lists particulars_fields that are not fields of the "
            f"template: {sorted(unknown_particulars)}"
        )
    discovery_kwargs = _build_discovery_metadata(raw, source)
    return DraftTemplateDefinition(
        draft_id=draft_id,
        name=raw["name"],
        hindi_name=raw["hindi_name"],
        category=raw["category"],
        description=raw["description"],
        authority_label=raw["authority_label"],
        applicable_acts_hint=raw.get("applicable_acts_hint", []),
        applicable_sections_hint=raw.get("applicable_sections_hint", []),
        required_fields=[_build_field(f, template_id=draft_id) for f in raw["required_fields"]],
        optional_fields=[_build_field(f, template_id=draft_id) for f in raw.get("optional_fields", [])],
        drafting_notes=raw.get("drafting_notes", ""),
        trigger_phrases=[str(phrase).lower() for phrase in raw.get("trigger_phrases", [])],
        field_synonyms={str(k).lower(): v for k, v in raw.get("field_synonyms", {}).items()},
        subject_template=raw.get("subject_template", ""),
        legal_enrichment_hints=[str(hint) for hint in raw.get("legal_enrichment_hints", [])],
        typical_supporting_documents=[str(doc) for doc in raw.get("typical_supporting_documents", [])],
        advocate_register=bool(raw.get("advocate_register", True)),
        particulars_fields=[str(key) for key in raw.get("particulars_fields", [])],
        statutory_response_period_days=period_days,
        statutory_response_period_basis=str(raw.get("statutory_response_period_basis", "")).strip(),
        permitted_statements=[str(key) for key in raw.get("permitted_statements", [])],
        **discovery_kwargs,
    )


def _build_discovery_metadata(raw: dict[str, Any], source: Path) -> dict[str, Any]:
    """Parses the optional problem-first-discovery metadata block (see
    `DraftTemplateDefinition`'s "Problem-first discovery metadata" fields).

    Every key here is optional -- a template YAML that sets none of them
    gets the dataclass's own defaults, so all 55 pre-existing templates keep
    loading unchanged. What IS validated (raising `ValueError` with the
    offending template's filename) is the *shape* of whatever a template
    chooses to declare, so a typo in a new template's metadata fails loudly
    at startup instead of silently producing a template `recommendation.py`
    can never match or a value the API schemas reject at request time.
    """
    document_family = str(raw.get("document_family", "")).strip()
    if document_family and document_family not in DOCUMENT_FAMILIES:
        raise ValueError(
            f"Draft template file {source.name} has invalid document_family "
            f"'{document_family}'; must be one of {sorted(DOCUMENT_FAMILIES)}"
        )
    case_stages = [str(stage) for stage in raw.get("case_stages", [])]
    invalid_stages = [stage for stage in case_stages if stage not in CASE_STAGES]
    if invalid_stages:
        raise ValueError(
            f"Draft template file {source.name} has invalid case_stages {sorted(invalid_stages)}; "
            f"must be from {sorted(CASE_STAGES)}"
        )
    template_status = str(raw.get("template_status", "production")).strip()
    if template_status not in TEMPLATE_STATUSES:
        raise ValueError(
            f"Draft template file {source.name} has invalid template_status "
            f"'{template_status}'; must be one of {sorted(TEMPLATE_STATUSES)}"
        )
    jurisdiction = raw.get("jurisdiction")
    if jurisdiction is not None and not isinstance(jurisdiction, dict):
        raise ValueError(f"Draft template file {source.name}: jurisdiction must be a mapping, not {type(jurisdiction).__name__}")
    aliases_raw = raw.get("aliases")
    aliases: dict[str, list[str]] = {}
    if aliases_raw is not None:
        if not isinstance(aliases_raw, dict):
            raise ValueError(f"Draft template file {source.name}: aliases must be a mapping of language -> list")
        for lang, names in aliases_raw.items():
            if not isinstance(names, list):
                # ValueError, not TypeError (TRY004's default suggestion): every
                # other malformed-field check in this function -- lines just
                # above/below this one -- raises ValueError for "this template
                # file's data doesn't match the expected shape"; switching only
                # this one case would make identical failures raise two
                # different exception types depending on which field is wrong.
                raise ValueError(  # noqa: TRY004
                    f"Draft template file {source.name}: aliases.{lang} must be a list of strings"
                )
            aliases[str(lang).lower()] = [str(name) for name in names]
    review_requirements_raw = raw.get("review_requirements")
    review_requirements: dict[str, bool] = {
        "jurisdiction_check": False,
        "limitation_check": False,
        "court_fee_check": False,
        "advocate_review_recommended": False,
    }
    if review_requirements_raw is not None:
        if not isinstance(review_requirements_raw, dict):
            raise ValueError(f"Draft template file {source.name}: review_requirements must be a mapping")
        unknown_keys = set(review_requirements_raw) - set(review_requirements)
        if unknown_keys:
            raise ValueError(
                f"Draft template file {source.name}: unknown review_requirements keys {sorted(unknown_keys)}"
            )
        review_requirements.update({k: bool(v) for k, v in review_requirements_raw.items()})
    kwargs: dict[str, Any] = {
        "document_family": document_family,
        "domain": str(raw.get("domain", "")).strip(),
        "subcategory": str(raw.get("subcategory", "")).strip(),
        "supported_issues": [str(v) for v in raw.get("supported_issues", [])],
        "user_roles": [str(v) for v in raw.get("user_roles", [])],
        "opposite_party_roles": [str(v) for v in raw.get("opposite_party_roles", [])],
        "desired_reliefs": [str(v) for v in raw.get("desired_reliefs", [])],
        "case_stages": case_stages,
        "forums": [str(v) for v in raw.get("forums", [])],
        "aliases": aliases,
        "supporting_documents": [str(v) for v in raw.get("supporting_documents", [])],
        "review_requirements": review_requirements,
        "template_status": template_status,
        "template_version": str(raw.get("template_version", "1.0")),
    }
    if jurisdiction is not None:
        kwargs["jurisdiction"] = jurisdiction
    return kwargs


@functools.lru_cache(maxsize=1)
def _load_all() -> dict[str, DraftTemplateDefinition]:
    templates: dict[str, DraftTemplateDefinition] = {}
    for path in sorted(_TEMPLATES_DIR.glob("*.yaml")):
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        template = _build_template(raw, path)
        if template.draft_id in templates:
            raise ValueError(f"Duplicate draft_id '{template.draft_id}' found in {path.name}")
        templates[template.draft_id] = template
    return templates


def list_templates() -> list[DraftTemplateDefinition]:
    return list(_load_all().values())


def get_template(draft_id: str | None) -> DraftTemplateDefinition | None:
    """The template for `draft_id`, or `None` if there is no such template.

    Accepts `None` because that is how it is actually called: every caller in
    `app/drafting/conversation.py` passes `memory.get("draft_template_id")`
    straight through, which is absent for a session that has not chosen a
    template yet. Returning `None` for `None` -- rather than looking up the
    literal key `None` -- keeps the one "no template here" answer those
    callers already branch on.
    """
    if not draft_id:
        return None
    return _load_all().get(draft_id)
