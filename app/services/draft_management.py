"""Draft management operations, in the service layer where they belong.

These four operations -- delete, duplicate, version comparison and the
structured review -- were implemented inside the FastAPI route bodies in
`app/api/drafting.py`. That was fine while REST was the only caller. It is
not fine now that a chat workflow performs the same operations: business
logic implemented in a route can only be reused by re-implementing it, and a
second implementation of "who may delete this draft" is exactly the kind of
divergence that turns into a cross-owner data leak.

So the logic moved here unchanged and both callers use it. In particular
`ensure_draft_access` is now the single ownership rule for drafts, enforced
in the service and therefore enforced identically whether the request
arrived as a REST call or as a sentence typed into the chat.
"""

import difflib
import re
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.casefile.facts import facts_from_fields
from app.core.exceptions import BadRequestError, ForbiddenError, NotFoundError
from app.drafting.engine import LegalDraftEngine
from app.schemas.phase2 import DraftCompareResponse, DraftDuplicateResponse, DraftReviewResponse


def ensure_draft_access(
    draft: dict[str, Any], authenticated_user_id: str | None, session_id: str | None
) -> None:
    """Mirrors `DocumentService._ensure_document_access`'s three-way
    visibility rule: no owner stamped on the draft (legacy record) -> open;
    `user_id` stamped -> only that exact authenticated user, from any
    session; `session_id`-only stamped (anonymous draft) -> only that exact
    session. Every mutating draft path must call this with the fetched
    draft record before acting on it -- drafts previously had no ownership
    enforcement at all.
    """
    owner_user_id = draft.get("user_id")
    owner_session_id = draft.get("session_id")
    if not owner_user_id and not owner_session_id:
        return
    if owner_user_id:
        if authenticated_user_id and authenticated_user_id == owner_user_id:
            return
        raise ForbiddenError("You do not have access to this draft.")
    if session_id and session_id == owner_session_id:
        return
    raise ForbiddenError("You do not have access to this draft.")


async def owned_draft(
    engine: LegalDraftEngine, draft_id: str, user_id: str | None, session_id: str | None
) -> dict[str, Any]:
    """Fetch-then-authorize, in one place so no caller can skip the second half."""
    draft = await engine.drafts.find_by_id(draft_id)
    if draft is None:
        raise NotFoundError(f"Draft not found: {draft_id}")
    ensure_draft_access(draft, user_id, session_id)
    return draft


def render_sections(sections: dict[str, str]) -> str:
    return "\n\n".join(f"{heading}\n{text}" for heading, text in sections.items())


async def delete_draft(draft_id: str, user_id: str | None, session_id: str | None) -> dict[str, object]:
    """Erase one draft and its version chain.

    A draft is the single most PII-dense record this product creates -- name,
    full postal address, mobile number, bank, transaction reference and a
    free-text account of what happened, all in one document. The version
    history goes with it: leaving `draft_versions` behind would retain every
    earlier revision of exactly the text just deleted.
    """
    engine = LegalDraftEngine()
    await owned_draft(engine, draft_id, user_id, session_id)
    versions_removed = await engine.versions.collection.delete_many({"draft_id": draft_id})
    await engine.drafts.delete_by_id(draft_id)
    return {"status": "deleted", "draft_id": draft_id, "versions_deleted": versions_removed.deleted_count}


async def duplicate_draft(
    draft_id: str, user_id: str | None, session_id: str | None
) -> DraftDuplicateResponse:
    engine = LegalDraftEngine()
    source = await owned_draft(engine, draft_id, user_id, session_id)
    now = datetime.now(UTC)
    new_id = str(uuid4())
    duplicate = {
        key: value for key, value in source.items()
        if key not in {"_id", "created_at", "updated_at", "lifecycle_state"}
    }
    duplicate.update({
        "_id": new_id,
        "created_at": now,
        "updated_at": now,
        "lifecycle_state": "preview_ready",
        "duplicated_from": draft_id,
    })
    if user_id:
        duplicate["user_id"] = user_id
    elif session_id:
        duplicate["session_id"] = session_id
    await engine.drafts.insert(duplicate)
    await engine.versions.insert({
        "draft_id": new_id,
        "version_number": 1,
        "sections": duplicate.get("sections", {}),
        "fields": duplicate.get("fields", {}),
        "language": duplicate.get("language", "english"),
        "document_status": "active",
        "note": f"Duplicated from {draft_id}",
    })
    return DraftDuplicateResponse(source_draft_id=draft_id, draft_id=new_id)


async def compare_versions(
    draft_id: str, original_version: int, revised_version: int,
    user_id: str | None, session_id: str | None,
) -> DraftCompareResponse:
    engine = LegalDraftEngine()
    await owned_draft(engine, draft_id, user_id, session_id)
    original = await engine.versions.get_version(draft_id, original_version)
    revised = await engine.versions.get_version(draft_id, revised_version)
    if original is None or revised is None:
        raise NotFoundError("One or both requested draft versions do not exist.")
    original_text = render_sections(original.get("sections", {}))
    revised_text = render_sections(revised.get("sections", {}))
    diff = "\n".join(difflib.unified_diff(
        original_text.splitlines(), revised_text.splitlines(),
        fromfile=f"version-{original_version}", tofile=f"version-{revised_version}", lineterm="",
    ))
    return DraftCompareResponse(
        draft_id=draft_id,
        original_version=original_version,
        revised_version=revised_version,
        original_text=original_text,
        revised_text=revised_text,
        unified_diff=diff,
    )


async def build_review(
    draft_id: str, user_id: str | None, session_id: str | None
) -> DraftReviewResponse:
    """Structured, read-only review workspace for a stored draft."""
    engine = LegalDraftEngine()
    draft = await owned_draft(engine, draft_id, user_id, session_id)
    template = engine._require_template(draft["draft_type"])
    fields = draft.get("fields", {})
    facts = facts_from_fields(fields)
    missing = [
        field.label for field in template.all_fields()
        if field.required and not str(fields.get(field.key, "")).strip()
    ]
    findings = engine._audit_findings(template, draft.get("sections", {}), fields)
    versions = await engine.list_versions(draft_id)
    full_text = render_sections(draft.get("sections", {}))
    legal_refs = sorted(set(re.findall(r"\b(?:Section|Article|Rule)\s+[0-9A-Za-z()./-]+", full_text, re.IGNORECASE)))
    annexures = [
        {"reference": match.group(0)}
        for match in re.finditer(r"\bAnnexure\s+[A-Z]{1,3}-\d+\b", full_text, re.IGNORECASE)
    ]
    unresolved_conflicts = [
        item for item in draft.get("conflicts", [])
        if item.get("slot") not in draft.get("resolved_conflicts", {})
    ]
    return DraftReviewResponse(
        draft_id=draft_id,
        user_facts=fields,
        extracted_facts=[
            {"slot": fact.slot, "label": fact.label, "value": fact.value, "source": fact.source}
            for fact in facts
        ],
        legal_references=legal_refs,
        missing_information=missing,
        possible_assumptions=findings,
        evidence_annexures=annexures,
        editable_sections=draft.get("sections", {}),
        version_history=[
            {
                "version_number": version["version_number"],
                "created_at": version.get("created_at"),
                "language": version.get("language", "english"),
                "note": version.get("note", ""),
            }
            for version in versions
        ],
        lifecycle_state=draft.get("lifecycle_state", "preview_ready"),
        conflicts=unresolved_conflicts,
        final_export_blocked=bool(unresolved_conflicts),
    )


async def resolve_conflict(
    draft_id: str, slot: str, chosen_value: str, user_id: str | None, session_id: str | None
) -> DraftReviewResponse:
    """Records the user's choice between two conflicting facts.

    `chosen_value` must be one of the values the conflict actually offered:
    a legal document has to carry a value the user can stand behind, and
    accepting a free-text third answer here would let an unreviewed value
    into the draft through the back door.
    """
    engine = LegalDraftEngine()
    draft = await owned_draft(engine, draft_id, user_id, session_id)
    conflicts = {item.get("slot"): item for item in draft.get("conflicts", [])}
    conflict = conflicts.get(slot)
    if conflict is None:
        raise BadRequestError(f"No unresolved conflict exists for '{slot}'.")
    allowed = {str(value) for value in conflict.get("values", [])}
    if allowed and chosen_value not in allowed:
        raise BadRequestError("chosen_value must be one of the values shown in the conflict.")
    resolved = {**draft.get("resolved_conflicts", {}), slot: chosen_value}
    await engine.drafts.update_by_id(draft_id, {"resolved_conflicts": resolved})
    return await build_review(draft_id, user_id, session_id)
