"""Draft lifecycle management, by talking.

Everything a user could previously only do by calling `/draft/approve`,
`/draft/lock`, `/draft/rollback`, `/draft/{id}/versions`, `/draft/{id}/
compare`, `/draft/{id}/duplicate`, `DELETE /draft/{id}` or `/draft/translate`
is reachable here as a sentence.

Two rules shape the design:

* **No ids.** The user says "delete the rent notice" or "the second one".
  Resolution happens against an owner-scoped list built by
  `LegalDraftEngine.history`, so a draft the user may not touch is never in
  the candidate list to be chosen at all.
* **No business logic.** Every action is a call into
  `app/services/draft_management.py` or `LegalDraftEngine` -- the same
  functions the REST routes call, including the same
  `ensure_draft_access` ownership check.
"""

import re
from typing import Any

from app.chatops import intents, selection
from app.chatops.artifacts import draft_artifact
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.registry import register_workflow
from app.drafting.engine import LegalDraftEngine
from app.schemas.drafting import DraftHistoryRequest, DraftTranslateRequest
from app.services import draft_management

# The closed set of actions this workflow can perform, each mapped to the
# vocabulary that selects it. Nothing outside this table can be reached: the
# action is chosen by matching a pattern, never by a model naming a method.
_ACTIONS: tuple[tuple[str, str], ...] = (
    ("delete", r"delete|remove|discard|hata|mita|हटा|मिटा|डिलीट"),
    ("approve", r"approve|finali[sz]e|manzoor|मंज़ूर|मंजूर|स्वीकृत"),
    ("unlock", r"unlock|अनलॉक"),
    ("lock", r"\block\b|लॉक"),
    ("rollback", r"rollback|roll\s*back|revert|restore|wapas|वापस|पिछले\s*संस्करण"),
    ("duplicate", r"duplicate|copy|clone|प्रतिलिपि|कॉपी"),
    ("compare", r"compare|diff|difference|तुलना|अंतर"),
    ("versions", r"versions?|version\s*history|history|संस्करण"),
    ("translate", r"translate|anuvad|अनुवाद|hindi\s*me|में\s*बदल"),
    ("review", r"review|samiksha|समीक्षा|jaanch|जांच"),
    ("open", r"open|show|resume|continue|kholo|खोल"),
)

# Actions that change or destroy something, and therefore may not run on a
# bare request. Reading a version list is not in this set; deleting the
# draft that list belongs to is.
_NEEDS_CONFIRMATION = frozenset({"delete", "approve", "lock", "unlock", "rollback"})

_ACTION_LABELS: dict[str, str] = {
    "delete": "delete this draft and its version history",
    "approve": "mark this draft approved",
    "lock": "lock this draft against further edits",
    "unlock": "unlock this draft for editing",
    "rollback": "roll this draft back to its previous version",
}


def _detect_action(message: str) -> str | None:
    for action, pattern in _ACTIONS:
        if re.search(pattern, message or "", re.IGNORECASE):
            return action
    return None


@register_workflow
class DraftManagementWorkflow(ChatWorkflow):
    """"Delete the rent notice", "approve my draft", "compare versions"."""

    name = "draft_manage"
    title = "draft management"
    may_interrupt_draft = True

    def matches_intent(self, message: str, language: str) -> float:
        # A request to merely show the saved-draft list is owned by the
        # dedicated listing workflow. Indic "show" forms overlap the weak
        # draft-management vocabulary, so resolve the semantic distinction
        # here instead of depending on registration order.
        action = _detect_action(message)
        if intents.SAVED_DRAFTS.search(message) and action in {None, "open"}:
            return 0.0
        # Above `saved_drafts`' 0.9 on purpose: "delete my draft" names both
        # a listing verb and a management verb, and the management one is
        # what the user asked for.
        return intents.score(intents.DRAFT_MANAGE, message, 0.95)

    def needs_confirmation(self, context: WorkflowContext) -> bool:
        return str(context.facts.get("action", "")) in _NEEDS_CONFIRMATION

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        if not context.facts.get("action"):
            action = _detect_action(context.message)
            if action:
                facts["action"] = action

        if context.facts.get("draft_id"):
            return facts

        # A draft the conversation is already working on needs no picking.
        parked = selection.from_dicts(list(context.facts.get("_choices") or []))
        if parked:
            chosen = selection.resolve(context.message, parked)
            if chosen is not None:
                facts["draft_id"] = chosen.key
                facts["draft_label"] = chosen.label
                return facts
            return facts

        if context.memory.get("draft_id") and facts.get("action", context.facts.get("action")) != "open":
            facts["draft_id"] = context.memory["draft_id"]
            return facts

        # A named/numbered saved draft takes precedence over the current one.
        choices = await self._candidates(context)
        chosen = selection.resolve(context.message, choices)
        if chosen:
            facts.update(draft_id=chosen.key, draft_label=chosen.label)
        elif selection.overlapping(context.message, choices):
            facts["_choices"] = [choice.as_dict() for choice in choices]
        elif len(choices) == 1:
            facts.update(draft_id=choices[0].key, draft_label=choices[0].label)
        elif choices:
            facts["_choices"] = [choice.as_dict() for choice in choices]

        return facts

    async def _candidates(self, context: WorkflowContext) -> list[selection.Choice]:
        """The user's own drafts, newest first. Owner-scoped by the service."""
        engine = LegalDraftEngine()
        request = (
            DraftHistoryRequest(user_id=context.authenticated_user_id, limit=15)
            if context.authenticated_user_id
            else DraftHistoryRequest(session_id=context.session_id, limit=15)
        )
        drafts = await engine.history(request)
        return [
            selection.Choice(
                key=draft.draft_id,
                label=draft.template_name,
                detail=f"{draft.status}, {draft.created_at[:10]}",
            )
            for draft in drafts
        ]

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["action", "draft_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        if missing[0] == "action":
            return (
                "What would you like to do with it — open it, see its versions, compare versions, "
                "translate it, duplicate it, approve it, lock it, or delete it?"
            )
        choices = selection.from_dicts(list(context.facts.get("_choices") or []))
        if choices:
            return "Which draft did you mean?\n\n" + selection.render(choices)
        return (
            "I could not find a saved draft to work with. Say \"show my drafts\" to see what you have, "
            "or tell me what document you need and I will start one."
        )

    def summarize_for_confirmation(self, context: WorkflowContext) -> str:
        action = str(context.facts.get("action", ""))
        label = str(context.facts.get("draft_label") or "your draft")
        described = _ACTION_LABELS.get(action, action)
        note = (
            "\n\nThis cannot be undone — the draft and every earlier version of it are erased."
            if action == "delete" else ""
        )
        return f"- Draft: {label}\n- Action: {described}{note}"

    def idempotency_key(self, context: WorkflowContext) -> str:
        # Identity is the action and the draft, not the whole fact bag: a
        # second "delete it" for the same draft is the same request even if
        # the label was resolved differently on the way in.
        return f"{self.name}:{context.facts.get('action')}:{context.facts.get('draft_id')}"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        action = str(context.facts.get("action", ""))
        draft_id = str(context.facts.get("draft_id", ""))
        user_id = context.authenticated_user_id
        session_id = context.session_id
        engine = LegalDraftEngine()

        # Ownership first, always, through the same service function the REST
        # routes use. Raises ForbiddenError/NotFoundError, both of which the
        # orchestrator turns into a plain refusal.
        draft = await draft_management.owned_draft(engine, draft_id, user_id, session_id)
        label = str(context.facts.get("draft_label") or draft.get("draft_type", "your draft"))

        if action == "delete":
            result = await draft_management.delete_draft(draft_id, user_id, session_id)
            return WorkflowTurn(
                message=(
                    f"Deleted **{label}**, along with {result['versions_deleted']} saved version(s). "
                    "Nothing about it remains."
                ),
                status="completed", finished=True,
            )
        if action == "approve":
            lifecycle = await engine.approve(draft_id)
            return WorkflowTurn(
                message=f"**{label}** is now `{lifecycle}`. You can still download it or lock it.",
                status="completed", allowed_actions=["download as PDF", "lock it"], finished=True,
            )
        if action == "lock":
            lifecycle = await engine.lock(draft_id)
            return WorkflowTurn(
                message=f"**{label}** is `{lifecycle}` — no further edits until you unlock it.",
                status="completed", allowed_actions=["unlock it", "download as PDF"], finished=True,
            )
        if action == "unlock":
            lifecycle = await engine.unlock(draft_id)
            return WorkflowTurn(
                message=f"**{label}** is `{lifecycle}` and editable again.",
                status="completed", allowed_actions=["edit it", "lock it"], finished=True,
            )
        if action == "versions":
            versions = await engine.list_versions(draft_id)
            if not versions:
                return WorkflowTurn(
                    message=f"**{label}** has no saved versions yet.", status="completed", finished=True
                )
            lines = "\n".join(
                f"- v{version['version_number']} · {version.get('language', 'english')}"
                + (f" · {version['created_at']:%d %b %Y}" if version.get("created_at") else "")
                + (f" · {version['note']}" if version.get("note") else "")
                for version in versions
            )
            return WorkflowTurn(
                message=f"**{label}** — {len(versions)} version(s):\n\n{lines}",
                status="completed",
                allowed_actions=["compare versions", "roll back"],
                finished=True,
            )
        if action == "compare":
            versions = await engine.list_versions(draft_id)
            if len(versions) < 2:
                return WorkflowTurn(
                    message=f"**{label}** has only one version, so there is nothing to compare yet.",
                    status="completed", finished=True,
                )
            newest = max(int(version["version_number"]) for version in versions)
            previous = sorted({int(v["version_number"]) for v in versions})[-2]
            comparison = await draft_management.compare_versions(
                draft_id, previous, newest, user_id, session_id
            )
            changes = [
                line for line in comparison.unified_diff.splitlines()
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
            ]
            body = "\n".join(changes[:20]) or "The two versions are identical."
            more = f"\n\n…and {len(changes) - 20} more changed lines." if len(changes) > 20 else ""
            return WorkflowTurn(
                message=(
                    f"**{label}** — v{previous} compared with v{newest}:\n\n```diff\n{body}\n```{more}"
                ),
                status="completed", allowed_actions=["roll back", "download as PDF"], finished=True,
            )
        if action == "rollback":
            versions = await engine.list_versions(draft_id)
            numbers = sorted({int(version["version_number"]) for version in versions})
            if len(numbers) < 2:
                return WorkflowTurn(
                    message=f"**{label}** has no earlier version to roll back to.",
                    status="completed", finished=True,
                )
            await engine.rollback(draft_id, numbers[-2])
            return WorkflowTurn(
                message=f"**{label}** is back to version {numbers[-2]}. The later version is still in the history.",
                status="completed", allowed_actions=["show versions", "download as PDF"], finished=True,
            )
        if action == "duplicate":
            copy = await draft_management.duplicate_draft(draft_id, user_id, session_id)
            context.memory["draft_id"] = copy.draft_id
            return WorkflowTurn(
                message=(
                    f"Made a copy of **{label}**. The copy is now the draft I will work on, "
                    "and the original is untouched."
                ),
                status="completed", allowed_actions=["edit it", "download as PDF"], finished=True,
            )
        if action == "translate":
            target = self._requested_language(context)
            translated = await engine.translate(
                DraftTranslateRequest(draft_id=draft_id, target_language=target, session_id=session_id)
            )
            return WorkflowTurn(
                message=(
                    f"**{label}** in {target.title()}:\n\n{translated.translated_text[:1500]}\n\n"
                    "Names, numbers, Act and section references are kept exactly as they were."
                ),
                status="completed",
                artifact=draft_artifact(draft_id, "pdf", label=f"Download {label} (PDF)"),
                finished=True,
            )
        if action == "review":
            review = await draft_management.build_review(draft_id, user_id, session_id)
            missing = "\n".join(f"- {item}" for item in review.missing_information[:8]) or "- nothing outstanding"
            conflicts = (
                "\n\n**Conflicting facts you must resolve before export:**\n"
                + "\n".join(f"- {item.get('label', item.get('slot'))}" for item in review.conflicts)
                if review.conflicts else ""
            )
            return WorkflowTurn(
                message=(
                    f"**{label}** — `{review.lifecycle_state}`\n\n**Still missing:**\n{missing}{conflicts}"
                ),
                status="completed",
                warnings=["Final export is blocked until the conflicts are resolved."]
                if review.final_export_blocked else [],
                allowed_actions=["fix a detail", "download as PDF"],
                finished=True,
            )

        current = await engine.get_current(draft_id)
        from app.drafting.conversation import DraftConversationEngine

        conversation = DraftConversationEngine()
        if context.memory.get("draft_id") != draft_id:
            conversation._park_current_draft(context.memory)
        lifecycle = draft.get("lifecycle_state", "preview_ready")
        context.memory.update(
            draft_id=draft_id, draft_mode=True,
            draft_stage=lifecycle if lifecycle in {"locked", "approved", "exported"} else "preview",
            draft_template_id=draft["draft_type"],
            draft_fields=dict(draft.get("fields") or {}),
            draft_language=draft.get("language") or context.language,
            draft_paused=False, draft_awaiting_unlock_confirm=False,
        )
        return WorkflowTurn(
            message=(
                f"**{label}**\n\n{current.full_text[:1500]}\n\n"
                "Tell me what to change, or ask for it as a PDF."
            ),
            status="completed",
            allowed_actions=["edit it", "download as PDF", "show versions"],
            finished=True,
        )

    @staticmethod
    def _requested_language(context: WorkflowContext) -> str:
        from app.language.detector import extract_requested_language

        requested = extract_requested_language(context.message)
        return requested or ("english" if context.language == "english" else context.language)
