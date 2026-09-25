"""Notarization capabilities, reachable by talking.

SAFETY BOUNDARY -- read before changing anything here.

When a user says "mujhe notary karani hai", they mean "make this notarized".
This product cannot do that, and must not imply it can. Every workflow below
therefore states, on its FIRST turn, what it will actually do: prepare the
document, collect consent, and submit it to a verified human notary for
review.

None of these workflows can reach the `notarized` state. They call
`NotarizationService`, whose `approve_request()` is the only path there and
which requires a verified, active notary account plus a fresh
re-authentication token. A normal user's chat session has no way to obtain
one. That is enforced in the service, not here -- these are adapters.

Nothing here fabricates a stamp, signature, registration number, certificate,
approval, or notarization record.
"""

from typing import Any

from app.chatops import intents
from app.chatops.artifacts import notarized_artifact
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.registry import register_workflow
from app.core import clock
from app.core.security import Role
from app.drafting.engine import LegalDraftEngine
from app.drafting.templates import get_template
from app.notarization import states
from app.notarization.checklist import ELIGIBLE_CATEGORIES
from app.notarization.checklist import evaluate as evaluate_checklist
from app.notarization.service import NotarizationService
from app.repositories.notarization import NotaryAccountRepository
from app.schemas.notarization import STANDARD_WARNINGS

# Said on the first turn of every preparation. The wording is deliberate: it
# names what the product does AND what it does not, before the user invests
# any effort.
_BOUNDARY_NOTICE = (
    "I can prepare your document for notarization and submit it to a verified notary for review. "
    "I do not issue notarial stamps, signatures, or approvals myself — only a licensed notary can do that."
)


@register_workflow
class PrepareNotarizationWorkflow(ChatWorkflow):
    """Draft → notary-ready → (consent) → request for human review."""

    name = "notarization_prepare"
    may_interrupt_draft = True
    title = "notarization preparation"
    requires_confirmation = True

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.NOTARIZATION_PREPARE, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        # Reuse what the conversation already knows rather than asking again.
        if not context.facts.get("draft_id"):
            draft_id = context.memory.get("draft_id")
            if draft_id:
                facts["draft_id"] = draft_id
        # A document uploaded in this conversation is the obvious subject.
        if not context.facts.get("document_ref") and context.document_ids:
            facts["document_ref"] = context.document_ids[0]

        message = context.message.strip()
        asked = (context.memory.get("chatops_stack") or [{}])[-1].get("asked", [])
        # Free-text answers are attributed to whichever field was just asked
        # for -- the orchestrator asks one at a time, so this is unambiguous.
        if asked and message:
            last_asked = asked[-1]
            if last_asked in {"full_name", "address", "place", "identity_document_type"} and not context.facts.get(last_asked):
                facts[last_asked] = message
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        # A subject document first; identity details only once we have one.
        if not context.facts.get("draft_id"):
            return ["draft_id"]
        return ["full_name", "address", "identity_document_type", "place"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        prefix = "" if context.facts else f"{_BOUNDARY_NOTICE}\n\n"
        questions = {
            "draft_id": (
                "Which document would you like prepared for notarization? "
                "If it is not in this conversation yet, please upload it or ask me to draft it first."
            ),
            "full_name": "What is the full name of the person who will sign the document?",
            "address": "What is the signer's address?",
            "identity_document_type": (
                "Which identity document will the signer present to the notary — "
                "Aadhaar, PAN, Passport, Voter ID, or Driving Licence? "
                "(I record only the type; never the number.)"
            ),
            "place": "Where will the document be signed (city/town)?",
        }
        return prefix + questions.get(missing[0], f"Please provide: {missing[0].replace('_', ' ')}.")

    async def validate(self, context: WorkflowContext) -> list[str]:
        draft_id = context.facts.get("draft_id")
        if not draft_id:
            return []
        engine = LegalDraftEngine()
        draft = await engine.drafts.find_by_id(str(draft_id))
        if draft is None:
            return ["I could not find that document any more. Which document should I prepare?"]
        template = get_template(draft.get("draft_type", ""))
        if template is None:
            return []
        readiness = evaluate_checklist(template.category, {}, [])
        if not readiness.eligible:
            # Told plainly, with what IS eligible, rather than a bare refusal.
            return [
                (
                    f"{readiness.ineligible_reason} "
                    f"Notarization preparation applies to: {', '.join(sorted(ELIGIBLE_CATEGORIES))}."
                )
            ]
        return []

    def summarize_for_confirmation(self, context: WorkflowContext) -> str:
        facts = context.facts
        return (
            f"- Signer: {facts.get('full_name', '')}\n"
            f"- Address: {facts.get('address', '')}\n"
            f"- Identity document type: {facts.get('identity_document_type', '')} (number not recorded)\n"
            f"- Place of signing: {facts.get('place', '')}\n\n"
            "I will prepare a **Notary-ready draft**. This does not notarize it — "
            "a verified notary must review and approve it."
        )

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:

        engine = LegalDraftEngine()
        draft = await engine.drafts.find_by_id(str(context.facts["draft_id"]))
        if draft is None:
            return WorkflowTurn(message="That document is no longer available.", status="failed", finished=True)
        template = get_template(draft["draft_type"])
        if template is None:
            return WorkflowTurn(message="That document type is not recognised.", status="failed", finished=True)

        service = NotarizationService()
        result = await service.prepare(
            draft_id=str(context.facts["draft_id"]),
            draft_title=draft.get("template_name", template.name),
            draft_category=template.category,
            sections=draft.get("sections", {}),
            signer={
                "full_name": context.facts.get("full_name", ""),
                "address": context.facts.get("address", ""),
                "identity_document_type": context.facts.get("identity_document_type", ""),
                "place": context.facts.get("place", ""),
                "date": clock.today().strftime("%d/%m/%Y"),
                "email": context.facts.get("email", ""),
            },
            witnesses=[],
            user_id=context.authenticated_user_id or "",
        )
        readiness = result["readiness"]
        document = result["document"]
        # Remembered so follow-up turns ("status kya hai", "e-sign shuru karo")
        # do not have to ask which document.
        context.memory["notarization_document_id"] = str(document["_id"])

        checklist_lines = "\n".join(
            f"{'✅' if item['satisfied'] else '⬜'} {item['label']}" for item in readiness.as_dicts()
        )
        return WorkflowTurn(
            message=(
                f"Your **{readiness.label}** is ready.\n\n"
                f"**Checklist**\n{checklist_lines}\n\n"
                f"Document fingerprint (SHA-256): `{result['document_hash']}`\n\n"
                "Next, I can start the e-signature step. Nothing is signed until you explicitly consent."
            ),
            status="completed",
            collected_facts=dict(context.facts),
            warnings=list(STANDARD_WARNINGS),
            allowed_actions=["start e-sign", "check status", "cancel"],
            finished=True,
        )


@register_workflow
class NotarizationStatusWorkflow(ChatWorkflow):
    """"Mera notarization ka status kya hai?" — reads the real state machine."""

    name = "notarization_status"
    may_interrupt_draft = True
    title = "notarization status check"

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.NOTARIZATION_STATUS, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        document_id = context.facts.get("document_id") or context.memory.get("notarization_document_id")
        return {"document_id": document_id} if document_id else {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["document_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return (
            "I do not have a document in this conversation yet. "
            "Which document's notarization status would you like to check?"
        )

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        service = NotarizationService()
        result = await service.document_status(
            document_id=str(context.facts["document_id"]),
            user_id=context.authenticated_user_id or "",
        )
        timeline = "\n".join(
            f"- {event['action'].replace('_', ' ').title()} ({event.get('actor_role', 'system')})"
            for event in result["timeline"][-8:]
        )
        artifact = None
        if result["downloadable"]:
            artifact = notarized_artifact(str(context.facts["document_id"]))
        return WorkflowTurn(
            message=(
                f"**Status:** {result['status_label']}\n\n"
                f"{timeline}\n\n"
                + (
                    "The notarized document is available to download below."
                    if result["downloadable"]
                    else "The final notarized document becomes available only after a verified notary approves it."
                )
            ),
            status="completed",
            artifact=artifact,
            warnings=list(STANDARD_WARNINGS),
            finished=True,
        )


@register_workflow
class VerifyDocumentWorkflow(ChatWorkflow):
    """Public verification by token. No login needed — that is the point."""

    name = "notarization_verify"
    may_interrupt_draft = True
    title = "document verification"

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.NOTARIZATION_VERIFY, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        import re

        # A verification token is a long URL-safe string; pick it out of the
        # message or out of a pasted verification URL.
        match = re.search(r"(?:/verify/)?([A-Za-z0-9_-]{22,})", context.message)
        return {"verification_token": match.group(1)} if match else {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["verification_token"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return (
            "Please paste the verification code from the document's QR code "
            "(or the full verification link)."
        )

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        service = NotarizationService()
        result = await service.verify(str(context.facts["verification_token"]))
        if not result.get("found"):
            return WorkflowTurn(
                message=result.get("message", "No notarized document matches that verification code."),
                status="completed",
                finished=True,
            )
        if result.get("revoked"):
            headline = "⛔ **This notarization has been REVOKED** and must not be relied upon."
        elif result.get("status") == states.NOTARIZED:
            headline = "✅ **Notarized** — verified against our record."
        else:
            headline = f"⚠️ This document is **not notarized** (status: {result.get('status_label', '')})."
        return WorkflowTurn(
            message=(
                f"{headline}\n\n"
                f"- Document type: {result.get('document_type', '—')}\n"
                f"- Notarized on: {result.get('notarized_at', '—')}\n"
                f"- Notary: {result.get('notary_name', '—')}\n"
                f"- Registration number: {result.get('notary_registration_number', '—')}\n\n"
                + (result.get("message") or "")
            ),
            status="completed",
            finished=True,
        )


@register_workflow
class NotaryQueueWorkflow(ChatWorkflow):
    """Verified-notary review queue, in chat.

    `required_role` gates the ROLE; the service additionally requires a
    verified, active notary ACCOUNT, so a role claim alone is not enough.
    """

    name = "notary_queue"
    title = "notary review queue"
    required_role = Role.lawyer

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.NOTARY_QUEUE, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return []

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return "Which request would you like to review?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        service = NotarizationService()
        # Verified-notary check happens in the service, which raises
        # ForbiddenError the orchestrator surfaces as-is.
        notary = await service._require_verified_notary(context.authenticated_user_id)
        pending = await service.requests.list_for_notary(str(notary["_id"]), None, 25)
        if not pending:
            return WorkflowTurn(message="You have no notarization requests waiting.", status="completed", finished=True)
        lines = "\n".join(
            f"- **{request.get('document_title', 'Document')}** — {request.get('review_status')} "
            f"(signer: {request.get('signer', {}).get('full_name', '—')}, "
            f"v{request.get('document_version', 1)})"
            for request in pending
        )
        return WorkflowTurn(
            message=(
                f"You have {len(pending)} request(s):\n\n{lines}\n\n"
                "Approving or rejecting requires re-authentication and confirmation of the document hash. "
                "Tell me which request you want to open."
            ),
            status="completed",
            allowed_actions=["start review", "approve", "reject"],
            finished=True,
        )


@register_workflow
class NotaryAdminWorkflow(ChatWorkflow):
    """Admin notary-account management and audit visibility, in chat."""

    name = "notary_admin"
    title = "notary administration"
    required_role = Role.admin

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.NOTARY_ADMIN, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        message = context.message.lower()
        if "audit" in message:
            return {"action": "audit"}
        if "failed" in message:
            return {"action": "failed"}
        return {"action": "accounts"}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["action"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return "Would you like to see notary accounts, the audit log, or failed attempts?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        from app.repositories.notarization import NotarizationAuditRepository

        action = context.facts.get("action", "accounts")
        if action == "audit":
            events = await NotarizationAuditRepository().list_recent(25)
            body = "\n".join(
                f"- `{event.get('occurred_at')}` **{event.get('action')}** ({event.get('actor_role')})"
                for event in events
            ) or "No audit events recorded yet."
            return WorkflowTurn(
                message=f"**Notarization audit log** (most recent first)\n\n{body}\n\n"
                        "_Audit events are append-only and cannot be edited or deleted._",
                status="completed", finished=True,
            )
        notaries = await NotaryAccountRepository().list_all(50)
        if not notaries:
            return WorkflowTurn(message="No notary accounts are registered yet.", status="completed", finished=True)
        body = "\n".join(
            f"- **{notary.get('full_name')}** — Reg. {notary.get('registration_number')} "
            f"({notary.get('jurisdiction_state')}) · {notary.get('verification_status')} · "
            f"{'active' if notary.get('active') else 'inactive'}"
            for notary in notaries
        )
        return WorkflowTurn(
            message=(
                f"**Notary accounts**\n\n{body}\n\n"
                "Verifying an account is an assertion that you have checked the notary's licence "
                "against the relevant register. Tell me which account to verify or revoke."
            ),
            status="completed",
            allowed_actions=["verify account", "revoke account"],
            finished=True,
        )
