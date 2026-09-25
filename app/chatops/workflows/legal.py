"""Phase-2 legal workflows and draft management, reachable by talking.

These previously each needed their own Streamlit page or sidebar panel. Every
one is a thin adapter over the existing service -- the cyber-fraud action
plan, the jurisdiction rules, the evidence tables and the timeline all still
come from `Phase2WorkflowService`, which is deterministic and whose legal
content is not model-generated. Nothing here re-derives any of it.
"""

import re
from typing import Any, cast

from app.chatops import intents
from app.chatops.artifacts import draft_artifact
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.registry import register_workflow
from app.drafting.engine import LegalDraftEngine
from app.schemas.phase2 import CyberFraudRequest, JurisdictionRequest, SupportedWorkflowLanguage
from app.services.phase2_workflow import cyber_fraud_workflow, jurisdiction_check

_EXPORT_FORMATS = ("pdf", "docx", "txt", "rtf")


# `SupportedWorkflowLanguage` is a narrow Literal in `schemas/phase2.py` --
# the Phase-2 service ships reviewed, non-model-generated text in only those
# languages. A chat in a language it does not cover falls back to English
# TEXT here rather than being refused; the surrounding chat reply is still
# composed in the user's own language.
_WORKFLOW_LANGUAGES = frozenset({"english", "hindi", "hinglish"})


def _workflow_language(language: str) -> SupportedWorkflowLanguage:
    """Narrows an arbitrary detected language to the three the Phase 2
    workflows accept, which is what their request schemas declare."""
    if language in _WORKFLOW_LANGUAGES:
        return cast(SupportedWorkflowLanguage, language)
    return "english"


# Matter-type vocabulary for the jurisdiction workflow, in the languages
# users actually name these matters in.
_MATTER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("cyber_complaint", re.compile(r"cyber|online\s*fraud|upi|otp|साइबर|ऑनलाइन", re.IGNORECASE)),
    ("consumer_complaint", re.compile(r"consumer|refund|defect|warranty|उपभोक्ता|सामान", re.IGNORECASE)),
    ("rti", re.compile(r"\brti\b|right\s*to\s*information|सूचना\s*का\s*अधिकार", re.IGNORECASE)),
    ("property", re.compile(r"property|land|rent|tenant|landlord|संपत्ति|ज़मीन|जमीन|किराया", re.IGNORECASE)),
    ("police_complaint", re.compile(r"police|fir|theft|chori|पुलिस|प्राथमिकी|चोरी|शिकायत", re.IGNORECASE)),
)


def _detect_matter_type(message: str) -> str | None:
    for matter, pattern in _MATTER_PATTERNS:
        if pattern.search(message or ""):
            return matter
    return None


@register_workflow
class CyberFraudWorkflow(ChatWorkflow):
    """Cyber-fraud emergency triage.

    Time-critical: the golden hour for freezing a fraudulent transfer is
    short, so this asks for the minimum (what happened) and returns the
    official reporting channels immediately rather than completing a long
    form first.
    """

    name = "cyber_fraud"
    title = "cyber-fraud emergency workflow"

    def matches_intent(self, message: str, language: str) -> float:
        # Topic AND an explicit request to act. "cyber fraud kya hota hai" is
        # a question and "I lost money in an online cyber fraud" is a
        # situation report -- both deserve a grounded legal answer from the
        # normal pipeline, not to be pulled into an emergency form.
        if not intents.wants_action(message):
            return 0.0
        return intents.score(intents.CYBER_FRAUD, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        message = context.message
        if not context.facts.get("description") and len(message.split()) >= 4:
            facts["description"] = message.strip()
        amount = re.search(r"(?:rs\.?|inr|₹)\s*([0-9][0-9,]*)", message, re.IGNORECASE)
        if amount:
            facts["amount"] = amount.group(1)
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["description"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return (
            "I can help you report this straight away. In a sentence or two: what happened, "
            "and roughly how much money is involved?"
        )

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        result = cyber_fraud_workflow(
            CyberFraudRequest(
                narrative=str(context.facts.get("description", "")),
                amount=context.facts.get("amount"),
                language=_workflow_language(context.language),
            )
        )
        steps = "\n".join(f"{index}. {step}" for index, step in enumerate(result.immediate_steps, start=1))
        channels = "\n".join(f"- {source.title}: {source.url}" for source in result.sources)
        # Contradictions block export in the service; surfaced here so the
        # user resolves them rather than discovering it at download time.
        warnings = [result.warning] if result.warning else []
        if result.contradictions:
            warnings.append(
                "I found details that conflict with each other — please confirm which is correct "
                "before this is filed."
            )
        still_needed = [
            item.get("label", item.get("field", ""))
            for item in result.required_information
            if item.get("required")
        ][:3]
        follow_up = (
            f"\n\nTo complete a formal complaint I still need: {', '.join(still_needed)}."
            if still_needed else ""
        )
        return WorkflowTurn(
            message=(
                f"**Act now — {result.detected_fraud_type.replace('_', ' ')} "
                f"(risk: {result.risk_level})**\n\n{steps}\n\n"
                f"**Official channels**\n{channels}{follow_up}\n\n"
                "Would you like me to draft a written cyber-crime complaint you can file?"
            ),
            status="completed",
            citations=[
                {"title": source.title, "url": source.url, "authority": source.authority}
                for source in result.sources
            ],
            warnings=warnings,
            allowed_actions=["draft a cyber crime complaint", "organise my evidence"],
            finished=True,
        )



# "Agar notice ke baad bhi payment na mile, complaint kab aur kahan file
# karni hogi?" asks how the RULE works in general (a conditional "agar ...
# to" framing, or "kab" asking about timing), not a request to locate a
# forum for the user's own case. Left unmatched, this workflow only
# started for a genuinely actionable jurisdiction request.
_HYPOTHETICAL_PROCEDURE = re.compile(
    r"\bagar\b.*\b(?:kab|kya|kaise)\b|\bkab\b.*\bfile\b"
    r"|अगर.*(?:कब|क्या|कैसे)|कब.*फाइल",
    re.IGNORECASE,
)

# A candidate "location" that is actually the question the user just asked
# ("Complaint kab aur kahan file karni hogi?") starts with an interrogative
# word rather than naming a place.
_QUESTION_WORD_START = re.compile(
    r"^\s*(?:what|why|how|when|where|which|who|is|are|can|does|do|should|agar|kya|kab|kahan|kaise)\b",
    re.IGNORECASE,
)


@register_workflow
class JurisdictionWorkflow(ChatWorkflow):
    """"Kahan complaint karun?" — which forum, which police station."""

    name = "jurisdiction"
    title = "jurisdiction check"

    def matches_intent(self, message: str, language: str) -> float:
        # "What is territorial jurisdiction?" is a legal question; only a
        # request to work out WHERE TO FILE starts this workflow.
        if _HYPOTHETICAL_PROCEDURE.search(message or ""):
            return 0.0
        if not intents.wants_action(message):
            return 0.0
        return intents.score(intents.JURISDICTION, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        if not context.facts.get("matter_type"):
            matter = _detect_matter_type(context.message)
            if matter:
                facts["matter_type"] = matter
        # A location the user mentions is taken as their own, which is the
        # ordinary reading of "main Delhi me hoon". The service treats every
        # basis as tentative and tells the user to verify before filing, so a
        # misread here cannot become a confident wrong answer. But the
        # message that STARTS the workflow is often the question itself
        # ("...complaint kahan file karni hogi?"), not a location -- taking
        # it verbatim previously stored the whole question as the
        # complainant's location. A candidate containing "?" or a leading
        # question word is never a location, no matter its word count.
        location_candidate = context.message.strip()
        looks_like_question = bool(
            "?" in location_candidate or _QUESTION_WORD_START.match(location_candidate)
        )
        if (
            not context.facts.get("complainant_location")
            and len(location_candidate.split()) >= 3
            and not looks_like_question
        ):
            facts["complainant_location"] = location_candidate
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["matter_type"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return (
            "What kind of matter is this — a police complaint, a consumer complaint, an RTI, "
            "a property dispute, or a cyber-crime complaint?"
        )

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        result = jurisdiction_check(
            JurisdictionRequest(
                matter_type=str(context.facts.get("matter_type", "police_complaint")),  # type: ignore[arg-type]
                complainant_location=context.facts.get("complainant_location"),
                respondent_location=context.facts.get("respondent_location"),
                incident_location=context.facts.get("incident_location"),
                language=_workflow_language(context.language),
            )
        )
        bases = "\n".join(f"- {basis}" for basis in result.possible_bases)
        missing = (
            "\n\n**Still needed for a firmer answer:** " + ", ".join(result.missing_facts)
            if result.missing_facts else ""
        )
        return WorkflowTurn(
            message=(
                f"**Tentative forum:** {result.tentative_forum}\n"
                f"_(confidence: {result.confidence})_\n\n{bases}{missing}"
            ),
            status="completed",
            warnings=[result.warning] if result.warning else [],
            finished=True,
        )


@register_workflow
class SavedDraftsWorkflow(ChatWorkflow):
    """"Meri saved drafts dikhao" — replaces the sidebar panel."""

    name = "saved_drafts"
    title = "saved drafts"
    may_interrupt_draft = True

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.SAVED_DRAFTS, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        from app.chatops import selection
        from app.chatops.workflows.drafts import DraftManagementWorkflow

        choices = selection.from_dicts(context.facts.get("_choices") or [])
        if not choices:
            choices = await DraftManagementWorkflow()._candidates(context)
            return {"_choices": [choice.as_dict() for choice in choices]}
        chosen = selection.resolve(context.message, choices)
        return {"draft_id": chosen.key, "draft_label": chosen.label} if chosen else {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["draft_id"] if context.facts.get("_choices") else []

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        from app.chatops import selection

        choices = selection.from_dicts(context.facts.get("_choices") or [])
        return "Here are your saved drafts:\n\n" + selection.render(choices) + "\n\nWhich draft would you like to open?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        from app.chatops.workflows.drafts import DraftManagementWorkflow

        if not context.facts.get("draft_id"):
            return WorkflowTurn(
                message="You do not have any saved drafts yet. Tell me what document you need and I will start one.",
                status="completed", finished=True,
            )
        context.facts["action"] = "open"
        return await DraftManagementWorkflow().execute(context)


@register_workflow
class DraftExportWorkflow(ChatWorkflow):
    """"PDF me download karna hai" — export the active draft, inline."""

    name = "draft_export"
    title = "draft download"
    may_interrupt_draft = True

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.DRAFT_EXPORT, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        lowered = context.message.lower()
        for fmt in _EXPORT_FORMATS:
            if fmt in lowered or (fmt == "docx" and "word" in lowered):
                facts["format"] = fmt
                break
        draft_id = context.facts.get("draft_id") or context.memory.get("draft_id")
        if draft_id:
            facts["draft_id"] = draft_id
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["draft_id", "format"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        if missing[0] == "draft_id":
            return "Which draft would you like to download? Say \"show my drafts\" if you want the list."
        return "Which format would you like — PDF, DOCX, TXT, or RTF?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        draft_id = str(context.facts["draft_id"])
        fmt = str(context.facts["format"])
        engine = LegalDraftEngine()
        draft = await engine.drafts.find_by_id(draft_id)
        if draft is None:
            return WorkflowTurn(message="That draft is no longer available.", status="failed", finished=True)
        # Ownership is re-checked by `/draft/export` when the link is used;
        # this only refuses obviously foreign drafts early, for a clearer message.
        owner = draft.get("user_id")
        if owner and context.authenticated_user_id and owner != context.authenticated_user_id:
            return WorkflowTurn(message="That draft is no longer available.", status="failed", finished=True)
        return WorkflowTurn(
            message=f"Your {fmt.upper()} is ready — the download link is below.",
            status="completed",
            artifact=draft_artifact(draft_id, fmt),
            finished=True,
        )
