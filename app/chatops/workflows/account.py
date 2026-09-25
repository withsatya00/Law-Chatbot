"""Account self-service, by talking.

Preferences, the download centre, background jobs and data erasure. All four
were REST-only; none of them needed a page, they needed a sentence.

The destructive one is deliberately the most careful thing in this package:
erasing an account's data asks for a typed phrase, not a "yes", because a
yes is one keystroke and this cannot be undone.
"""

import re
from typing import Any

from app.chatops import intents, selection
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.registry import register_workflow
from app.core.security import Role
from app.schemas.phase3 import UserPreferencesRequest
from app.services.phase3 import BackgroundJobService, PreferenceService

# The exact words that authorise account erasure. Typed in full, in the
# user's own language -- a bare "yes" is not enough for an irreversible act
# that spans every session the account ever had.
_ERASE_PHRASES = ("delete my data", "erase my data", "मेरा डेटा हटाओ", "mera data delete karo")

_LANGUAGE_WORDS = {
    "english": r"\benglish\b|अंग्रेज़ी|अंग्रेजी",
    "hindi": r"\bhindi\b|हिंदी|हिन्दी",
    "hinglish": r"\bhinglish\b|हिंग्लिश",
    "marathi": r"\bmarathi\b|मराठी",
    "gujarati": r"\bgujarati\b|ગુજરાતી",
    "bengali": r"\bbengali\b|বাংলা",
    "tamil": r"\btamil\b|தமிழ்",
    "telugu": r"\btelugu\b|తెలుగు",
    "kannada": r"\bkannada\b|ಕನ್ನಡ",
    "malayalam": r"\bmalayalam\b|മലയാളം",
    "punjabi": r"\bpunjabi\b|ਪੰਜਾਬੀ",
    "odia": r"\bodia\b|ଓଡ଼ିଆ",
    "urdu": r"\burdu\b|اردو",
}


@register_workflow
class PreferencesWorkflow(ChatWorkflow):
    """"Show my settings", "always answer me in Hindi", "default to DOCX"."""

    name = "preferences"
    title = "your preferences"
    required_role = Role.user

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.PREFERENCES, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        message = context.message.lower()
        for name, pattern in _LANGUAGE_WORDS.items():
            if re.search(pattern, context.message, re.IGNORECASE):
                facts["language"] = name
                break
        for mode in ("simple", "detailed", "advocate"):
            if re.search(rf"\b{mode}\b", message):
                facts["explanation_mode"] = mode
                break
        fmt = re.search(r"\b(pdf|docx|txt|rtf)\b", message)
        if fmt:
            facts["preferred_document_format"] = fmt.group(1)
        if re.search(r"\b(?:voice|read\s*aloud|bol\s*kar|आवाज़)\b", message):
            facts["voice_output"] = not re.search(r"\b(?:off|band|no|mat)\b", message)
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return []

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return "Which preference would you like to change — language, explanation style, or download format?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        service = PreferenceService()
        owner = str(context.authenticated_user_id)
        changes = {
            key: value for key, value in context.facts.items()
            if key in {"language", "explanation_mode", "preferred_document_format", "voice_output"}
        }
        if changes:
            preferences = await service.update(owner, UserPreferencesRequest(**changes))
            changed = ", ".join(f"{key.replace('_', ' ')} → {value}" for key, value in changes.items())
            message = f"Updated: {changed}."
        else:
            preferences = await service.get(owner)
            message = "Here is how I am set up for you."
        return WorkflowTurn(
            message=(
                f"{message}\n\n"
                f"- Language: {preferences.language}\n"
                f"- Explanation style: {preferences.explanation_mode}\n"
                f"- Download format: {preferences.preferred_document_format}\n"
                f"- Spoken replies: {'on' if preferences.voice_output else 'off'}\n\n"
                "These cover how I talk to you. Case facts stay in your case files and drafts."
            ),
            status="completed",
            allowed_actions=["answer me in Hindi", "use detailed explanations", "default to DOCX"],
            finished=True,
        )


@register_workflow
class DownloadsWorkflow(ChatWorkflow):
    """"What files do I have?" — the download centre, inline."""

    name = "downloads"
    title = "your downloads"
    required_role = Role.user

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.DOWNLOADS, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return []

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return "Which file would you like?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        from app.repositories.phase3 import DownloadArtifactRepository

        items = await DownloadArtifactRepository().list_for_owner(str(context.authenticated_user_id))
        if not items:
            return WorkflowTurn(
                message="You have no generated files yet. Ask me for a draft as a PDF and one will appear here.",
                status="completed", finished=True,
            )
        lines = "\n".join(
            f"- **{item.get('filename', 'file')}** ({str(item.get('format', '')).upper()})"
            for item in items[:15]
        )
        more = f"\n\n…and {len(items) - 15} more." if len(items) > 15 else ""
        return WorkflowTurn(
            message=f"**Your files**\n\n{lines}{more}\n\nSay which one you want and I will give you the link.",
            status="completed", finished=True,
        )


@register_workflow
class BackgroundJobsWorkflow(ChatWorkflow):
    """"Is my export done?" and "retry the failed one"."""

    name = "background_jobs"
    title = "background jobs"
    required_role = Role.user

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.JOBS, message)

    def needs_confirmation(self, context: WorkflowContext) -> bool:
        # Retrying re-runs a side effect (an export, an analysis, an
        # ingestion). Listing does not.
        return str(context.facts.get("action", "")) == "retry"

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        if not context.facts.get("action"):
            facts["action"] = (
                "retry" if re.search(r"\bretry|dobara|फिर\s*से|पुनः\b", context.message, re.IGNORECASE)
                else "list"
            )
        action = str(facts.get("action") or context.facts.get("action") or "")
        if action == "retry" and not context.facts.get("job_id"):
            parked = selection.from_dicts(list(context.facts.get("_choices") or []))
            if parked:
                chosen = selection.resolve(context.message, parked)
                if chosen is not None:
                    facts["job_id"] = chosen.key
                    facts["job_label"] = chosen.label
                return facts
            jobs = await BackgroundJobService().list(str(context.authenticated_user_id))
            failed = [job for job in jobs if job.status == "failed"]
            if len(failed) == 1:
                facts["job_id"] = failed[0].job_id
                facts["job_label"] = failed[0].job_type
            elif failed:
                facts["_choices"] = [
                    selection.Choice(key=job.job_id, label=job.job_type, detail=job.error or "failed").as_dict()
                    for job in failed
                ]
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        if str(context.facts.get("action", "")) == "retry":
            return ["job_id"]
        return []

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        choices = selection.from_dicts(list(context.facts.get("_choices") or []))
        if choices:
            return "Which one should I run again?\n\n" + selection.render(choices)
        return "I do not see a failed job to retry. Would you like the list of all your jobs?"

    def summarize_for_confirmation(self, context: WorkflowContext) -> str:
        return f"- Job: {context.facts.get('job_label', 'the failed job')}\n- Action: run it again"

    def idempotency_key(self, context: WorkflowContext) -> str:
        return f"{self.name}:retry:{context.facts.get('job_id')}"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        service = BackgroundJobService()
        owner = str(context.authenticated_user_id)
        if str(context.facts.get("action", "")) == "retry":
            job = await service.retry(owner, str(context.facts["job_id"]))
            return WorkflowTurn(
                message=f"Queued **{job.job_type}** again — it is `{job.status}`. I will have the result shortly.",
                status="completed", finished=True,
            )
        jobs = await service.list(owner)
        if not jobs:
            return WorkflowTurn(
                message="You have no background jobs running or finished.", status="completed", finished=True
            )
        lines = "\n".join(
            f"- **{job.job_type}** — {job.status}"
            + (f" ({job.progress}%)" if job.status == "running" else "")
            + (f" · {job.error}" if job.error else "")
            for job in jobs[:15]
        )
        failed = [job for job in jobs if job.status == "failed"]
        hint = "\n\nSay \"retry\" if you want me to run a failed one again." if failed else ""
        return WorkflowTurn(
            message=f"**Your jobs**\n\n{lines}{hint}",
            status="completed",
            allowed_actions=["retry the failed job"] if failed else [],
            finished=True,
        )


@register_workflow
class ClearConversationWorkflow(ChatWorkflow):
    """"Delete this conversation" — the session, not the account.

    Kept separate from `delete_my_data` on purpose: clearing one conversation
    is routine tidying and deleting an account's data is not. Conflating them
    would make the routine action alarmingly destructive.
    """

    name = "clear_conversation"
    title = "clearing this conversation"
    requires_confirmation = True

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.CLEAR_CONVERSATION, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return []

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return "Shall I clear this conversation?"

    def summarize_for_confirmation(self, context: WorkflowContext) -> str:
        return (
            "- This conversation's messages and its working memory\n"
            "- The query and intent records it produced\n\n"
            "Your drafts, cases and files are NOT touched — say \"delete my data\" if you want those gone too."
        )

    def idempotency_key(self, context: WorkflowContext) -> str:
        return f"{self.name}:{context.session_id}"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        from app.services.user_data import erase_session_data

        result = await erase_session_data(context.session_id)
        counts = result.get("deleted", {})
        total = sum(value for value in counts.values() if isinstance(value, int)) if isinstance(counts, dict) else 0
        # The stack and ledger live in the memory dict this turn is holding,
        # so they are cleared here rather than being resurrected on save.
        context.memory.clear()
        return WorkflowTurn(
            message=f"Cleared — {total} record(s) from this conversation are gone. Your drafts and cases are untouched.",
            status="completed", finished=True,
        )


@register_workflow
class DeleteMyDataWorkflow(ChatWorkflow):
    """Account-wide erasure. The most careful path in the product."""

    name = "delete_my_data"
    title = "erase your data"
    required_role = Role.user
    requires_confirmation = True

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.DELETE_MY_DATA, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        # The typed phrase IS the confirmation. Collected as a fact so the
        # orchestrator's yes/no turn cannot substitute for it.
        lowered = context.message.strip().lower()
        if any(phrase in lowered for phrase in _ERASE_PHRASES):
            return {"typed_confirmation": context.message.strip()}
        return {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["typed_confirmation"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return (
            "This erases everything I hold for your account — every conversation, every draft and its "
            "versions, your cases, saved files, preferences and background jobs, across every session "
            "you have ever used. It cannot be undone and I cannot recover any of it.\n\n"
            "If you are sure, type exactly: **delete my data**"
        )

    def summarize_for_confirmation(self, context: WorkflowContext) -> str:
        return (
            "- Conversations, drafts and all their versions\n"
            "- Case files, evidence index, tasks and reminders\n"
            "- Saved downloads and the files themselves\n"
            "- Preferences and background jobs\n\n"
            "Permanent, immediate, and across every session on this account."
        )

    def idempotency_key(self, context: WorkflowContext) -> str:
        return f"{self.name}:{context.authenticated_user_id}"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        from app.services.user_data import erase_user_data

        # Acts on the AUTHENTICATED id only. Nothing the user typed selects
        # whose data is erased.
        result = await erase_user_data(str(context.authenticated_user_id))
        counts = result.get("deleted", {})
        total = sum(value for value in counts.values() if isinstance(value, int)) if isinstance(counts, dict) else 0
        return WorkflowTurn(
            message=(
                f"Done — {total} record(s) erased across your account. Your login itself still exists; "
                "ask your administrator if you want the account closed as well."
            ),
            status="completed", finished=True,
        )
