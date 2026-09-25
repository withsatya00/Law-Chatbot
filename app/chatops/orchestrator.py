"""The conversational workflow orchestrator.

One turn, in order:

    1. Control verbs first (cancel / pause / restart / back). These must win
       over everything, including a workflow that thinks the message is a
       field answer -- "cancel" typed while being asked for an address is a
       cancellation, not an address.
    2. If a workflow is awaiting confirmation, resolve the yes/no. Nothing
       else may run: an ambiguous reply re-asks rather than proceeding.
    3. If a workflow is active, let it continue, unless the message clearly
       starts a DIFFERENT workflow -- in which case the current one is parked
       with its facts intact and can be resumed.
    4. Otherwise, match a new workflow.
    5. Anything unmatched returns None, and `ChatService` answers normally.

Authorization is checked here, once, from the JWT claims -- never inferred
from the message and never delegated to a workflow's own judgement.
"""

import re
from typing import Any

import structlog

from app.chatops import confirm, intents, state

# Import for the registration side effect -- every workflow registers
# itself on import, so this is what populates `WORKFLOWS`.
from app.chatops import workflows as _workflows  # noqa: F401
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.registry import all_matches, best_match, get_workflow
from app.core.constants import language_preference_acknowledgement
from app.core.exceptions import AppError, ForbiddenError
from app.core.security import Role
from app.language.detector import is_language_preference_command

log = structlog.get_logger(__name__)


# Roles that satisfy a workflow's `required_role`. Admin outranks notary for
# read-only visibility, but NOT for notarial acts -- those are gated a second
# time inside `NotarizationService._require_verified_notary`, which checks for
# an actual verified notary ACCOUNT, not merely a role claim.
_ROLE_HIERARCHY: dict[str, set[str]] = {
    Role.user.value: {Role.user.value, Role.lawyer.value, Role.admin.value, Role.super_admin.value},
    Role.lawyer.value: {Role.lawyer.value, Role.admin.value, Role.super_admin.value},
    Role.admin.value: {Role.admin.value, Role.super_admin.value},
    Role.super_admin.value: {Role.super_admin.value},
}


# Where a pending "which of these did you mean?" is parked. One key in the
# same conversation-memory dict as the workflow stack, and read for exactly
# one turn -- see `_resolve_pending_choice`.
_PENDING_CHOICE_KEY = "chatops_pending_choice"


def _role_permits(required: Role | None, actual: str) -> bool:
    if required is None:
        return True
    return actual in _ROLE_HIERARCHY.get(required.value, {required.value})


def _forbidden_turn(workflow: ChatWorkflow) -> WorkflowTurn:
    """Refused before the workflow runs at all, so an unauthorised user never
    even reaches fact collection -- and the message says what is required
    without naming internals."""
    log.info(
        "chatops_role_refused",
        workflow=workflow.name,
        required=getattr(workflow.required_role, "value", None),
    )
    if workflow.required_role is Role.user:
        # Not a privilege refusal -- the user simply is not signed in, and
        # saying "only a notary may do this" would be both wrong and
        # alarming.
        message = (
            "You will need to be signed in for that, so I can keep the record to your account only. "
            "Sign in and ask me again."
        )
    else:
        message = (
            "This action is available only to a verified notary or an administrator. "
            "If you believe you should have access, please sign in with that account."
        )
    return WorkflowTurn(message=message, status="forbidden", workflow_name=workflow.name)


class ChatOrchestrator:
    """Drives `ChatWorkflow`s across turns. Stateless; state lives in memory."""

    async def handle_turn(
        self,
        *,
        session_id: str,
        message: str,
        language: str,
        memory: dict[str, Any],
        authenticated_user_id: str | None = None,
        claims: dict[str, Any] | None = None,
        document_ids: list[str] | None = None,
    ) -> WorkflowTurn | None:
        """This turn's workflow response, or None to fall through to RAG."""
        claims = claims or {}
        document_ids = document_ids or []
        # A workflow nobody has touched for `state.STALE_AFTER` is dropped
        # before anything reads it, so yesterday's half-typed address is
        # never silently reused as though it had just been said.
        expired = state.expire_stale(memory)
        if expired:
            log.info("chatops_state_expired", workflows=expired)
        active_entry = state.active(memory)

        # A standalone language switch is conversation control, not the
        # answer to whichever fact a workflow most recently requested. Only
        # handled here while a workflow is ACTIVE -- that is the case this
        # branch exists for (the switch must not be collected as a field
        # value, and the half-collected facts must survive). With no active
        # workflow there is nothing to protect, and `ChatService` routes the
        # same message as the "Language Preference" conversation intent,
        # which persists the preference to the conversation record and
        # acknowledges in the requested language rather than in whatever
        # language this turn happened to be detected as.
        #
        # Recognised by subtraction in `is_language_preference_command` (see
        # its docstring), not by a phrase regex. The regex this replaced
        # matched nothing longer than "answer in hindi", so every natural
        # phrasing a user actually types -- "Mujhe simple Hindi mein jawab
        # diya karo", "Actually Hinglish mein jawab do" -- fell through to
        # legal routing and was answered from retrieval.
        if active_entry is not None:
            requested_language = is_language_preference_command(message)
            if requested_language:
                memory["language_preference"] = requested_language
                state.touch(active_entry)
                return WorkflowTurn(
                    message=language_preference_acknowledgement(requested_language, True),
                    status="collecting",
                    workflow_name=str(active_entry.get("name", "")),
                    finished=False,
                )

        # -- 0. A pending "which of these did you mean?" ------------------
        pending_choice = self._resolve_pending_choice(message, memory)
        if pending_choice is not None:
            workflow = get_workflow(pending_choice)
            if workflow is not None:
                return await self._start_and_run(
                    workflow, session_id, message, language, memory,
                    authenticated_user_id, claims, document_ids,
                )

        # -- 1. Control verbs -------------------------------------------
        control = self._control_turn(message, memory, language, active_entry)
        if control is not None:
            return control

        # -- 2. Pending confirmation ------------------------------------
        if active_entry and active_entry.get("pending_confirmation"):
            return await self._resolve_confirmation(
                session_id, message, language, memory, active_entry,
                authenticated_user_id, claims, document_ids,
            )

        # -- 2b. An ordinary question asked mid-workflow ------------------
        # Parked, not swallowed. The question is answered by the normal
        # pipeline and the collected facts survive for "continue".
        if self._is_interruption(message, language, active_entry):
            state.pause(memory)
            log.info("chatops_parked_for_question", workflow=(active_entry or {}).get("name"))
            return None

        # -- 3/4. Continue, switch, or start ----------------------------
        if active_entry is None:
            ambiguous = self._ambiguity_turn(message, language, memory, claims)
            if ambiguous is not None:
                return ambiguous

        workflow, is_new = self._select_workflow(message, language, memory, active_entry)
        if workflow is None:
            return None

        if not _role_permits(workflow.required_role, str(claims.get("role") or "")):
            return _forbidden_turn(workflow)

        if is_new:
            state.start(memory, workflow.name)
            active_entry = state.active(memory)

        return await self._run_workflow(
            workflow, session_id, message, language, memory, active_entry or {},
            authenticated_user_id, claims, document_ids,
        )

    # -- control verbs ---------------------------------------------------

    def _control_turn(
        self, message: str, memory: dict[str, Any], language: str, active_entry: dict[str, Any] | None
    ) -> WorkflowTurn | None:
        # "Continue" works even when nothing is in the foreground -- that is
        # the whole point of parking a workflow to answer a question.
        if intents.RESUME.search(message):
            revived = state.resume(memory)
            if revived is not None:
                workflow = get_workflow(str(revived.get("name", "")))
                title = workflow.title if workflow else revived.get("name", "")
                collected = dict(revived.get("facts", {}))
                return WorkflowTurn(
                    message=(
                        f"Picking the {title} back up — I still have "
                        f"{len(collected)} of your answers saved."
                    ),
                    status="collecting",
                    workflow_name=str(revived.get("name", "")),
                    collected_facts=collected,
                    allowed_actions=["continue", "start over", "cancel"],
                )
        if active_entry is None:
            return None
        name = active_entry.get("name", "")
        workflow = get_workflow(name)
        title = workflow.title if workflow else name

        if intents.CANCEL.search(message):
            state.finish(memory, name)
            return WorkflowTurn(
                message=f"Cancelled the {title}. Nothing was saved or submitted. What would you like to do instead?",
                status="cancelled",
                finished=True,
            )
        if intents.PAUSE.search(message):
            state.pause(memory)
            return WorkflowTurn(
                message=(
                    f"Paused the {title} — your answers are saved. "
                    f'Say "continue" whenever you want to pick it back up.'
                ),
                status="paused",
            )
        if intents.RESTART.search(message):
            state.finish(memory, name)
            state.start(memory, name)
            return WorkflowTurn(
                message=f"Starting the {title} again from the beginning.",
                status="collecting",
            )
        if intents.BACK.search(message):
            # Drop the most recently collected fact so the previous question
            # is asked again. Deliberately simple and visible: the user is
            # told exactly what was cleared.
            facts = active_entry.get("facts", {})
            if facts:
                last_key = list(facts)[-1]
                facts.pop(last_key, None)
                return WorkflowTurn(
                    message=f"Cleared the last answer ({last_key.replace('_', ' ')}). Let's set it again.",
                    status="collecting",
                    missing_field=last_key,
                    workflow_name=name,
                )
        if intents.CORRECTION.search(message):
            # "the name is actually Rahul" -- clear the named field so the
            # workflow asks again and re-extracts from what the user just
            # said. Which field is named is decided by matching the field
            # NAME against the message, never by asking a model.
            facts = active_entry.get("facts", {})
            lowered = message.lower()
            for key in list(facts):
                label = (workflow.field_label(key) if workflow else key.replace("_", " ")).lower()
                if label and label in lowered:
                    state.forget_fact(memory, key)
                    return WorkflowTurn(
                        message=f"Cleared the {label} so we can set it again. What should it be?",
                        status="collecting",
                        missing_field=key,
                        workflow_name=name,
                        collected_facts=dict(facts),
                    )
        return None

    # -- interruptions and ambiguity -------------------------------------

    def _is_interruption(
        self, message: str, language: str, active_entry: dict[str, Any] | None
    ) -> bool:
        """Whether an in-progress workflow should step aside for this message.

        Only for a genuine question that starts no workflow of its own. A
        message that matches another workflow is a task SWITCH, handled by
        `_select_workflow`, and a message that is simply an answer is neither.
        """
        if active_entry is None or active_entry.get("pending_confirmation"):
            return False
        if not intents.QUESTION.search(message):
            return False
        return best_match(message, language) is None

    def _ambiguity_turn(
        self, message: str, language: str, memory: dict[str, Any], claims: dict[str, Any]
    ) -> WorkflowTurn | None:
        """Asks which capability was meant when two score identically.

        Deliberately a numbered list of REGISTERED workflow titles: the reply
        is resolved against that list, so nothing the user types can name a
        workflow that does not exist.

        Candidates are filtered by role FIRST. Otherwise this question would
        itself be a disclosure -- offering "notary administration" as an
        option tells an ordinary user what privileged capabilities exist, and
        the refusal they would get afterwards comes too late to unsay it.
        """
        role = str(claims.get("role") or "")
        matches = [
            (workflow, score)
            for workflow, score in all_matches(message, language)
            if _role_permits(workflow.required_role, role)
        ]
        if len(matches) < 2 or matches[0][1] > matches[1][1]:
            return None
        tied = [workflow for workflow, score in matches if score == matches[0][1]]
        if len(tied) < 2:
            return None
        memory[_PENDING_CHOICE_KEY] = [workflow.name for workflow in tied]
        options = "\n".join(
            f"{index}. {workflow.title}" for index, workflow in enumerate(tied, start=1)
        )
        return WorkflowTurn(
            message=(
                "That could mean a couple of things — which did you have in mind?\n\n"
                f"{options}\n\nReply with the number, or say it another way."
            ),
            status="collecting",
            allowed_actions=[workflow.title for workflow in tied],
        )

    def _resolve_pending_choice(self, message: str, memory: dict[str, Any]) -> str | None:
        """Reads a reply to `_ambiguity_turn`, or expires it.

        One turn only. A reply that names none of the offered options clears
        the pending state and is routed as a fresh message, so a stale
        clarification can never capture an unrelated question.
        """
        names = memory.get(_PENDING_CHOICE_KEY)
        if not isinstance(names, list) or not names:
            return None
        memory.pop(_PENDING_CHOICE_KEY, None)
        text = (message or "").strip().lower()
        digits = re.match(r"^\s*([1-9])\b", text)
        if digits:
            index = int(digits.group(1)) - 1
            if 0 <= index < len(names):
                return str(names[index])
        for name in names:
            workflow = get_workflow(str(name))
            if workflow is not None and workflow.title.lower() in text:
                return str(name)
        return None

    async def _start_and_run(
        self, workflow: ChatWorkflow, session_id: str, message: str, language: str,
        memory: dict[str, Any], authenticated_user_id: str | None,
        claims: dict[str, Any], document_ids: list[str],
    ) -> WorkflowTurn:
        if not _role_permits(workflow.required_role, str(claims.get("role") or "")):
            return _forbidden_turn(workflow)
        state.start(memory, workflow.name)
        return await self._run_workflow(
            workflow, session_id, message, language, memory, state.active(memory) or {},
            authenticated_user_id, claims, document_ids,
        )

    # -- confirmation ----------------------------------------------------

    async def _resolve_confirmation(
        self, session_id: str, message: str, language: str, memory: dict[str, Any],
        active_entry: dict[str, Any], authenticated_user_id: str | None,
        claims: dict[str, Any], document_ids: list[str],
    ) -> WorkflowTurn:
        workflow = get_workflow(active_entry.get("name", ""))
        if workflow is None:
            state.finish(memory)
            return WorkflowTurn(message="That workflow is no longer available.", status="failed", finished=True)

        decision = confirm.classify(message)
        if decision == "no":
            state.finish(memory, workflow.name)
            return WorkflowTurn(
                message=f"Understood — I have not proceeded with the {workflow.title}. Nothing was submitted.",
                status="cancelled",
                finished=True,
            )
        if decision == "unclear":
            # A clear request for another workflow/action is not an unclear
            # confirmation. Abandon the unconfirmed side effect and handle
            # the new request instead. Example: while re-index is waiting for
            # yes/no, "KB staging status dikhao" must remain read-only and
            # must not trap the admin in the confirmation prompt.
            replacement = best_match(message, language)
            if replacement is not None:
                state.finish(memory, workflow.name)
                return await self._start_and_run(
                    replacement[0],
                    session_id,
                    message,
                    language,
                    memory,
                    authenticated_user_id,
                    claims,
                    document_ids,
                )
            # Never treat an ambiguous reply as consent.
            return WorkflowTurn(
                message=(
                    "Sorry, I need a clear yes or no before I go ahead. "
                    'Reply "yes" to proceed, or "no" to stop.'
                ),
                status="awaiting_confirmation",
                requires_confirmation=True,
                allowed_actions=["yes", "no"],
            )

        active_entry["pending_confirmation"] = False
        context = self._context(session_id, message, language, memory, active_entry, authenticated_user_id, claims, document_ids)
        return await self._execute(workflow, context, memory)

    # -- selection -------------------------------------------------------

    def _select_workflow(
        self, message: str, language: str, memory: dict[str, Any], active_entry: dict[str, Any] | None
    ) -> tuple[ChatWorkflow | None, bool]:
        """Returns `(workflow, is_new)`.

        A confident match on a DIFFERENT workflow parks the current one rather
        than hijacking it -- switching tasks must never silently discard
        answers already given.
        """
        match = best_match(message, language)
        if active_entry is not None:
            current = get_workflow(active_entry.get("name", ""))
            if match is not None and current is not None and match[0].name != current.name:
                state.pause(memory)
                return match[0], True
            if current is not None:
                return current, False
        if match is not None:
            return match[0], True
        return None, False


    # -- structured progress ---------------------------------------------

    @staticmethod
    def _with_progress(
        workflow: ChatWorkflow, context: WorkflowContext, turn: WorkflowTurn
    ) -> WorkflowTurn:
        """Fills the progress fields a workflow did not set for itself.

        Derived from `required_fields`, so a workflow gets a progress bar by
        declaring what it needs -- no per-workflow bookkeeping, and no way
        for the reported progress to drift from the facts actually held.
        """
        turn.workflow_name = turn.workflow_name or workflow.name
        # Facts whose names start with "_" are the workflow's own bookkeeping
        # (a parked candidate list and its record ids). They are never echoed
        # back to a client: the user picks by number or name, and an internal
        # id has no business travelling out in a chat response.
        turn.collected_facts = {
            key: value for key, value in turn.collected_facts.items() if not key.startswith("_")
        }
        if turn.total_steps:
            return turn
        required = workflow.required_fields(context)
        # +1 for the final act (confirm and execute), which is a step the
        # user experiences even when no fields are outstanding.
        total = len(required) + 1
        collected = [
            name for name in required if str(context.facts.get(name, "") or "").strip()
        ]
        done = len(collected)
        if turn.status in {"completed", "failed", "cancelled"}:
            done = total
        elif turn.status == "awaiting_confirmation":
            done = len(required)
        turn.total_steps = total
        turn.completed_steps = min(done, total)
        turn.progress_percentage = round(100 * turn.completed_steps / total) if total else 0
        if not turn.current_step:
            if turn.missing_field:
                turn.current_step = workflow.field_label(turn.missing_field)
            elif turn.status == "awaiting_confirmation":
                turn.current_step = "confirmation"
            elif turn.status == "completed":
                turn.current_step = "done"
            else:
                turn.current_step = workflow.title
        return turn

    def _render(
        self, workflow: ChatWorkflow, turn: WorkflowTurn, context: WorkflowContext
    ) -> WorkflowTurn:
        return self._with_progress(workflow, context, workflow.render_chat_response(turn, context))

    # -- execution -------------------------------------------------------

    def _context(
        self, session_id: str, message: str, language: str, memory: dict[str, Any],
        active_entry: dict[str, Any], authenticated_user_id: str | None,
        claims: dict[str, Any], document_ids: list[str],
    ) -> WorkflowContext:
        return WorkflowContext(
            session_id=session_id,
            message=message,
            language=language,
            memory=memory,
            facts=active_entry.setdefault("facts", {}),
            authenticated_user_id=authenticated_user_id,
            claims=claims,
            document_ids=document_ids,
        )

    async def _run_workflow(
        self, workflow: ChatWorkflow, session_id: str, message: str, language: str,
        memory: dict[str, Any], active_entry: dict[str, Any], authenticated_user_id: str | None,
        claims: dict[str, Any], document_ids: list[str],
    ) -> WorkflowTurn:
        context = self._context(
            session_id, message, language, memory, active_entry, authenticated_user_id, claims, document_ids
        )
        state.touch(active_entry)

        try:
            discovered = await workflow.extract_facts(context)
        except Exception as exc:  # noqa: BLE001 - extraction is best-effort
            log.warning("chatops_extract_failed", workflow=workflow.name, error=str(exc))
            discovered = {}
        context.remember(**discovered)
        state.remember_facts(memory, context.facts)

        problems = await workflow.validate(context)
        if problems:
            # Contradictions must be resolved by the user, never guessed.
            return self._render(
                workflow,
                WorkflowTurn(
                    message="Before I continue, please clear this up:\n\n"
                    + "\n".join(f"- {problem}" for problem in problems),
                    status="collecting",
                    collected_facts=dict(context.facts),
                    warnings=problems,
                ),
                context,
            )

        missing = workflow.missing_fields(context)
        if missing:
            field_name = missing[0]
            state.mark_asked(memory, field_name)
            return self._render(
                workflow,
                WorkflowTurn(
                    message=workflow.next_question(context, missing),
                    status="collecting",
                    missing_field=field_name,
                    collected_facts=dict(context.facts),
                ),
                context,
            )

        if workflow.needs_confirmation(context) and not active_entry.get("confirmed"):
            active_entry["pending_confirmation"] = True
            summary = workflow.summarize_for_confirmation(context)
            return self._render(
                workflow,
                WorkflowTurn(
                    message=(
                        f"Here is what I have for the {workflow.title}:\n\n{summary}\n\n"
                        "Shall I go ahead? (yes / no)"
                    ),
                    status="awaiting_confirmation",
                    requires_confirmation=True,
                    collected_facts=dict(context.facts),
                    allowed_actions=["yes", "no", "change a detail", "cancel"],
                ),
                context,
            )

        return await self._execute(workflow, context, memory)

    async def _execute(
        self, workflow: ChatWorkflow, context: WorkflowContext, memory: dict[str, Any]
    ) -> WorkflowTurn:
        key = workflow.idempotency_key(context)

        # A replayed turn -- a retry, a double-tapped send, a client that
        # resent on a flaky connection -- must not run the same side effect
        # twice. Checked here, in deterministic state, rather than trusting
        # every downstream service to be idempotent on its own.
        if state.already_executed(memory, key):
            log.info("chatops_duplicate_execution_blocked", workflow=workflow.name)
            state.finish(memory, workflow.name)
            return self._with_progress(
                workflow,
                context,
                WorkflowTurn(
                    message=(
                        f"I have already completed the {workflow.title} with these details, "
                        "so I have not repeated it. Ask me to show the result if you need it again."
                    ),
                    status="completed",
                    finished=True,
                ),
            )

        try:
            turn = await workflow.execute(context)
        except ForbiddenError as exc:
            # Backend authorization refused. Surfaced as-is (these messages
            # are written for users) but never with internals attached.
            log.info("chatops_execute_forbidden", workflow=workflow.name)
            return WorkflowTurn(message=str(exc), status="forbidden", workflow_name=workflow.name)
        except AppError as exc:
            log.info("chatops_execute_rejected", workflow=workflow.name, error=str(exc))
            return WorkflowTurn(message=str(exc), status="failed", workflow_name=workflow.name)
        except Exception as exc:
            log.exception("chatops_execute_failed", workflow=workflow.name, error=str(exc))
            return WorkflowTurn(
                message=(
                    "Something went wrong while completing that step, and nothing was submitted. "
                    "Please try again in a moment."
                ),
                status="failed",
                workflow_name=workflow.name,
            )

        # Recorded only on a successful run, so a failure the user retries is
        # genuinely retried rather than reported as already done.
        if turn.status == "completed":
            state.mark_executed(memory, key, workflow.title)
        if turn.finished or turn.status in {"completed", "cancelled"}:
            state.finish(memory, workflow.name)
        return self._render(workflow, turn, context)


orchestrator = ChatOrchestrator()
