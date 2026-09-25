"""Tests for the chat-first conversational orchestration.

The acceptance criterion these defend: a user can reach every permitted
capability by talking, in English, Hindi or Hinglish, without opening a
feature page -- and the safety and authorization guarantees that used to sit
behind those pages still hold when the door is a sentence instead of a button.
"""

import asyncio
from typing import Any

import pytest

# Importing the orchestrator registers every workflow.
import app.chatops.orchestrator  # noqa: F401
from app.chatops import confirm, intents, state
from app.chatops.artifacts import draft_artifact, notarized_artifact
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.orchestrator import ChatOrchestrator, _role_permits
from app.chatops.registry import WORKFLOWS, all_matches, best_match
from app.core.security import Role
from app.language.detector import extract_requested_language


def _run(coro):
    return asyncio.run(coro)


def _orchestrate(message: str, memory: dict[str, Any] | None = None, **kwargs) -> WorkflowTurn | None:
    return _run(
        ChatOrchestrator().handle_turn(
            session_id="s1",
            message=message,
            language=kwargs.pop("language", "english"),
            memory=memory if memory is not None else {},
            **kwargs,
        )
    )


# ---------------------------------------------------------------------------
# 1. Every capability is reachable by talking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        # English
        ("I want to report a cyber fraud", "cyber_fraud"),
        ("Please check the jurisdiction for my case", "jurisdiction"),
        ("Show me my saved drafts", "saved_drafts"),
        ("I need to verify this document", "notarization_verify"),
        ("Please prepare my document for notarization", "notarization_prepare"),
        # Hinglish
        ("Mujhe cyber fraud report karna hai", "cyber_fraud"),
        ("jurisdiction check karo", "jurisdiction"),
        ("meri saved drafts dikhao", "saved_drafts"),
        ("mera document notary ke liye prepare karo", "notarization_prepare"),
        ("PDF me download karna hai", "draft_export"),
        # Hindi
        ("मुझे साइबर फ्रॉड दर्ज करना है", "cyber_fraud"),
        ("नोटरी के लिए तैयार करो", "notarization_prepare"),
    ],
)
def test_capabilities_are_reachable_from_plain_language(message: str, expected: str) -> None:
    match = best_match(message, "english")
    assert match is not None, f"no workflow matched: {message!r}"
    assert match[0].name == expected


def test_every_registered_workflow_implements_the_full_interface() -> None:
    """A workflow missing a method would fail at runtime, mid-conversation."""
    for name, workflow in WORKFLOWS.items():
        assert isinstance(workflow, ChatWorkflow), name
        for method in (
            "matches_intent", "extract_facts", "required_fields", "next_question",
            "validate", "summarize_for_confirmation", "execute", "render_chat_response",
        ):
            assert callable(getattr(workflow, method, None)), f"{name} is missing {method}"


# ---------------------------------------------------------------------------
# 2. The orchestrator must not hijack ordinary questions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "What is anticipatory bail?",
        "cyber fraud kya hota hai",
        "I lost money in an online cyber fraud.",
        "Explain territorial jurisdiction in simple terms",
        "Review this PDF and draft a legal notice.",
        "Please review this uploaded PDF document, tell me the risky clauses, and draft a legal notice.",
        "What does notarization mean?",
    ],
)
def test_informational_messages_fall_through_to_normal_answering(message: str) -> None:
    """Topic vocabulary alone must never start a workflow.

    A question about cyber fraud, or a statement that one occurred, wants a
    grounded legal answer -- not to be pulled into an emergency form. Only an
    explicit request to DO something routes to a workflow.
    """
    assert best_match(message, "english") is None
    assert _orchestrate(message) is None


@pytest.mark.parametrize(
    "message",
    [
        "Mera ₹2 lakh ka cheque bounce ho gaya hai. Legal notice kitne din ke andar bhejna hota hai?",
        "Cheque bounce mein notice kab tak bhejna hota hai?",
        # "file" here is the VERB ("complaint file karna" = to file a
        # complaint), not a reference to an uploaded document -- a live
        # session got stuck asking for an attachment on exactly this phrasing.
        "Consumer complaint file karne ke liye kitna time limit hai?",
    ],
)
def test_legal_deadline_questions_do_not_start_document_timeline(message: str) -> None:
    """A legal time-limit question does not imply that the user has a file to upload."""
    assert best_match(message, "hinglish") is None
    assert _orchestrate(message, language="hinglish") is None


def test_document_deadline_question_still_starts_document_timeline() -> None:
    match = best_match("Is document mein deadlines kya hain?", "hinglish")
    assert match is not None
    assert match[0].name == "document_timeline"


def test_action_cue_is_what_distinguishes_a_request_from_a_question() -> None:
    assert intents.wants_action("mujhe report karna hai")
    assert intents.wants_action("please file a complaint")
    assert not intents.wants_action("what is cyber fraud")
    assert not intents.wants_action("I lost money in an online cyber fraud.")


# ---------------------------------------------------------------------------
# 3. One question at a time; never re-ask a known fact
# ---------------------------------------------------------------------------


def test_missing_fields_are_asked_one_at_a_time() -> None:
    memory: dict[str, Any] = {}
    turn = _orchestrate("mera document notary ke liye prepare karo", memory, authenticated_user_id="u1")
    assert turn is not None
    assert turn.status == "collecting"
    # Exactly one field is being asked about.
    assert turn.missing_field is not None
    assert turn.message.count("?") <= 2  # the boundary notice may add one


def test_a_fact_already_collected_is_never_requested_again() -> None:
    """`missing_fields` is central so this holds for every workflow."""
    workflow = WORKFLOWS["notarization_prepare"]
    context = WorkflowContext(
        session_id="s", message="", language="english", memory={},
        facts={"draft_id": "d1", "full_name": "Rahul", "address": "24 Shastri Nagar"},
    )
    missing = workflow.missing_fields(context)
    assert "full_name" not in missing
    assert "address" not in missing
    assert "identity_document_type" in missing


def test_facts_already_in_conversation_memory_are_reused() -> None:
    """An active draft in memory must not be asked for again."""
    workflow = WORKFLOWS["notarization_prepare"]
    context = WorkflowContext(
        session_id="s", message="notary ke liye prepare karo", language="english",
        memory={"draft_id": "draft-from-chat"}, facts={},
    )
    discovered = _run(workflow.extract_facts(context))
    assert discovered.get("draft_id") == "draft-from-chat"


# ---------------------------------------------------------------------------
# 4. Pause / resume / cancel / restart / switch
# ---------------------------------------------------------------------------


# The cyber-fraud workflow deliberately completes in ONE turn (it is
# time-critical triage), so it is the wrong subject for lifecycle tests.
# Notarization preparation is genuinely multi-turn and is used instead.
_MULTI_TURN = "notary ke liye prepare karo"


def test_cancel_ends_the_workflow_without_executing() -> None:
    memory: dict[str, Any] = {}
    _orchestrate(_MULTI_TURN, memory, authenticated_user_id="u1")
    assert state.active_name(memory) == "notarization_prepare"
    turn = _orchestrate("cancel karo", memory, authenticated_user_id="u1")
    assert turn is not None and turn.status == "cancelled"
    assert state.active_name(memory) is None


def test_pause_keeps_facts_and_resume_finds_them() -> None:
    memory: dict[str, Any] = {}
    _orchestrate(_MULTI_TURN, memory, authenticated_user_id="u1")
    state.remember_facts(memory, {"full_name": "Rahul Sharma"})
    turn = _orchestrate("baad me continue karenge", memory, authenticated_user_id="u1")
    assert turn is not None and turn.status == "paused"
    # The workflow and its facts survive the pause.
    entry = next(e for e in memory[state.MEMORY_KEY] if e["name"] == "notarization_prepare")
    assert entry["facts"]["full_name"] == "Rahul Sharma"


def test_switching_tasks_parks_the_first_without_losing_it() -> None:
    """Starting a second task mid-flow must not discard the first's answers."""
    memory: dict[str, Any] = {}
    _orchestrate(_MULTI_TURN, memory, authenticated_user_id="u1")
    state.remember_facts(memory, {"full_name": "Rahul Sharma"})

    _orchestrate("meri saved drafts dikhao", memory, authenticated_user_id="u1")

    entry = next(
        (e for e in memory[state.MEMORY_KEY] if e["name"] == "notarization_prepare"), None
    )
    assert entry is not None, "the notarization workflow was lost when the user switched tasks"
    assert entry["facts"]["full_name"] == "Rahul Sharma"


def test_restart_clears_collected_facts() -> None:
    memory: dict[str, Any] = {}
    _orchestrate(_MULTI_TURN, memory, authenticated_user_id="u1")
    state.remember_facts(memory, {"full_name": "Rahul Sharma"})
    _orchestrate("shuru se shuru karo", memory, authenticated_user_id="u1")
    assert state.active(memory)["facts"] == {}


def test_workflow_stack_is_bounded() -> None:
    """A user naming many tasks must not grow memory without limit."""
    memory: dict[str, Any] = {}
    for index in range(12):
        state.start(memory, f"workflow_{index}")
    assert len(memory[state.MEMORY_KEY]) <= 5


# ---------------------------------------------------------------------------
# 5. Confirmation before irreversible actions
# ---------------------------------------------------------------------------


def test_notarization_preparation_requires_explicit_confirmation() -> None:
    assert WORKFLOWS["notarization_prepare"].requires_confirmation is True


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("haan", "yes"), ("theek hai generate karo", "yes"), ("confirm", "yes"),
        ("हाँ", "yes"), ("ஆம்", "yes"), ("অবশ্যই", "unclear"), ("হ্যাঁ", "yes"),
        ("nahi", "no"), ("नहीं", "no"), ("வேண்டாம்", "no"), ("cancel", "no"),
        ("maybe", "unclear"), ("what does that mean", "unclear"),
    ],
)
def test_confirmation_is_understood_across_languages(reply: str, expected: str) -> None:
    """Indic combining marks are not `\\w`, so a `\\b`-terminated pattern
    silently failed on every Indic affirmative -- a Hindi user could not say
    yes. Pinned here so it cannot regress."""
    assert confirm.classify(reply) == expected


def test_an_unclear_reply_is_never_treated_as_consent() -> None:
    memory: dict[str, Any] = {}
    state.start(memory, "notarization_prepare")
    state.update(memory, pending_confirmation=True)
    turn = _orchestrate("hmm I am not sure", memory, authenticated_user_id="u1")
    assert turn is not None
    assert turn.status == "awaiting_confirmation"
    assert turn.requires_confirmation is True
    # Still pending -- nothing executed.
    assert state.active(memory)["pending_confirmation"] is True


def test_declining_a_confirmation_executes_nothing() -> None:
    memory: dict[str, Any] = {}
    state.start(memory, "notarization_prepare")
    state.update(memory, pending_confirmation=True)
    turn = _orchestrate("nahi", memory, authenticated_user_id="u1")
    assert turn is not None and turn.status == "cancelled"
    assert state.active_name(memory) is None


# ---------------------------------------------------------------------------
# 6. Role enforcement
# ---------------------------------------------------------------------------


def test_notary_and_admin_workflows_declare_a_required_role() -> None:
    assert WORKFLOWS["notary_queue"].required_role is Role.lawyer
    assert WORKFLOWS["notary_admin"].required_role is Role.admin
    # A normal user's workflows must NOT be role-gated, or ordinary users
    # would be locked out of the product.
    assert WORKFLOWS["cyber_fraud"].required_role is None
    assert WORKFLOWS["notarization_prepare"].required_role is None


def test_a_normal_user_cannot_reach_the_notary_queue_by_asking() -> None:
    turn = _orchestrate(
        "meri pending review requests dikhao",
        {},
        authenticated_user_id="u1",
        claims={"sub": "u1", "role": "user"},
    )
    assert turn is not None
    assert turn.status == "forbidden"


def test_a_normal_user_cannot_reach_notary_administration_by_asking() -> None:
    turn = _orchestrate(
        "notarization audit log dikhao",
        {},
        authenticated_user_id="u1",
        claims={"sub": "u1", "role": "user"},
    )
    assert turn is not None
    assert turn.status == "forbidden"


def test_an_anonymous_user_cannot_reach_privileged_workflows() -> None:
    turn = _orchestrate("pending notary accounts dikhao", {}, claims={})
    assert turn is not None
    assert turn.status == "forbidden"


def test_role_hierarchy_is_explicit_not_inferred() -> None:
    assert _role_permits(None, "") is True
    assert _role_permits(Role.admin, "user") is False
    assert _role_permits(Role.admin, "admin") is True
    assert _role_permits(Role.admin, "super_admin") is True
    assert _role_permits(Role.lawyer, "user") is False
    assert _role_permits(Role.lawyer, "lawyer") is True


# ---------------------------------------------------------------------------
# 7. Notarization safety boundary
# ---------------------------------------------------------------------------


def test_no_chat_workflow_can_mark_a_document_notarized() -> None:
    """Nothing reachable from a normal user's chat may reach `notarized`.

    The only path is `NotarizationService.approve_request`, which needs a
    verified notary account plus a fresh re-authentication token -- neither
    obtainable from a chat turn.
    """
    import ast
    import inspect

    from app.chatops.workflows import notarization as module

    # Compare against the AST, not the raw text: the module docstring
    # legitimately NAMES `approve_request` while explaining why nothing here
    # may call it, and a naive grep would flag its own documentation.
    tree = ast.parse(inspect.getsource(module))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "approve_request" not in called, "a chat workflow must not call approve_request"

    assigned = {
        target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
    }
    for banned in ("notarized_by_notary_id", "verification_token"):
        assert banned not in assigned, f"a chat workflow must not assign {banned}"


def test_preparation_workflow_states_the_boundary_before_collecting_anything() -> None:
    """The user must be told what the product will and will not do, first."""
    workflow = WORKFLOWS["notarization_prepare"]
    context = WorkflowContext(session_id="s", message="notary karani hai", language="english", memory={}, facts={})
    question = workflow.next_question(context, ["draft_id"])
    assert "do not issue notarial stamps" in question.lower() or "i do not issue" in question.lower()


def test_no_workflow_fabricates_notarial_artifacts() -> None:
    import inspect

    from app.chatops.workflows import notarization as module

    source = inspect.getsource(module).lower()
    # Nothing here may invent an attestation.
    for banned in ("fake", "dummy_stamp", "generate_seal", "make_certificate"):
        assert banned not in source


# ---------------------------------------------------------------------------
# 8. Secure artifacts
# ---------------------------------------------------------------------------


def test_artifacts_carry_api_routes_never_filesystem_paths() -> None:
    for artifact in (draft_artifact("d1", "pdf"), notarized_artifact("doc1")):
        path = artifact["download_path"]
        assert path.startswith("/"), path
        # No drive letters, no traversal, no OS separators beyond URL slashes.
        assert ":" not in path and "\\" not in path and ".." not in path
        assert "storage" not in path


def test_notarized_artifact_points_at_the_gated_route() -> None:
    artifact = notarized_artifact("doc-1")
    assert artifact["download_path"] == "/notarization/documents/doc-1/download"
    assert artifact["kind"] == "notarized_document"


# ---------------------------------------------------------------------------
# 9. Multi-intent
# ---------------------------------------------------------------------------


def test_multiple_capabilities_in_one_message_are_all_detected() -> None:
    """"analyse karo aur PDF bana do" names more than one thing; the
    orchestrator must see all of them rather than silently dropping one."""
    matches = all_matches("meri saved drafts dikhao aur PDF me download karna hai", "english")
    names = {workflow.name for workflow, _score in matches}
    assert {"saved_drafts", "draft_export"} <= names


# ---------------------------------------------------------------------------
# 10. Language coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "language",
    [
        "hindi", "bengali", "marathi", "telugu", "tamil", "gujarati", "urdu", "kannada",
        "odia", "malayalam", "punjabi", "assamese", "maithili", "santali", "kashmiri",
        "nepali", "konkani", "sindhi", "dogri", "manipuri", "bodo", "sanskrit",
    ],
)
def test_every_eighth_schedule_language_can_be_requested_by_name(language: str) -> None:
    """All 22 were already supported for OUTPUT, but only about half could be
    ASKED for by name -- so a request in the other half silently fell through
    and the reply came back in the wrong language."""
    from app.core.constants import SUPPORTED_LANGUAGES

    assert language in SUPPORTED_LANGUAGES
    assert extract_requested_language(f"{language} me draft banao") == language


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("অসমীয়া ভাষা", "assamese"),
        ("ગુજરાતી ભાષા", "gujarati"),
        ("தமிழ் மொழி", "tamil"),
        ("తెలుగు భాష", "telugu"),
        ("ಕನ್ನಡ ಭಾಷೆ", "kannada"),
        ("മലയാളം ഭാഷ", "malayalam"),
        ("ଓଡ଼ିଆ ଭାଷା", "odia"),
        ("संस्कृत में बनाओ", "sanskrit"),
        ("नेपाली में", "nepali"),
    ],
)
def test_languages_can_be_requested_in_their_own_script(phrase: str, expected: str) -> None:
    assert extract_requested_language(phrase) == expected


# ---------------------------------------------------------------------------
# 11. Failure handling
# ---------------------------------------------------------------------------


def test_a_workflow_crash_never_leaks_a_traceback_into_chat() -> None:
    class _Exploding(ChatWorkflow):
        name = "exploding_test_workflow"
        title = "exploding workflow"

        def matches_intent(self, message: str, language: str) -> float:
            return 0.0

        async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
            return {}

        def required_fields(self, context: WorkflowContext) -> list[str]:
            return []

        def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
            return "?"

        async def execute(self, context: WorkflowContext) -> WorkflowTurn:
            raise RuntimeError("secret internal detail: /srv/app/storage/key.pem")

    memory: dict[str, Any] = {}
    context = WorkflowContext(session_id="s", message="", language="english", memory=memory, facts={})
    turn = _run(ChatOrchestrator()._execute(_Exploding(), context, memory))
    assert turn.status == "failed"
    assert "secret internal detail" not in turn.message
    assert "/srv/" not in turn.message
    assert "nothing was submitted" in turn.message.lower()


# ---------------------------------------------------------------------------
# 12. End-to-end: every capability is reachable from POST /chat itself
# ---------------------------------------------------------------------------


def _chat_service():
    """A `ChatService` with I/O collaborators mocked, mirroring the harness in
    `test_chat_service_routing.py`. Retrieval raises if reached, so a workflow
    that silently fell through to RAG fails loudly."""
    from unittest.mock import AsyncMock

    from app.services.chat_service import ChatService

    service = ChatService()
    service.prompt_scanner.scan = lambda text: (False, [])
    empty_memory = {"messages": [], "summary": "", "current_intent": None, "legal_category": None}
    service.memory.append = AsyncMock(return_value=empty_memory)
    service.memory.load = AsyncMock(return_value=empty_memory)
    service.memory.update = AsyncMock(return_value={})
    service.memory.summarize_if_needed = AsyncMock(return_value={})
    service.history.insert = AsyncMock(return_value="history-id")
    service.query_log.insert = AsyncMock(return_value="query-log-id")
    service.response_cache.lookup = AsyncMock(return_value=(None, "miss"))
    service.response_cache.store = AsyncMock(return_value=None)
    service.retriever.retrieve = AsyncMock(side_effect=AssertionError("a workflow turn must not reach retrieval"))
    service.reranker.rerank = AsyncMock(side_effect=AssertionError("a workflow turn must not reach reranking"))
    return service


@pytest.mark.parametrize(
    "question,expected_workflow",
    [
        ("Mujhe cyber fraud report karna hai", "cyber_fraud"),
        ("jurisdiction check karo", "jurisdiction"),
        ("meri saved drafts dikhao", "saved_drafts"),
        ("mera document notary ke liye prepare karo", "notarization_prepare"),
        ("मुझे साइबर फ्रॉड दर्ज करना है", "cyber_fraud"),
    ],
)
def test_workflows_are_reachable_through_the_chat_endpoint(question: str, expected_workflow: str) -> None:
    """The acceptance criterion: no feature page needed, just `/chat`."""
    from app.schemas.chat import ChatRequest

    service = _chat_service()
    response = _run(service.answer(ChatRequest(question=question)))
    assert response.conversation_intent == "Workflow"
    assert response.workflow_status is not None
    assert response.assistant_message
    # `intent`/`active_workflow` name the workflow while it is still in the
    # foreground; a one-shot workflow reports its status instead.
    assert expected_workflow in {response.intent, response.active_workflow} or response.workflow_status == "completed"


def test_a_workflow_turn_never_reaches_retrieval() -> None:
    from app.schemas.chat import ChatRequest

    service = _chat_service()
    _run(service.answer(ChatRequest(question="meri saved drafts dikhao")))
    service.retriever.retrieve.assert_not_awaited()


def test_ordinary_questions_still_route_to_the_existing_pipeline() -> None:
    """Backward compatibility: chatops must be invisible to a normal turn."""
    from unittest.mock import AsyncMock

    from app.llm.base import LLMResponse
    from app.schemas.chat import ChatRequest

    service = _chat_service()
    service.llm.chat = AsyncMock(return_value=LLMResponse(content="Happy to help!", model="t", provider="t"))
    response = _run(service.answer(ChatRequest(question="Thanks, that helps")))
    assert response.conversation_intent == "General Conversation"
    # None of the orchestration fields are populated on a non-workflow turn.
    assert response.active_workflow is None
    assert response.assistant_message == ""
    assert response.artifact is None
    assert response.collected_facts == {}


def test_chat_response_orchestration_fields_are_all_optional() -> None:
    """A pre-existing client that never sends or reads them keeps working."""
    from app.schemas.chat import ChatResponse

    added = [
        "intent", "active_workflow", "assistant_message", "missing_field", "collected_facts",
        "requires_confirmation", "allowed_actions", "artifact", "citations", "warnings",
        "upload_required", "secure_action_url",
    ]
    for name in added:
        assert name in ChatResponse.model_fields
        assert not ChatResponse.model_fields[name].is_required(), f"{name} must be optional"


# ---------------------------------------------------------------------------
# 13. The feature navigation is actually gone, not merely hidden
# ---------------------------------------------------------------------------


def _frontend_source() -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "streamlit_app"
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py"))
    )


@pytest.mark.parametrize(
    "label",
    [
        "Legal Workflows",
        "Verify a Document",
        "Notary Queue",
        "Notarization Admin",
        "Admin Dashboard",
        "Download center",
    ],
)
def test_feature_navigation_buttons_are_removed(label: str) -> None:
    """These must not come back as buttons. Every one is now a sentence."""
    source = _frontend_source()
    assert f'st.button("{label}' not in source
    assert f'st.button("🛠️ {label}' not in source
    assert f'st.button("🔎 {label}' not in source
    assert f'st.button("🏛️ {label}' not in source
    assert f'st.button("🔐 {label}' not in source


def test_no_feature_pages_remain_in_the_frontend() -> None:
    source = _frontend_source()
    for gone in (
        "show_phase2", "show_verification", "show_notary", "show_notarization_admin",
        "render_notary_dashboard", "render_notarization_admin", "render_public_verification",
        "render_prepare_for_notarization", "_render_phase2_workspace", "_render_admin_panel",
    ):
        assert gone not in source, f"{gone} still present in the frontend"


def test_the_sidebar_keeps_only_what_the_spec_allows() -> None:
    """New chat, previous conversations, and minimal settings."""
    source = _frontend_source()
    assert "＋ New Chat" in source
    assert "⚙️ Settings" in source


def test_the_composer_keeps_its_essential_controls() -> None:
    source = _frontend_source()
    assert "st.chat_input" in source        # text + send
    assert "st.file_uploader" in source or "attachment" in source.lower()
    assert "st.audio_input" in source       # voice


# ---------------------------------------------------------------------------
# 14. The frontend never calls a rendering helper that does not exist
# ---------------------------------------------------------------------------


def test_every_notarization_ui_helper_the_app_calls_actually_exists() -> None:
    """A missing helper crashed the whole chat turn at runtime.

    `app.py` called `notarization_ui.render_workflow_state`, which existed on
    disk but not in the module Streamlit had cached from before the edit.
    Static equivalent of that check, so a rename or deletion fails here
    instead of in front of a user.
    """
    import ast
    import re
    from pathlib import Path

    frontend = Path(__file__).resolve().parents[1] / "streamlit_app"
    app_source = (frontend / "app.py").read_text(encoding="utf-8")
    module_tree = ast.parse((frontend / "notarization_ui.py").read_text(encoding="utf-8"))

    defined = {
        node.name for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    # Require a following "(" so this matches actual CALLS -- a bare
    # `notarization_ui.py` inside a comment is a filename, not a helper.
    called = set(re.findall(r"notarization_ui\.(\w+)\s*\(", app_source))
    missing = called - defined
    assert not missing, f"app.py calls notarization_ui.{{{', '.join(sorted(missing))}}} which do not exist"


def test_rendering_failures_cannot_destroy_the_assistant_turn() -> None:
    """Extras are decorations around an answer the user already has.

    One raising helper used to abort the turn and take the answer with it.
    """
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "streamlit_app" / "app.py").read_text(encoding="utf-8")
    start = app_source.index("def _render_assistant_extras(")
    end = app_source.index("\ndef ", start + 1)
    body = app_source[start:end]
    assert "notarization_ui.render_workflow_state" in body
    assert "try:" in body, "workflow rendering must be guarded"
    assert "except AttributeError" in body, "a stale module must degrade to a message, not a traceback"
