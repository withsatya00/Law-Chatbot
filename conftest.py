"""Test-wide storage isolation.

Every filesystem-backed setting in `app.core.config.Settings` points at a real
directory under `storage/` by default. Nothing redirected them during tests, so
a full run wrote into the live `storage/uploads`, `knowledge_base`,
`kb_staging`, `kb_review`, `archive`, `operations` and `drafts` trees -- an
observed run left 17 extra files in `storage/uploads` alone.

The redirection happens HERE, at conftest import time, rather than in a
fixture: `app.core.config` builds a module-level `settings = get_settings()`
singleton on first import, and services capture directory values from it at
import/construction time. By the time any fixture runs, those values are
already cached. Setting the environment before `app` is imported at all is the
only point at which the redirect is guaranteed to apply everywhere.

Production paths are untouched: this file is never imported outside pytest.
"""

import os
import tempfile
from pathlib import Path

import pytest

# The repository's real storage tree, resolved before anything can chdir.
REPO_ROOT = Path(__file__).resolve().parent
REAL_STORAGE = REPO_ROOT / "storage"

# One temp root for the whole session. Not `tmp_path`, which is function-scoped
# and therefore too late for a module-level settings singleton.
_TEST_STORAGE_ROOT = Path(tempfile.mkdtemp(prefix="legal-ai-test-storage-"))

# setting name -> subdirectory under the test storage root.
_ISOLATED_DIRS: dict[str, str] = {
    "UPLOAD_STORAGE_DIR": "uploads",
    "KNOWLEDGE_BASE_DIR": "knowledge_base",
    "KB_STAGING_DIR": "kb_staging",
    "KB_REVIEW_DIR": "kb_review",
    "ARCHIVE_DIR": "archive",
    "OPERATIONS_OUTPUT_DIR": "operations",
    "BACKUP_DIR": "backups",
    "DRAFT_OUTPUT_DIR": "drafts",
}

for _env_name, _subdir in _ISOLATED_DIRS.items():
    _target = _TEST_STORAGE_ROOT / _subdir
    _target.mkdir(parents=True, exist_ok=True)
    # An explicit override in the environment wins, so a developer can still
    # point a run at a scratch tree of their own.
    os.environ.setdefault(_env_name, str(_target))

os.environ.setdefault("ENVIRONMENT", "test")
# Ordinary unit tests keep their existing mocked Mongo repository contracts.
# The opt-in PostgreSQL integration suite explicitly sets this back to true.
os.environ.setdefault("POSTGRESQL_ENABLED", "false")
# `LegalDraftEngine._retrieve_legal_context` (grounds draft generation in the
# real Knowledge Base text of a template's hinted Acts/sections) runs a real
# hybrid retrieval -- including loading the local sentence-transformers
# embedding model -- on every single draft generation. None of the ~50
# `LegalDraftEngine()` call sites across the drafting test suite mock this
# out (they mock `.llm`, since that's what each test actually varies), so
# leaving it on by default here made every one of those tests pay a real
# multi-second model load for a signal they don't assert on. Off by default
# for the whole test session, the same way storage paths are redirected
# above; a test that specifically exercises this feature turns it back on
# and mocks `engine.retriever.retrieve` itself (see `test_draft_legal_context.py`).
os.environ.setdefault("DRAFT_LEGAL_CONTEXT_ENABLED", "false")

# No test in this suite should ever make a REAL outbound network connection --
# every KB-adapter/official-source/LLM-provider test mocks its own `fetcher`/
# `httpx.AsyncClient`/`LLMProvider` rather than talking to a live server (a
# prior review flagged a suspected leak in the KB-adapter tests specifically;
# a targeted re-check with exactly this guard found none live in this session,
# but the guard is cheap, catches ANY future leak instantly and loudly instead
# of silently succeeding or hanging on a real request, and is applied here --
# at conftest IMPORT time, like the storage isolation above -- so a leak
# during test COLLECTION, not just inside a test function body, is caught
# too. Full suite verified clean under this guard (3442 passed) before it was
# made permanent.
#
# Loopback is exempted: `test_p2_real_redis.py` and similar
# `PHASE1_LIVE_TESTS=1`-gated tests deliberately talk to a disposable local
# Redis on `127.0.0.1:36379`. Skipped entirely when a developer has opted
# into one of this suite's existing "hit real infrastructure" flags
# (`PHASE1_LIVE_TESTS`, `LEGAL_AI_LIVE_BASE_URL`, `LEGAL_AI_RUN_BENCHMARK`) --
# this guard exists to catch ACCIDENTAL leaks, not to block a deliberate,
# already-opted-into live run.
if not any(os.environ.get(name) for name in ("PHASE1_LIVE_TESTS", "LEGAL_AI_LIVE_BASE_URL", "LEGAL_AI_RUN_BENCHMARK")):
    import socket as _socket

    _LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
    _real_socket_connect = _socket.socket.connect

    def _guarded_socket_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else address
        if host in _LOOPBACK_HOSTS:
            return _real_socket_connect(self, address, *args, **kwargs)
        raise RuntimeError(
            f"Test attempted a REAL network connection to {address!r}. Every test must mock its "
            "fetcher/HTTP client/DB client rather than reach a live host -- see conftest.py's "
            "network-guard comment for the opt-out env vars if this is a deliberate live test."
        )

    _socket.socket.connect = _guarded_socket_connect


def session_storage_root() -> Path:
    """The session's isolated storage root, for tests that need to look at it."""
    return _TEST_STORAGE_ROOT


def _snapshot(directory: Path) -> set[Path]:
    if not directory.exists():
        return set()
    return {path for path in directory.rglob("*") if path.is_file()}


@pytest.fixture(scope="session", autouse=True)
def _real_storage_is_never_written(request: pytest.FixtureRequest):
    """Fails the session if anything appeared in the repository's real storage.

    A per-test guard would be cheaper to attribute but far more expensive to
    run; this catches the leak, and `-p no:randomly --lf` narrows it down.
    """
    before = _snapshot(REAL_STORAGE)
    yield
    added = sorted(str(path.relative_to(REPO_ROOT)) for path in _snapshot(REAL_STORAGE) - before)
    if added:
        pytest.fail(
            "Tests wrote into the repository's real storage tree; every storage "
            "setting must be redirected in conftest.py. New files:\n  "
            + "\n  ".join(added)
        )


@pytest.fixture
def isolated_storage_root() -> Path:
    return _TEST_STORAGE_ROOT


@pytest.fixture(autouse=True)
def _unverified_disclosure_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ChatService._no_verified_context_answer` makes a second `retriever.
    retrieve()` call (against `needs_review` chunks) when this is on -- see
    `app/services/chat_service.py`'s `_unverified_candidate_disclosure`. Every
    pre-existing test that mocks `retriever.retrieve` and asserts on
    `call_args`/`assert_called_once()` was written assuming exactly one call
    per turn, before that feature existed. Defaulting it off here keeps that
    assumption true everywhere except the feature's own tests
    (`test_unverified_source_disclosure.py`), which turn it on explicitly.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "chat_unverified_disclosure_enabled", False)


@pytest.fixture(autouse=True)
def _general_knowledge_fallback_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `no_verified_context` is true, `ChatService` calls the LLM for a
    General Knowledge fallback attempt (`app.core.gk_fallback`) only if this
    setting is on -- see `app/services/chat_service.py`'s `elif
    no_verified_context:` branches. Production enables it (`.env`), but every
    pre-existing test asserting the strict-RAG guardrail's "zero verified
    chunks never calls the LLM" invariant (e.g.
    `test_no_verified_context_short_circuits_without_calling_llm`) was written
    before that feature existed and would otherwise start failing purely
    because a REAL LLM call is now attempted on that path. Defaulting it off
    here keeps that assumption true everywhere except the feature's own tests
    (`test_gk_fallback.py`, which exercises `general_knowledge_answer`
    directly and never reads this setting) and any test that explicitly turns
    it on to exercise the fallback through `ChatService`.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "general_knowledge_fallback_enabled", False)
