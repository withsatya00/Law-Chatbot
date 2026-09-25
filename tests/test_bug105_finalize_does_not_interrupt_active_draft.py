"""Regression test for BUG-105 (QA session 2026-09-24): "finalize"/"finalise"
used to be listed in `_DRAFT_VERB_STRONG` (app/chatops/intents.py) alongside
verbs that genuinely can only mean draft administration ("approve", "lock",
"delete", ...). Unlike those, "finalize the draft" is at least as commonly
said to an actively-collecting draft ("please generate it now with what
I've given you") as it is a request to manage a saved one -- and
`interrupts_draft` has no notion of whether the session's draft is still
`collecting`, so it interrupted unconditionally, losing the in-progress
draft's conversation state.

Live-reproduced: "Please finalize the draft with the information already
provided", asked mid-collection, got routed to `DraftManagementWorkflow`,
which replied "I could not find a saved draft to work with" -- true for its
own purpose, but the active draft was abandoned.
"""

from app.chatops.orchestrator import orchestrator  # noqa: F401 - registers workflows
from app.chatops.registry import interrupts_draft


def test_finalize_the_draft_no_longer_interrupts_an_active_draft():
    message = "Please finalize the draft with the information already provided."
    assert interrupts_draft(message, "english") is False


def test_finalise_british_spelling_also_does_not_interrupt():
    # Deliberately avoids "my draft" -- that phrasing separately matches
    # `intents.SAVED_DRAFTS`'s own `\bmy\s*drafts?\b` pattern (an unrelated,
    # pre-existing, arguably-correct behavior: "my draft" plausibly DOES mean
    # an existing saved one), which is out of scope for this fix. This test
    # is about the "finalize"/"finalise" verb specifically, isolated from
    # that other pattern.
    assert interrupts_draft("Can you finalise the draft now?", "english") is False


def test_genuine_management_verbs_still_interrupt_no_regression():
    for message in ("delete my draft", "approve my draft", "lock my draft", "compare draft versions"):
        assert interrupts_draft(message, "english") is True, f"{message!r} should still interrupt"
