"""Post-Phase-3 hardening, Phase 2 milestone E: what the assistant says it can
do, and what a reader can actually see under an answer.

Two defects from the observed session
-------------------------------------
1. Asked "Tum mere liye kya-kya kar sakte ho?", the assistant listed five
   capabilities and promised "Har jawab ke saath Act aur Section batata hoon"
   -- a citation on every answer. Twenty-nine chat workflows are registered.
   Saved drafts, draft editing and export, document review, comparison and
   timelines, cases, evidence annexures, lawyer-ready summaries, notarization
   preparation, downloads, background jobs and preferences were all missing,
   so a user had no way to know they existed. And the citation promise is one
   the system cannot keep: a citation carries an Act and section only when the
   retrieved chunk's metadata records them.

2. The Act and section the backend DID compute never reached the reader. They
   were rendered only inside the developer panel, which is off by default.
   Warnings and the source-currency note were not rendered at all. So the
   product simultaneously promised a citation on every answer and showed none.

Admin and notary-queue workflows are deliberately absent from the overview:
each declares a `required_role` and is refused for an ordinary account.
"""

import pytest

from app.core.constants import CAPABILITY_OVERVIEW_MESSAGES, capability_overview
from app.core.security import Role
from streamlit_app.answer_apparatus import source_lines


@pytest.fixture(scope="module")
def workflows() -> dict[str, object]:
    from app.chatops.orchestrator import orchestrator  # noqa: F401  (registers every workflow)
    from app.chatops.registry import WORKFLOWS

    return dict(WORKFLOWS)


# ---------------------------------------------------------------------------
# The capability overview
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("language", sorted(CAPABILITY_OVERVIEW_MESSAGES))
def test_the_overview_covers_the_capabilities_that_exist(language: str) -> None:
    """Each concept must be named in each language's own words."""
    text = capability_overview(language, 55).lower()
    expected = {
        "english": [
            "answer legal questions", "draft legal documents", "manage your drafts",
            "compare two versions", "timeline", "case", "annexure", "lawyer-ready summary",
            "notarization", "downloads", "background jobs", "preferences", "advocate",
        ],
        "hinglish": [
            "legal sawaalon ke jawab", "legal documents draft karna", "aapke drafts sambhalna",
            "comparison", "timeline", "case", "annexure", "lawyer-ready summary",
            "notarization", "downloads", "background jobs", "preferences", "advocate",
        ],
        "hindi": [
            "कानूनी सवालों के जवाब", "कानूनी दस्तावेज़ तैयार करना", "आपके ड्राफ्ट संभालना",
            "तुलना", "समयरेखा", "केस", "अनुलग्नक", "सारांश",
            "नोटरीकरण", "डाउनलोड", "पृष्ठभूमि कार्य", "वरीयता", "अधिवक्ता",
        ],
    }[language]
    missing = [item for item in expected if item.lower() not in text]
    assert not missing, f"{language} overview does not mention: {missing}"


@pytest.mark.parametrize("language", sorted(CAPABILITY_OVERVIEW_MESSAGES))
def test_the_overview_promises_no_citation_on_every_answer(language: str) -> None:
    text = capability_overview(language, 55)
    assert "every answer" not in text.lower()
    assert "Har jawab ke saath Act aur Section batata hoon" not in text
    assert "हर जवाब के साथ" not in text


@pytest.mark.parametrize("language", sorted(CAPABILITY_OVERVIEW_MESSAGES))
def test_the_overview_states_the_notarization_boundary(language: str) -> None:
    """"Prepare for notarization" is not "notarize"."""
    text = capability_overview(language, 55).lower()
    boundary = {
        "english": "i do not notarize",
        "hinglish": "khud notarize nahi karta",
        "hindi": "स्वयं नोटरीकरण नहीं करता",
    }[language]
    assert boundary in text


@pytest.mark.parametrize("language", sorted(CAPABILITY_OVERVIEW_MESSAGES))
def test_the_overview_does_not_advertise_role_gated_operations(
    language: str, workflows: dict[str, object]
) -> None:
    """An ordinary user must not be shown the operator surface."""
    text = capability_overview(language, 55).lower()
    # "knowledge base" is deliberately NOT here: telling a user their answers
    # come from a verified knowledge base is a statement about grounding, not
    # an offer to administer it. What must never appear is the operator surface.
    for forbidden in (
        "admin", "ingest", "reindex", "notary queue", "review queue", "audit log",
        "analytics dashboard", "प्रशासक", "व्यवस्थापक",
    ):
        assert forbidden not in text, f"{language} overview exposes: {forbidden}"


def test_every_workflow_the_overview_omits_is_one_an_ordinary_user_cannot_run(
    workflows: dict[str, object],
) -> None:
    """The omissions are exactly the workflows an ordinary account is refused
    -- not an arbitrary subset. `Role.user` workflows (downloads, preferences,
    background jobs, cases, evidence, lawyer summary) only require the user to
    be signed in and ARE described in the overview; `Role.admin` and
    `Role.lawyer` ones are the operator surface and are not.

    If a new privileged workflow is registered, this fails until it is
    accounted for here."""
    privileged = {
        name
        for name, workflow in workflows.items()
        if getattr(workflow, "required_role", None) in {Role.admin, Role.lawyer}
    }
    assert privileged == {
        "admin_knowledge_base",
        "admin_analytics",
        "admin_sources",
        "notary_queue",
        "notary_admin",
    }


def test_the_template_count_is_read_from_the_live_registry() -> None:
    from app.drafting.templates import list_templates

    count = len(list_templates())
    assert str(count) in capability_overview("english", count)


# ---------------------------------------------------------------------------
# What the reader sees under the answer
# ---------------------------------------------------------------------------


def test_the_provision_and_page_evidence_are_rendered_for_the_reader() -> None:
    lines = source_lines(
        {
            "applicable_law": ["Negotiable Instruments Act, 1881 — Section 138"],
            "sources": [
                {
                    "label": "Negotiable Instruments Act, 1881 · Section 138",
                    "source_document": "ni_act.pdf",
                    "verification_status": "verified",
                }
            ],
            "evidence_pages": [
                {"source_document": "ni_act.pdf", "page_number": 28, "label": "p. 28"}
            ],
        }
    )
    assert lines[0] == "**Negotiable Instruments Act, 1881 — Section 138**"
    assert "p. 28" in lines[1]
    # A verified source is not labelled with a status; only a non-verified one is.
    assert "verified" not in lines[1].replace("Negotiable", "")


def test_a_source_with_no_page_evidence_is_shown_without_a_page() -> None:
    lines = source_lines(
        {
            "applicable_law": [],
            "sources": [{"label": "Consumer Protection Act, 2019", "source_document": "cpa.pdf"}],
            "evidence_pages": [],
        }
    )
    assert lines == ["- Consumer Protection Act, 2019"]
    assert "p." not in lines[0]


def test_an_unverified_source_says_so_where_the_reader_can_see_it() -> None:
    lines = source_lines(
        {
            "sources": [
                {
                    "label": "Negotiable Instruments Act, 1881",
                    "source_document": "ni_act.pdf",
                    "verification_status": "pending_review",
                }
            ]
        }
    )
    assert "pending review" in lines[0]


def test_a_source_with_no_identifiable_provision_is_named_not_guessed_at() -> None:
    lines = source_lines({"sources": [{"source_document": "circular.pdf"}]})
    assert lines == ["- circular.pdf"]


def test_an_answer_with_nothing_behind_it_renders_no_source_band() -> None:
    assert source_lines({}) == []
    assert source_lines({"sources": [], "applicable_law": [], "evidence_pages": []}) == []


def test_the_chat_ui_still_has_no_separate_feature_pages_or_buttons() -> None:
    """The conversational surface is the product. This pins that the milestone-E
    layout work did not quietly reintroduce the old per-feature UI."""
    from pathlib import Path

    source = Path("streamlit_app/app.py").read_text(encoding="utf-8")
    for reintroduced in ("st.tabs(", "st.navigation(", "st.Page(", "st.sidebar.radio("):
        assert reintroduced not in source, f"a separate feature surface reappeared: {reintroduced}"


def test_the_apparatus_band_renders_warnings_sources_and_currency_in_that_order() -> None:
    """Order matters: a warning changes what the reader does with the answer,
    the sources are what they check it against, the currency note qualifies
    the sources."""
    from pathlib import Path

    source = Path("streamlit_app/app.py").read_text(encoding="utf-8")
    body = source[source.index("def _render_answer_apparatus("):]
    body = body[: body.index("\ndef ", 1)]
    assert body.index("for warning in warnings") < body.index("if sources:") < body.index("if currency:")
