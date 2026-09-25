"""Document intelligence, by talking.

Summarising an uploaded document, listing its risky clauses, and choosing
between several uploads -- all against documents the CONVERSATION uploaded,
resolved to real ids internally so the user never types one.

Two boundaries are load-bearing:

* A message that also asks for a draft ("review this PDF and draft a legal
  notice") is a multi-intent chain owned by
  `app/services/workflow_orchestrator.py`. These workflows decline it, so
  that flow keeps working exactly as it did.
* Ownership is `document_insight.load_document`'s job, which calls the one
  `ensure_document_access` rule. Nothing here decides who may read what.
"""

import re
from typing import Any

from app.chatops import intents, selection
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.registry import register_workflow
from app.schemas.document import DocumentAnalysisRequest
from app.services import document_insight
from app.services.document_service import DocumentService

# "…and draft a legal notice" -- the tell that this is a chain, not a
# standalone document question.
_ALSO_WANTS_A_DRAFT = re.compile(
    r"\b(?:draft|prepare|generate|write|banao|bana\s*do|likho)\b[^.?!]{0,30}"
    r"\b(?:notice|complaint|application|affidavit|agreement|letter|petition)\b",
    re.IGNORECASE,
)


def _conversation_documents(context: WorkflowContext) -> list[selection.Choice]:
    """Documents uploaded in THIS conversation, newest last.

    Only this list is selectable. A user cannot reach a document by naming an
    id they happen to know: it would not be in the candidate list, and
    `load_document` would refuse it besides.
    """
    uploads = context.memory.get("uploaded_documents") or []
    choices = [
        selection.Choice(
            key=str(item.get("document_id", "")),
            label=str(item.get("filename", "document")),
            detail=str(item.get("uploaded_at", ""))[:10],
        )
        for item in uploads
        if isinstance(item, dict) and item.get("document_id")
    ]
    if choices:
        return choices
    # A conversation from before the list existed still knows its most recent
    # upload, so the single-document case keeps working.
    latest = context.memory.get("last_uploaded_document_id")
    return [selection.Choice(key=str(latest), label="your uploaded document")] if latest else []


class _DocumentWorkflow(ChatWorkflow):
    """Shared "which document?" resolution for the document workflows."""

    # Naming a document explicitly outranks an open draft -- see
    # `ChatWorkflow.may_interrupt_draft`.
    may_interrupt_draft = True

    async def _resolve_document(self, context: WorkflowContext) -> dict[str, Any]:
        if context.facts.get("document_id"):
            return {}
        parked = selection.from_dicts(list(context.facts.get("_choices") or []))
        if parked:
            chosen = selection.resolve(context.message, parked)
            if chosen is None:
                return {}
            context.memory["last_uploaded_document_id"] = chosen.key
            return {"document_id": chosen.key, "document_label": chosen.label}
        choices = _conversation_documents(context)
        if not choices:
            return {}
        if len(choices) == 1:
            context.memory["last_uploaded_document_id"] = choices[0].key
            return {"document_id": choices[0].key, "document_label": choices[0].label}
        # Draft-editing-parity fix (2026-09-12, PDF Q&A acceptance pass): an
        # explicit reference resolves immediately even on the FIRST message
        # ("summarise the loan agreement", "PDF B mein kya likha hai") --
        # previously only a PARKED disambiguation reply (a SECOND turn,
        # after this workflow had already asked "which document?") was ever
        # checked against `selection.resolve`; naming the document up front
        # still fell through to asking again.
        named = selection.resolve(context.message, choices)
        if named is not None:
            context.memory["last_uploaded_document_id"] = named.key
            return {"document_id": named.key, "document_label": named.label}
        # `resolve` returning None collapses two different situations into
        # one: the message names no document at all (safe to assume
        # context), or it TRIES to name one but matches more than one
        # ambiguously (confirmed live: "the loan agreement" against
        # ["rent-agreement.pdf", "loan-agreement.pdf"] shares the word
        # "agreement" with both). `selection.overlapping` tells them apart --
        # an ambiguous naming attempt must ask, never silently fall back to
        # whichever document was last discussed and risk answering from the
        # WRONG one.
        if selection.overlapping(context.message, choices):
            return {"_choices": [choice.as_dict() for choice in choices]}
        # No reference to any document at all: stick with whichever one was
        # last resolved (`last_uploaded_document_id` doubles as "document
        # currently in focus" -- every branch above keeps it in sync, not
        # just uploads) rather than asking "which one?" again for an
        # ordinary follow-up ("explain this clause"). Load-bearing for
        # follow-up continuity: a completed workflow is POPPED off the stack
        # the moment it finishes (`chatops.state.finish`), so nothing else
        # survives across turns to remember what the conversation was just
        # discussing.
        active_id = context.memory.get("last_uploaded_document_id")
        if active_id:
            active_choice = next((choice for choice in choices if choice.key == str(active_id)), None)
            if active_choice is not None:
                return {"document_id": active_choice.key, "document_label": active_choice.label}
        return {"_choices": [choice.as_dict() for choice in choices]}

    def _which_document_question(self, context: WorkflowContext) -> str:
        choices = selection.from_dicts(list(context.facts.get("_choices") or []))
        if choices:
            return "Which document did you mean?\n\n" + selection.render(choices)
        return "Attach the document here and I will read it."

    @staticmethod
    def _upload_needed() -> WorkflowTurn:
        return WorkflowTurn(
            message="Attach the document to this chat and I will read it straight away.",
            status="awaiting_upload",
            upload_required=True,
        )


@register_workflow
class DocumentSummaryWorkflow(_DocumentWorkflow):
    """"Summarise this agreement" / "PDF ka summary do"."""

    name = "document_summary"
    title = "document summary"

    def matches_intent(self, message: str, language: str) -> float:
        if _ALSO_WANTS_A_DRAFT.search(message or ""):
            return 0.0
        return intents.score(intents.DOCUMENT_SUMMARY, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return await self._resolve_document(context)

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["document_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return self._which_document_question(context)

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        document = await document_insight.load_document(
            str(context.facts["document_id"]), context.authenticated_user_id, context.session_id
        )
        analysis = await DocumentService().analyze(
            DocumentAnalysisRequest(
                document_id=document.document_id,
                analysis_type="summary",
                language=context.language,
                session_id=context.session_id,
            ),
            authenticated_user_id=context.authenticated_user_id,
        )
        dates = "\n".join(f"- {item}" for item in analysis.important_dates[:5])
        names = ", ".join(analysis.important_names[:6])
        evidence = self._evidence_note(document)
        return WorkflowTurn(
            message=(
                f"**{document.filename}**\n\n{analysis.executive_summary}\n\n"
                + (f"**Parties named:** {names}\n" if names else "")
                + (f"\n**Dates it mentions:**\n{dates}\n" if dates else "")
                + evidence
            ),
            status="completed",
            allowed_actions=["show the risky clauses", "review it against a checklist", "ask a question about it"],
            warnings=list(analysis.missing_information[:3]),
            finished=True,
        )

    @staticmethod
    def _evidence_note(document: document_insight.LoadedDocument) -> str:
        if document.has_page_evidence:
            pages = {chunk.page_number for chunk in document.chunks if chunk.page_number}
            return f"\n_Read from {len(pages)} page(s) of the file; ask about any clause and I will cite its page._"
        return (
            "\n_This file carries no page numbers I can cite — it was indexed without page capture, "
            "or its format has no pages. I will quote the text instead._"
        )


@register_workflow
class RiskyClauseWorkflow(_DocumentWorkflow):
    """"What are the risky clauses?" with the page each was read from."""

    name = "risky_clauses"
    title = "risky-clause review"

    def matches_intent(self, message: str, language: str) -> float:
        if _ALSO_WANTS_A_DRAFT.search(message or ""):
            return 0.0
        return intents.score(intents.RISKY_CLAUSES, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return await self._resolve_document(context)

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["document_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return self._which_document_question(context)

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        document = await document_insight.load_document(
            str(context.facts["document_id"]), context.authenticated_user_id, context.session_id
        )
        analysis = await DocumentService().analyze(
            DocumentAnalysisRequest(
                document_id=document.document_id,
                analysis_type="risk",
                language=context.language,
                session_id=context.session_id,
            ),
            authenticated_user_id=context.authenticated_user_id,
        )
        risky = [clause for clause in analysis.analyzed_clauses if clause.risk_level.lower() != "low"]
        if not risky and not analysis.key_risks:
            return WorkflowTurn(
                message=(
                    f"I did not find clauses I would flag in **{document.filename}**. That is not the same as "
                    "\"this document is safe\" — it means nothing matched the risk patterns I check for, and a "
                    "lawyer reading it may still see something I cannot."
                ),
                status="completed",
                allowed_actions=["summarise it", "review it against a checklist"],
                finished=True,
            )
        lines = []
        for clause in risky[:8]:
            page = document.page_for(clause.text)
            where = f" _(page {page})_" if page else ""
            lines.append(
                f"- **{clause.name}** — {clause.risk_level} risk{where}\n"
                f"  {clause.risk_reason or clause.explanation}"
            )
        extra = [f"- {risk}" for risk in analysis.key_risks[:5] if risk]
        body = "\n".join(lines) or "\n".join(extra)
        also = "\n\n**Also worth checking:**\n" + "\n".join(extra) if lines and extra else ""
        return WorkflowTurn(
            message=(
                f"**{document.filename}** — what I would question\n\n{body}{also}\n\n"
                "These are points to raise, not legal conclusions."
            ),
            status="completed",
            warnings=(
                []
                if document.has_page_evidence
                else ["This file carries no page numbers, so I cannot cite where each clause sits."]
            ),
            allowed_actions=["summarise it", "review it against a checklist", "compare it with another document"],
            finished=True,
        )


@register_workflow
class DocumentTranscribeWorkflow(_DocumentWorkflow):
    """"Isko likh ke de do proper" / "extract the text from this" / "give me
    a typed version of this" -- retype the uploaded document itself (often a
    handwritten/scanned application) into a clean, professionally formatted
    version, preserving its own facts and original language/script.

    Distinct from `DocumentSummaryWorkflow` (a short executive summary of
    what the document is about) and `DocumentQuestionWorkflow` (one specific
    question about its content): this returns the document's own content,
    retyped, not a description of it.
    """

    name = "document_transcribe"
    title = "document transcription"

    def matches_intent(self, message: str, language: str) -> float:
        if _ALSO_WANTS_A_DRAFT.search(message or ""):
            return 0.0
        return intents.score(intents.DOCUMENT_FORMAT, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return await self._resolve_document(context)

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["document_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return self._which_document_question(context)

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        document = await document_insight.load_document(
            str(context.facts["document_id"]), context.authenticated_user_id, context.session_id
        )
        result = await document_insight.transcribe_document(document, context.language)
        if result is None:
            return WorkflowTurn(
                message=(
                    f"I could not produce a reliable typed version of **{document.filename}** just now "
                    "(the transcription step failed or timed out). Please try again — I would rather say "
                    "that than guess at a document's own facts."
                ),
                status="completed", finished=True,
            )
        data = result.response
        warnings = []
        if result.injection_suspected:
            warnings.append(
                "This document's text contains wording that resembles an embedded instruction "
                "(e.g. \"ignore previous instructions\"). It was transcribed as literal document "
                "content only, and was never executed or obeyed."
            )
        if data.verification_required and data.issues:
            warnings.append(
                "Flagged for verification: "
                + "; ".join(f"{issue.field} ({issue.confidence:.0%} confidence) — {issue.reason}" for issue in data.issues[:5])
            )
        header = (
            f"**{document.filename}** — {data.document_type}"
            + (f" ({data.primary_language})" if data.primary_language else "")
        )

        from app.repositories.document_transcripts import DocumentTranscriptRepository

        await DocumentTranscriptRepository().save(
            document.document_id, context.authenticated_user_id, context.session_id, data.model_dump(),
        )
        return WorkflowTurn(
            message=f"{header}\n\n{data.formatted_text}",
            status="completed",
            warnings=warnings,
            artifact={
                "kind": "document_transcript",
                "label": "Download this document",
                "formats": ["pdf", "docx", "txt"],
                "download_path": f"/documents/{document.document_id}/transcript/export",
                "document_id": document.document_id,
                # Not every upload is tied to an authenticated user -- an
                # anonymous/session-scoped document's export route can only
                # confirm ownership via the SAME session_id it was uploaded
                # under, and the shared `render_artifact` GET path has no
                # other way to learn this session's id.
                "session_id": context.session_id,
            },
            allowed_actions=["summarise it", "show the risky clauses", "translate it"],
            finished=True,
        )


@register_workflow
class DocumentTimelineWorkflow(_DocumentWorkflow):
    """"What are the dates and deadlines in this?"

    Deadlines are the part people miss, because documents state them as rules
    ("within 30 days from the date of receipt") rather than as dates. This
    reports the rule AND asks for the anchor date instead of computing a
    confident, wrong limitation date from an assumption.
    """

    name = "document_timeline"
    title = "document dates and deadlines"

    def matches_intent(self, message: str, language: str) -> float:
        if _ALSO_WANTS_A_DRAFT.search(message or ""):
            return 0.0
        # Explicit document-timeline wording outranks the generic
        # `CASE_TIMELINE` pattern used by case management.
        return intents.score(intents.DOCUMENT_TIMELINE, message, 0.96)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return await self._resolve_document(context)

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["document_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return self._which_document_question(context)

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        from app.entity_extraction.timeline import extract_timeline

        document = await document_insight.load_document(
            str(context.facts["document_id"]), context.authenticated_user_id, context.session_id
        )
        result = extract_timeline(document.text, page_of=document.page_index())

        if not result.events and not result.undated_events:
            return WorkflowTurn(
                message=(
                    f"I could not find any dates or deadlines in **{document.filename}**. "
                    "If you know the document has them, the text may not have extracted cleanly — "
                    "tell me what it says and I will work from that."
                ),
                status="completed", finished=True,
            )

        dated = "\n".join(
            f"- **{event.normalized_date}** — {event.category.replace('_', ' ')}"
            + (f" _(page {event.source_page})_" if event.source_page else "")
            + (f" · **{event.expiry_status.replace('_', ' ')}**" if event.is_deadline else "")
            + f"\n  {event.description[:140]}"
            for event in result.events[:12]
        )
        undated = "\n".join(
            f"- \"{event.original_text}\""
            + (f" — runs from {event.relative_to}" if event.relative_to else "")
            + (f", {event.relative_days} day(s)" if event.relative_days else "")
            for event in result.undated_events[:8]
        )
        sections = [f"**{document.filename}** — chronology\n\n{dated}" if dated else ""]
        if undated:
            sections.append(
                "**Not placed on the timeline** (the document does not give the date these run from):\n"
                f"{undated}"
            )
        if result.contradictions:
            sections.append(
                "**These dates cannot all be right:**\n"
                + "\n".join(f"- {problem}" for problem in result.contradictions)
            )
        first_question = result.questions[0] if result.questions else ""
        if first_question:
            sections.append(first_question)

        return WorkflowTurn(
            message="\n\n".join(part for part in sections if part),
            status="completed",
            warnings=result.contradictions,
            allowed_actions=["summarise it", "show the risky clauses", "add these to a case"],
            finished=True,
        )


@register_workflow
class DocumentReviewWorkflow(_DocumentWorkflow):
    """"Is anything missing from my rent agreement?"

    Runs the checklist for the document's own type and distinguishes three
    outcomes — present, missing, and unable to determine — because saying
    "your agreement has no termination clause" when really the scan failed is
    a confident falsehood about a document somebody is about to sign.
    """

    name = "document_review"
    title = "document review"

    def matches_intent(self, message: str, language: str) -> float:
        if _ALSO_WANTS_A_DRAFT.search(message or ""):
            return 0.0
        return intents.score(intents.DOCUMENT_REVIEW, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return await self._resolve_document(context)

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["document_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return self._which_document_question(context)

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        from app.services import document_review

        document = await document_insight.load_document(
            str(context.facts["document_id"]), context.authenticated_user_id, context.session_id
        )
        result = document_review.review(document)

        if result.detected_type == "unknown":
            return WorkflowTurn(
                message=(
                    f"I could not tell what kind of document **{document.filename}** is, so I have not "
                    "run a checklist over it — the wrong checklist produces confident nonsense.\n\n"
                    "Tell me what it is (a rent agreement, an employment contract, an NDA, a service "
                    "agreement, a sale agreement, a partnership deed, a legal notice, an affidavit, a "
                    "complaint, or a power of attorney) and I will review it properly."
                ),
                status="completed",
                allowed_actions=["it is a rent agreement", "summarise it instead"],
                finished=True,
            )

        present = result.by_status("present")
        missing = result.by_status("missing")
        undetermined = result.by_status("unable_to_determine")

        sections = [
            (
                f"**{document.filename}** — reviewed as a "
                f"{result.detected_type.replace('_', ' ')} (confidence {result.type_confidence:.0%})\n"
                f"Overall risk: **{result.risk_level}**"
            )
        ]
        if present:
            sections.append(
                "**Present** (" + str(len(present)) + "):\n"
                + "\n".join(
                    f"- {item.label}" + (f" _(page {item.source_page})_" if item.source_page else "")
                    for item in present[:10]
                )
            )
        if missing:
            sections.append(
                "**I could not find these:**\n"
                + "\n".join(
                    f"- {item.label}" + (" — **important**" if item.critical else "")
                    for item in missing
                )
            )
        if undetermined:
            sections.append(
                "**Could not determine** (not enough text extracted to say either way):\n"
                + "\n".join(f"- {item.label}" for item in undetermined[:8])
            )
        if result.risky_clauses:
            sections.append(
                "**Wording worth questioning:**\n"
                + "\n".join(
                    f"- {item.label}"
                    + (f" _(page {item.source_page})_" if item.source_page else "")
                    + f" — {item.question}"
                    for item in result.risky_clauses[:6]
                )
            )
        if result.suggested_questions:
            sections.append(
                "**Ask about:**\n" + "\n".join(f"- {question}" for question in result.suggested_questions[:5])
            )

        return WorkflowTurn(
            message="\n\n".join(sections),
            status="completed",
            warnings=result.notes,
            allowed_actions=["show the risky clauses", "compare it with another document", "summarise it"],
            finished=True,
        )


@register_workflow
class DocumentComparisonWorkflow(_DocumentWorkflow):
    """"What changed in the new agreement?" — two documents, side by side.

    Both documents must be ones THIS conversation uploaded, and each is loaded
    through the single ownership rule. A document belonging to somebody else
    raises before any of its text is read, so nothing about it — not its
    contents, not its existence — can reach the comparison.
    """

    name = "document_compare"
    title = "document comparison"

    def matches_intent(self, message: str, language: str) -> float:
        if _ALSO_WANTS_A_DRAFT.search(message or ""):
            return 0.0
        return intents.score(intents.DOCUMENT_COMPARE, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        if context.facts.get("old_document_id") and context.facts.get("new_document_id"):
            return facts

        parked = selection.from_dicts(list(context.facts.get("_choices") or []))
        if parked:
            chosen = selection.resolve(context.message, parked)
            if chosen is not None:
                # The second pick: whichever slot is still empty.
                if not context.facts.get("old_document_id"):
                    facts["old_document_id"] = chosen.key
                    facts["old_label"] = chosen.label
                elif chosen.key != context.facts.get("old_document_id"):
                    facts["new_document_id"] = chosen.key
                    facts["new_label"] = chosen.label
            return facts

        choices = _conversation_documents(context)
        if len(choices) == 2:
            # Oldest first, which is what "what changed" means.
            facts["old_document_id"], facts["old_label"] = choices[0].key, choices[0].label
            facts["new_document_id"], facts["new_label"] = choices[1].key, choices[1].label
        elif len(choices) > 2:
            facts["_choices"] = [choice.as_dict() for choice in choices]
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["old_document_id", "new_document_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        choices = selection.from_dicts(list(context.facts.get("_choices") or []))
        if not choices:
            return (
                "I need both documents here to compare them. Attach them to this chat — I can only "
                "compare files from this conversation."
            )
        if missing[0] == "old_document_id":
            return "Which is the earlier version?\n\n" + selection.render(choices)
        return "And which is the newer one?\n\n" + selection.render(choices)

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        from app.services import document_comparison

        # Loaded one at a time, each through the ownership rule.
        old = await document_insight.load_document(
            str(context.facts["old_document_id"]), context.authenticated_user_id, context.session_id
        )
        new = await document_insight.load_document(
            str(context.facts["new_document_id"]), context.authenticated_user_id, context.session_id
        )
        result = document_comparison.compare(old, new)

        changed = [item for item in result.items if item.change_type == "changed"]
        added = [item for item in result.items if item.change_type == "added"]
        removed = [item for item in result.items if item.change_type == "removed"]

        sections = [
            f"**{old.filename} → {new.filename}**\n\n{result.executive_summary}"
        ]
        for heading, entries in (
            ("Changed", changed), ("Added in the new version", added), ("Gone from the new version", removed),
        ):
            if not entries:
                continue
            lines = []
            for item in entries[:8]:
                where = self._pages(item.old_page, item.new_page)
                lines.append(f"- **{item.label}**{where}")
                if item.old_value and item.change_type != "added":
                    lines.append(f"  - was: {item.old_value[:180]}")
                if item.new_value and item.change_type != "removed":
                    lines.append(f"  - now: {item.new_value[:180]}")
                if item.risk_note:
                    lines.append(f"  - {item.risk_note}")
            sections.append(f"**{heading}**\n" + "\n".join(lines))

        if not result.has_page_evidence:
            sections.append(
                "_One or both files carry no page numbers, so I cannot always say which page a clause is on._"
            )

        return WorkflowTurn(
            message="\n\n".join(sections),
            status="completed",
            warnings=result.unresolved,
            allowed_actions=["review the new one against a checklist", "show its risky clauses"],
            finished=True,
        )

    @staticmethod
    def _pages(old_page: int | None, new_page: int | None) -> str:
        if old_page and new_page:
            return f" _(p.{old_page} → p.{new_page})_"
        if new_page:
            return f" _(p.{new_page})_"
        if old_page:
            return f" _(p.{old_page})_"
        return ""


@register_workflow
class DocumentChooseWorkflow(_DocumentWorkflow):
    """"Which documents do you have?" — the conversation's own uploads."""

    name = "document_choose"
    title = "your uploaded documents"

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.DOCUMENT_CHOOSE, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        return {}

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return []

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return "Which document would you like to work with?"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        choices = _conversation_documents(context)
        if not choices:
            return self._upload_needed()
        return WorkflowTurn(
            message=(
                "You have uploaded:\n\n" + selection.render(choices)
                + "\n\nSay which one you want summarised, reviewed, or compared."
            ),
            status="completed",
            allowed_actions=["summarise the first one", "show the risky clauses", "compare two of them"],
            finished=True,
        )


@register_workflow
class DocumentQuestionWorkflow(_DocumentWorkflow):
    """A SPECIFIC question about an uploaded document's own content --
    "What is the notice period in this agreement?", "Does this contract
    allow subletting?", "Explain this clause" -- as opposed to every other
    workflow above, each of which reports a fixed shape (summary/risks/
    dates/checklist) regardless of what was actually asked.

    PDF Q&A acceptance pass (2026-09-12): the gap this closes -- there was
    previously no way to ask an arbitrary question about an uploaded
    document and get an answer grounded in THAT document specifically, with
    a page citation, or an honest "not in this document" when it genuinely
    is not there. The closest existing behaviour (`ChatService`'s
    "Document Analysis" conversation intent) always returns the SAME canned
    executive-summary-plus-risks analysis regardless of the question asked,
    and a question naming no analyze/summarize/review verb at all (most
    ordinary questions) never reached it in the first place -- it fell
    through to plain RAG, which can retrieve from the shared knowledge base
    AND every document this session or account owns mixed together, with no
    per-document scoping and no acknowledgement that the answer might not
    be grounded in the one document the user actually meant.

    Registered LAST among the document workflows on purpose: `DOCUMENT_
    QUESTION`'s pattern is deliberately broad (any "this/the document/pdf/
    agreement/contract ...") and can score the same 0.9 as a more specific
    workflow's own pattern on a message that names both (e.g. "What are the
    risky clauses in this contract?" also matches this pattern) --
    `chatops.registry.best_match` breaks a tie by registration order, so the
    more specific workflow must be registered FIRST to win it. Only a
    question none of the others recognise ever reaches this one.
    """

    name = "document_question"
    title = "document question"

    def matches_intent(self, message: str, language: str) -> float:
        if _ALSO_WANTS_A_DRAFT.search(message or ""):
            return 0.0
        return intents.score(intents.DOCUMENT_QUESTION, message)

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts = await self._resolve_document(context)
        result = dict(facts)
        # The ORIGINAL question must survive a "which document did you
        # mean?" round-trip -- `context.message` on a later turn is the
        # user's disambiguating reply ("the loan agreement"), not a
        # question, so it is remembered once, on the turn it was actually
        # asked, and never overwritten after that (see `execute()`, which
        # reads it back from `context.facts`, never from `context.message`).
        if not context.facts.get("question"):
            result["question"] = context.message
        return result

    def required_fields(self, context: WorkflowContext) -> list[str]:
        return ["document_id"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        return self._which_document_question(context)

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        document = await document_insight.load_document(
            str(context.facts["document_id"]), context.authenticated_user_id, context.session_id
        )
        question = str(context.facts.get("question") or context.message)
        result = await document_insight.answer_question(document, question, context.language)
        if result is None:
            return WorkflowTurn(
                message=(
                    f"I couldn't check **{document.filename}** for that just now — the answering service did "
                    "not respond in time. Nothing was changed; please try again."
                ),
                status="completed",
                finished=True,
            )
        if not result.found:
            return WorkflowTurn(
                message=(
                    f"I couldn't find that in **{document.filename}**. It may not be covered in this document, "
                    "or it may be phrased differently there — try quoting the exact wording if you have it."
                ),
                status="completed",
                allowed_actions=["summarise it", "show the risky clauses", "ask about something else in it"],
                finished=True,
            )
        page = result.page
        where = f" _(page {page})_" if page else ""
        quote_line = f"\n\n> {result.quote}{where}" if result.quote else ""
        return WorkflowTurn(
            message=f"**{document.filename}**\n\n{result.answer}{quote_line}",
            status="completed",
            warnings=(
                []
                if document.has_page_evidence or not result.quote
                else ["This file carries no page numbers, so I cannot cite where this was read from."]
            ),
            allowed_actions=["ask another question about it", "summarise it", "show the risky clauses"],
            finished=True,
        )
