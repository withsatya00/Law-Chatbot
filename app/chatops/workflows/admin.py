"""Administration, by talking.

Twenty-six admin routes -- knowledge base, analytics and legal-source
governance -- had no conversational path. These three workflows cover them,
subject to three rules that are stricter here than anywhere else in the
package:

* **Role first.** Every workflow declares `Role.admin`, checked by the
  orchestrator before a single fact is collected, and the underlying routes
  keep their own `require_admin` dependency.
* **Confirmation for anything that changes state.** Re-indexing, flushing
  the cache, applying a reconciliation, assigning ownership and recording a
  source review all pass through an explicit yes.
* **Verification is never automatic.** Marking a legal source verified is an
  assertion that a human compared it against the issuing authority's own
  text, so this workflow refuses to record it without an evidence URL --
  the same rule `LegalUpdateService.review` enforces underneath.

Results are summarised and capped. An admin listing in a chat bubble is a
summary with a pointer, not a data dump.
"""

import asyncio
import re
from typing import Any

import structlog

from app.chatops import intents, selection
from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.registry import register_workflow
from app.core import clock
from app.core.security import Role
from app.schemas.phase3 import SourceReviewRequest
from app.services import admin_operations
from app.services.analytics_service import AnalyticsService
from app.services.phase3 import AuditService, EvaluationService, LegalUpdateService

log = structlog.get_logger(__name__)

_URL = re.compile(r"https?://[^\s)>\]]+")


@register_workflow
class AdminKnowledgeBaseWorkflow(ChatWorkflow):
    """KB status, staging, re-indexing, reconciliation, cache and ownership."""

    name = "admin_knowledge_base"
    title = "knowledge-base administration"
    required_role = Role.admin

    def matches_intent(self, message: str, language: str) -> float:
        score = intents.score(intents.ADMIN_KNOWLEDGE_BASE, message)
        # These phrases also contain generic words such as "files", which
        # can legitimately match the user's download-list workflow. The
        # explicit KB review operation is stronger evidence and must win for
        # an admin instead of producing an unrelated ambiguity prompt.
        if re.search(
            r"\b(?:needs?[\s_-]*review|review\s*queue|path[_ -]*missing|missing\s+files?|"
            r"files?\b[^.?!]{0,12}\b(?:approv\w*|reject\w*|archive))\b",
            message,
            re.IGNORECASE,
        ):
            return max(score, 0.95)
        return score

    def needs_confirmation(self, context: WorkflowContext) -> bool:
        return str(context.facts.get("action", "")) in {
            "reindex", "flush_cache", "reconcile_apply", "assign_owner",
            "staging_approve", "staging_archive", "staging_close_missing",
        }

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        message = context.message
        if not context.facts.get("action"):
            facts["action"] = self._detect_action(message)
        action = str(facts.get("action") or context.facts.get("action") or "")

        if action == "job_status" and not context.facts.get("job_id"):
            job = re.search(r"\b([0-9a-f]{8}-[0-9a-f-]{27,})\b", message, re.IGNORECASE)
            if job:
                facts["job_id"] = job.group(1)

        if action in {"staging_approve", "staging_archive"}:
            facts.update(await self._extract_review_selection(context, action, message))
        if action == "staging_close_missing":
            facts.update(await self._extract_missing_selection(context, message))

        if action == "assign_owner" and not context.facts.get("source_document"):
            parked = selection.from_dicts(list(context.facts.get("_choices") or []))
            if parked:
                chosen = selection.resolve(message, parked)
                if chosen is not None:
                    facts["source_document"] = chosen.key
            else:
                report = await admin_operations.unowned_documents()
                documents = [str(item.get("source_document", item)) for item in report["documents"]]
                if documents:
                    facts["_choices"] = [
                        selection.Choice(key=name, label=name).as_dict() for name in documents[:20]
                    ]
        return facts

    async def _extract_missing_selection(self, context: WorkflowContext, message: str) -> dict[str, Any]:
        """Resolve one open path-missing ledger finding; never a disk file.

        These rows are intentionally separate from the five readable review
        candidates. Closing one records an administrative explanation but
        does not move, index, or fabricate a replacement document.
        """
        facts: dict[str, Any] = {}
        parked = selection.from_dicts(list(context.facts.get("_missing_choices") or []))
        if not parked:
            listing = await admin_operations.staging_records("all")
            records = [
                record for record in listing["records"]
                if record.get("_id") and record.get("path_missing") and not record.get("resolution")
            ][:20]
            choices = [
                selection.Choice(
                    key=str(record["_id"]),
                    label=str(record.get("original_filename") or "missing file"),
                    detail=f"id `{record['_id']}` · {record.get('reason') or 'recorded file is missing'}",
                ).as_dict()
                for record in records
            ]
            if choices:
                facts["_missing_choices"] = choices
                facts["_missing_records"] = {
                    str(record["_id"]): self._review_fields(record) for record in records
                }
                parked = selection.from_dicts(choices)
        if parked and not context.facts.get("staging_id"):
            chosen = selection.resolve(message, parked)
            if chosen is not None:
                facts["staging_id"] = chosen.key
                facts["staging_filename"] = chosen.label
        if not context.facts.get("reason"):
            reason = self._extract_reason(message)
            if reason:
                facts["reason"] = reason
        return facts

    async def _extract_review_selection(
        self, context: WorkflowContext, action: str, message: str
    ) -> dict[str, Any]:
        """Resolves "file 2" to one exact staging record, and picks up a
        rejection reason if one was given in the same breath.

        The candidate list is rebuilt from `status="needs_review"` ONLY, so a
        failed, corrupt or already-indexed record is never in the list to be
        chosen -- the status rule is enforced by the candidate set as well as
        by `KnowledgeBaseIngestionService.approve_needs_review` underneath.
        `selection.resolve` returns None when a message names several records,
        which is what makes "sab approve kar do" re-ask instead of acting: the
        no-bulk-approval rule falls out of the resolver rather than needing a
        separate check.
        """
        facts: dict[str, Any] = {}
        parked = selection.from_dicts(list(context.facts.get("_review_choices") or []))
        if not parked:
            staging = await admin_operations.staging_records("needs_review")
            records = [record for record in staging["records"][:20] if record.get("_id")]
            candidates = [
                selection.Choice(
                    key=str(record["_id"]),
                    label=str(record.get("original_filename") or "file"),
                    detail=self._review_detail(record),
                ).as_dict()
                for record in records
            ]
            if candidates:
                facts["_review_choices"] = candidates
                # Parked alongside the choices so the confirmation summary can
                # show hash and provenance without a second query.
                facts["_review_records"] = {
                    str(record["_id"]): self._review_fields(record) for record in records
                }
                parked = selection.from_dicts(candidates)

        if parked and not context.facts.get("staging_id"):
            chosen = selection.resolve(message, parked)
            if chosen is not None:
                facts["staging_id"] = chosen.key
                facts["staging_filename"] = chosen.label

        if action == "staging_archive" and not context.facts.get("reason"):
            reason = self._extract_reason(message)
            if reason:
                facts["reason"] = reason
        return facts

    @staticmethod
    def _extract_reason(message: str) -> str:
        """A rejection reason stated inline ("reject karo kyunki scan
        unreadable hai"). Anything shorter is not a reason and is left for
        `next_question` to ask for properly."""
        match = re.search(
            r"(?:because|since|reason\s*(?:is|:)?|kyunki|kyuki|क्योंकि|वजह)\s+(.{4,200})$",
            message,
            re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip().strip(".\"'") if match else ""

    @staticmethod
    def _review_fields(record: dict[str, Any]) -> dict[str, str]:
        return {
            "filename": str(record.get("original_filename") or "file"),
            "content_hash": str(record.get("content_hash") or ""),
            "reason": str(record.get("reason") or ""),
            "source": str(record.get("ingestion_source") or record.get("uploaded_by") or "unknown"),
            "status": str(record.get("status") or ""),
            "file_exists": "yes" if record.get("file_exists") else "no",
        }

    @classmethod
    def _review_detail(cls, record: dict[str, Any]) -> str:
        fields = cls._review_fields(record)
        parts = [f"id `{record.get('_id')}`"]
        if fields["reason"]:
            parts.append(fields["reason"])
        parts.append(f"source {fields['source']}")
        if fields["file_exists"] == "no":
            parts.append("**file missing**")
        return " · ".join(parts)

    @staticmethod
    def _detect_action(message: str) -> str:
        lowered = message.lower()
        if re.search(r"\bassign\b.{0,20}\bowner|\bowner\b.{0,20}\bassign", lowered):
            return "assign_owner"
        if re.search(r"\bunowned\b", lowered):
            return "unowned"
        if re.search(r"\bcache\b", lowered):
            return "flush_cache" if re.search(r"flush|purge|clear", lowered) else "status"
        if re.search(r"\breconcil", lowered):
            return "reconcile_apply" if re.search(r"\bapply|fix|prune\b", lowered) else "reconcile_dry_run"
        if re.search(r"\bre-?index", lowered):
            return "reindex"
        if re.search(r"\bjob\b", lowered):
            return "job_status"
        if re.search(r"\bclose\w*\b", lowered) and re.search(
            r"\b(?:path[_ -]*missing|missing\s+files?|missing\s+records?)\b", lowered
        ):
            return "staging_close_missing"
        # Hinglish is how this is actually asked ("kaun si files failed hain?"),
        # so each concrete question routes to its own status filter rather than
        # dropping into the generic staging list.
        if re.search(r"\bfail(ed|ure)?\b|\bfail hui", lowered):
            return "staging_failed"
        # Approve/reject act on ONE record picked out of the numbered review
        # list, so both are checked before the plain "review" listing --
        # otherwise "file 2 approve karo" reads as another request to list.
        if re.search(r"\bapprov\w*|\bmanzoor\b|स्वीकृत", lowered):
            return "staging_approve"
        if re.search(r"\breject\w*|\barchive\b|\bkharij\b|अस्वीकृत", lowered):
            return "staging_archive"
        if re.search(r"\b(?:path[_ -]*missing|missing\s+files?|missing\s+records?)\b", lowered):
            return "staging_missing"
        if re.search(r"\breview\b|needs[_ ]review|\bpending review\b", lowered):
            return "staging_review"
        if re.search(r"\bindexed\b|index hui", lowered):
            return "staging_indexed"
        if re.search(r"\bstaging\b", lowered):
            return "staging"
        if re.search(r"\bdashboard\b", lowered):
            return "dashboard"
        return "status"

    def required_fields(self, context: WorkflowContext) -> list[str]:
        action = str(context.facts.get("action", ""))
        if action == "assign_owner":
            return ["action", "source_document"]
        if action == "job_status":
            return ["action", "job_id"]
        if action == "staging_approve":
            return ["action", "staging_id"]
        if action == "staging_archive":
            # The reason is required BEFORE the confirmation summary, so the
            # admin confirms a rejection they have already justified.
            return ["action", "staging_id", "reason"]
        if action == "staging_close_missing":
            return ["action", "staging_id", "reason"]
        return ["action"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        if missing[0] == "source_document":
            choices = selection.from_dicts(list(context.facts.get("_choices") or []))
            if choices:
                return "Which unowned document should I attribute?\n\n" + selection.render(choices)
            return "There are no unowned documents waiting for attribution."
        if missing[0] == "staging_id":
            choices = selection.from_dicts(
                list(context.facts.get("_review_choices") or context.facts.get("_missing_choices") or [])
            )
            if choices:
                return (
                    "Which one? Say its number.\n\n"
                    + selection.render(choices)
                    + "\n\nOne at a time — I don't approve in bulk."
                )
            return "There is nothing in the review queue right now."
        if missing[0] == "reason":
            if context.facts.get("action") == "staging_close_missing":
                return (
                    f"Why should the missing-path record for "
                    f"**{context.facts.get('staging_filename', 'this file')}** be closed? "
                    "The reason will be retained in the audit log."
                )
            return (
                f"Why is **{context.facts.get('staging_filename', 'this file')}** being rejected? "
                "The reason is recorded in the audit log alongside the archived file."
            )
        if missing[0] == "job_id":
            return "Which indexing job? Paste the job id from when it was queued."
        return (
            "What would you like — knowledge-base status, the staging queue, an index reconciliation, "
            "a re-index, or the unowned-document report?"
        )

    def summarize_for_confirmation(self, context: WorkflowContext) -> str:
        action = str(context.facts.get("action", ""))
        return {
            "reindex": (
                "- Action: re-index every new, changed and deleted file under the knowledge-base root\n"
                "- Effect: retrieval results change; this can take a while on a large corpus"
            ),
            "flush_cache": (
                "- Action: flush the entire response cache\n"
                "- Effect: the next request for every cached question goes to the model again"
            ),
            "reconcile_apply": (
                "- Action: prune BM25 entries that no longer exist in MongoDB\n"
                "- Effect: irreversible for the index; MongoDB is untouched and remains the source of truth"
            ),
            "assign_owner": (
                f"- Document: {context.facts.get('source_document')}\n"
                f"- New owner: {context.facts.get('owner_user_id') or 'none (stays globally visible)'}\n"
                "- Effect: recorded as an audited ownership decision"
            ),
            "staging_approve": self._review_summary(
                context, "Approve for indexing into the shared Knowledge Base"
            ),
            "staging_archive": self._review_summary(
                context, "Reject and archive (the file is moved, never deleted)"
            ),
            "staging_close_missing": self._missing_summary(context),
        }.get(action, f"- Action: {action}")

    def _missing_summary(self, context: WorkflowContext) -> str:
        records = dict(context.facts.get("_missing_records") or {})
        fields = dict(records.get(str(context.facts.get("staging_id")), {}))
        return "\n".join(
            [
                "- Action: Close this missing-path ledger finding",
                f"- File: {fields.get('filename') or context.facts.get('staging_filename', 'unknown')}",
                f"- Staging id: `{context.facts.get('staging_id')}`",
                f"- Reason: {context.facts.get('reason')}",
                "- Effect: ledger resolution only; no file is moved, deleted, created or indexed",
            ]
        )

    def _review_summary(self, context: WorkflowContext, headline: str) -> str:
        """Filename, hash and provenance, so the admin confirms a specific
        file rather than a list position they may have miscounted."""
        records = dict(context.facts.get("_review_records") or {})
        fields = dict(records.get(str(context.facts.get("staging_id")), {}))
        lines = [
            f"- Action: {headline}",
            f"- File: {fields.get('filename') or context.facts.get('staging_filename', 'unknown')}",
            f"- Staging id: `{context.facts.get('staging_id')}`",
            f"- Content hash: `{fields.get('content_hash') or 'not recorded'}`",
            f"- Source: {fields.get('source', 'unknown')}",
            f"- Current status: {fields.get('status') or 'needs_review'}",
        ]
        if context.facts.get("reason"):
            lines.append(f"- Reason: {context.facts['reason']}")
        if fields.get("file_exists") == "no":
            lines.append("- ⚠️ The file is not at its recorded path, so this will fail.")
        return "\n".join(lines)

    def idempotency_key(self, context: WorkflowContext) -> str:
        return (
            f"{self.name}:{context.facts.get('action')}:"
            f"{context.facts.get('source_document', '')}{context.facts.get('staging_id', '')}"
        )

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        action = str(context.facts.get("action", "status"))

        if action == "status":
            status = await admin_operations.knowledge_base_status()
            logs = await admin_operations.logs_status()
            return self._done(
                "**Knowledge base**\n"
                f"- Indexed chunks: {status['indexed_chunks']}\n"
                f"- Active documents: {status['active_documents']}\n"
                f"- Deleted documents: {status['deleted_documents']}\n"
                f"- State: {status['status']}\n\n"
                f"Logs — prompt {logs['prompt_logs']}, query {logs['query_logs']}, system {logs['system_logs']}.",
                ["staging queue", "reconcile the index", "re-index"],
            )
        if action == "dashboard":
            board = await admin_operations.kb_dashboard()
            ledger, disk = board["ledger"], board["disk"]
            return self._done(
                "**Ingestion dashboard**\n"
                f"- Ledger — indexed {ledger['indexed']} · processing {ledger['processing']}"
                f" · pending {ledger['pending']}\n"
                f"- Ledger — duplicates {ledger['duplicate']} · failed {ledger['failed']}"
                f" · needs review {ledger['needs_review']} · stale/missing {ledger['stale_or_missing']}\n"
                f"- Disk — staging {disk['staging_files']} · review queue {disk['review_queue_files']}"
                f" · knowledge base {disk['knowledge_base_files']}\n"
                f"- Storage: {board['storage_usage_bytes'] / 1_048_576:.1f} MB",
                ["staging queue", "which files failed", "which files need review"],
            )
        if action in {"staging", "staging_failed", "staging_review", "staging_indexed", "staging_missing"}:
            status_filter = {
                "staging_failed": "failed",
                "staging_review": "needs_review",
                "staging_indexed": "indexed",
                "staging_missing": "all",
            }.get(action)
            staging = await admin_operations.staging_records(status_filter)
            if action == "staging_missing":
                records = [
                    record for record in staging["records"]
                    if record.get("path_missing") and not record.get("resolution")
                ]
                if not records:
                    return self._done("There are no open missing-path records.", [])
                choices = [
                    selection.Choice(
                        key=str(record["_id"]),
                        label=str(record.get("original_filename") or "missing file"),
                        detail=f"id `{record['_id']}` · {record.get('reason') or 'recorded file is missing'}",
                    )
                    for record in records[:20]
                ]
                return self._done(
                    f"**Open missing-path records** ({len(records)})\n\n"
                    + selection.render(choices)
                    + '\n\nSay "close missing file 2 because …". One record at a time.',
                    ["close missing file 1 because it cannot be re-sourced"],
                )
            if not staging["count"]:
                return self._done(f"Nothing in the staging ledger with status '{staging['status_filter']}'.", [])
            if action == "staging_review":
                # Numbered, with the staging id, filename, reason and
                # provenance on every row -- the admin picks by position
                # ("file 2 approve karo") and never has to paste an id.
                records = [record for record in staging["records"][:20] if record.get("_id")]
                choices = [
                    selection.Choice(
                        key=str(record["_id"]),
                        label=str(record.get("original_filename") or "file"),
                        detail=self._review_detail(record),
                    )
                    for record in records
                ]
                return self._done(
                    f"**Files needing review** ({staging['count']})\n\n"
                    + selection.render(choices)
                    + "\n\nSay \"file 2 approve karo\", or \"file 2 reject karo kyunki …\" with a reason. "
                    "I act on one file at a time.",
                    ["file 1 approve karo", "staging queue"],
                )
            rows = "\n".join(
                f"- **{record.get('original_filename', 'file')}** — {record.get('status')}"
                + (f" · {record.get('reason')}" if record.get("reason") else "")
                + f" · at {record.get('current_path') or 'no recorded path'}"
                + ("" if record.get("file_exists") else " · **file missing**")
                for record in staging["records"][:10]
            )
            more = f"\n\n…and {staging['count'] - 10} more." if staging["count"] > 10 else ""
            return self._done(
                f"**Staging ledger** ({staging['status_filter']}, {staging['count']})\n\n{rows}{more}", []
            )
        if action in {"staging_approve", "staging_archive"}:
            return await self._resolve_review(context, action)
        if action == "staging_close_missing":
            return await self._resolve_missing(context)
        if action == "unowned":
            report = await admin_operations.unowned_documents()
            if not report["count"]:
                return self._done("Every indexed document has an ownership decision recorded.", [])
            rows = "\n".join(f"- {item}" for item in report["documents"][:10])
            more = f"\n\n…and {report['count'] - 10} more." if report["count"] > 10 else ""
            return self._done(
                f"**Unowned documents** ({report['count']})\n\n{rows}{more}\n\n"
                "Say \"assign owner\" to attribute one.",
                ["assign owner"],
            )
        if action == "assign_owner":
            result = await admin_operations.assign_owner(
                str(context.facts["source_document"]), context.facts.get("owner_user_id")
            )
            await AuditService().record(
                actor_user_id=context.authenticated_user_id or "",
                action="assign_document_ownership",
                resource_type="indexed_document",
                resource_id=str(result["source_document"]),
                details={
                    "source_document": result["source_document"],
                    "owner_user_id": result["owner_user_id"],
                    "chunks_updated": result["chunks_updated"],
                    "channel": "chat",
                },
            )
            return self._done(
                f"Recorded. {result['chunks_updated']} chunk(s) of **{result['source_document']}** now "
                f"belong to {result['owner_user_id'] or 'no one (globally visible, decision logged)'}.",
                ["unowned documents"],
            )
        if action == "flush_cache":
            result = await admin_operations.flush_cache()
            return self._done(f"Response cache {result['status']} ({result['scope']}).", [])
        if action in {"reconcile_dry_run", "reconcile_apply"}:
            return await self._reconcile(apply=action == "reconcile_apply")
        if action == "reindex":
            runner, job_id, root = await admin_operations.queue_reindex(None)
            # Scheduled on the running loop, the chat equivalent of the
            # route's `BackgroundTasks`. The runner records its own status on
            # the job document, so a failure is visible without this handle.
            task = asyncio.create_task(runner.run_for_job(job_id, root))
            task.add_done_callback(
                lambda finished: log.info(
                    "chat_reindex_finished",
                    job_id=job_id,
                    failed=bool(finished.cancelled() or finished.exception()),
                )
            )
            return self._done(
                f"Re-index queued for `{root}`. Job id `{job_id}` — ask me for its status any time.",
                ["indexing job status"],
            )
        job = await admin_operations.reindex_job(str(context.facts["job_id"]))
        return self._done(
            f"**Indexing job** `{job.get('_id')}` — {job.get('status')}\n"
            f"- Indexed: {len(job.get('indexed', []))} · updated: {len(job.get('updated', []))}"
            f" · deleted: {len(job.get('deleted', []))} · failed: {len(job.get('failed', []))}",
            [],
        )

    async def _resolve_review(self, context: WorkflowContext, action: str) -> WorkflowTurn:
        """Calls the existing `kb_approve`/`kb_archive` service for exactly one
        record, then records who did it.

        The audit entry is written with the ACTOR's authenticated user id from
        the JWT claims, never anything read out of the message -- an approval
        with no attributable actor is the failure this whole flow exists to
        prevent, so an unattributable turn is refused rather than recorded
        against an empty actor.
        """
        staging_id = str(context.facts.get("staging_id") or "")
        actor = context.authenticated_user_id or str(context.claims.get("sub") or "")
        if not staging_id:
            return WorkflowTurn(
                message="I need to know which file first — ask me for the review queue.",
                status="failed", allowed_actions=["which files need review"], finished=True,
            )
        if not actor:
            return WorkflowTurn(
                message="I can't record who approved this, so I won't do it. Please sign in again.",
                status="awaiting_reauth", allowed_actions=[], finished=True,
            )
        reason = str(context.facts.get("reason") or "")
        filename = str(context.facts.get("staging_filename") or "the file")
        try:
            if action == "staging_approve":
                result = await admin_operations.kb_approve(staging_id)
            else:
                result = await admin_operations.kb_archive(staging_id, reason)
        except Exception as exc:  # noqa: BLE001 - surfaced to the admin, not swallowed
            # `approve_needs_review` refuses anything that is not
            # `needs_review` (a failed or corrupt record included); that
            # refusal is the answer, so it is shown rather than retried.
            log.warning("chat_kb_review_action_failed", action=action, staging_id=staging_id, error=str(exc))
            return WorkflowTurn(
                message=f"That didn't go through: {exc}", status="failed",
                allowed_actions=["which files need review"], finished=True,
            )

        await AuditService().record(
            actor_user_id=actor,
            action="kb_approve_needs_review" if action == "staging_approve" else "kb_archive_rejected",
            resource_type="kb_staging_record",
            resource_id=staging_id,
            details={
                "staging_id": staging_id,
                "original_filename": filename,
                "reason": reason,
                "channel": "chat",
                "result_status": str(result.get("status", "")),
            },
        )
        verb = "approved and queued for indexing" if action == "staging_approve" else "rejected and archived"
        suffix = f"\n\nReason recorded: {reason}" if reason else ""
        return self._done(
            f"**{filename}** {verb}. Logged against your account.{suffix}",
            ["which files need review"],
        )

    async def _resolve_missing(self, context: WorkflowContext) -> WorkflowTurn:
        staging_id = str(context.facts.get("staging_id") or "")
        reason = str(context.facts.get("reason") or "").strip()
        actor = context.authenticated_user_id or str(context.claims.get("sub") or "")
        if not staging_id or not reason or not actor:
            return WorkflowTurn(
                message="I need one missing record, a reason, and an authenticated admin before I can close it.",
                status="failed",
                allowed_actions=["show missing files"],
                finished=True,
            )
        result = await admin_operations.kb_close_missing(staging_id, reason)
        await AuditService().record(
            actor_user_id=actor,
            action="kb_close_missing_file",
            resource_type="kb_staging_record",
            resource_id=staging_id,
            details={"reason": reason, "channel": "chat", "resolution": str(result.get("resolution", ""))},
        )
        return self._done(
            f"Closed the missing-path record for **{context.facts.get('staging_filename', 'the file')}**. "
            "Only the ledger was updated; no document was moved or indexed.",
            ["show missing files"],
        )

    async def _reconcile(self, *, apply: bool) -> WorkflowTurn:
        from app.rag.reconciliation import IndexReconciler

        reconciler = IndexReconciler()
        report = (await reconciler.apply()) if apply else (await reconciler.analyze())
        data = report.as_dict()
        body = (
            f"- Mongo chunks: {data['mongo_chunk_count']}\n"
            f"- BM25 chunks: {data['bm25_chunk_count']}"
            + (f" → {data['bm25_chunk_count_after']}" if data.get("bm25_chunk_count_after") is not None else "")
            + f"\n- Stale BM25 records: {data['stale_bm25_records']}"
            f" (with owner metadata: {data['stale_private_records']})\n"
            f"- Missing from BM25: {data['missing_from_bm25']}\n"
            f"- Duplicate chunk ids: {data['duplicate_chunk_ids']}\n"
            f"- Ownership problems: {data['ownership_metadata_problems']}\n"
            f"- Drifted: {data['drifted']}"
        )
        heading = "**Reconciliation applied**" if apply else "**Reconciliation dry run** (nothing was changed)"
        follow = "" if apply else "\n\nSay \"apply the reconciliation\" to prune the stale entries."
        return self._done(f"{heading}\n\n{body}{follow}", [] if apply else ["apply the reconciliation"])

    @staticmethod
    def _done(message: str, actions: list[str]) -> WorkflowTurn:
        return WorkflowTurn(message=message, status="completed", allowed_actions=actions, finished=True)


@register_workflow
class AdminAnalyticsWorkflow(ChatWorkflow):
    """The admin dashboard and the unanswered-question queue."""

    name = "admin_analytics"
    title = "admin analytics"
    required_role = Role.admin

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.ADMIN_ANALYTICS, message)

    def needs_confirmation(self, context: WorkflowContext) -> bool:
        return str(context.facts.get("action", "")) == "review"

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        lowered = context.message.lower()
        if not context.facts.get("action"):
            if re.search(r"\bmark\b|\breview(?:ed)?\b|\bresolved?\b", lowered):
                facts["action"] = "review"
            elif re.search(r"unanswered|queue|gap", lowered):
                facts["action"] = "queue"
            else:
                facts["action"] = "dashboard"
        action = str(facts.get("action") or context.facts.get("action") or "")
        if action == "review":
            if not context.facts.get("message_id"):
                parked = selection.from_dicts(list(context.facts.get("_choices") or []))
                if parked:
                    chosen = selection.resolve(context.message, parked)
                    if chosen is not None:
                        facts["message_id"] = chosen.key
                        facts["message_label"] = chosen.label
                else:
                    items = await AnalyticsService().unanswered_queue(
                        days=30, status="pending", limit=10, skip=0
                    )
                    if items:
                        facts["_choices"] = [
                            selection.Choice(
                                key=str(item.get("message_id", "")),
                                label=str(item.get("question", ""))[:70],
                            ).as_dict()
                            for item in items
                            if item.get("message_id")
                        ]
            if not context.facts.get("review_status"):
                facts["review_status"] = "resolved" if "resolved" in lowered else "reviewed"
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        if str(context.facts.get("action", "")) == "review":
            return ["action", "message_id"]
        return ["action"]

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        choices = selection.from_dicts(list(context.facts.get("_choices") or []))
        if choices:
            return "Which queue entry?\n\n" + selection.render(choices)
        return "There is nothing pending in the unanswered queue right now."

    def summarize_for_confirmation(self, context: WorkflowContext) -> str:
        return (
            f"- Entry: {context.facts.get('message_label', context.facts.get('message_id'))}\n"
            f"- Mark as: {context.facts.get('review_status')}\n"
            "- Effect: recorded as triaged. No document is indexed and no answer changes."
        )

    def idempotency_key(self, context: WorkflowContext) -> str:
        return f"{self.name}:review:{context.facts.get('message_id')}:{context.facts.get('review_status')}"

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        service = AnalyticsService()
        action = str(context.facts.get("action", "dashboard"))

        if action == "review":
            updated = await service.mark_reviewed(
                str(context.facts["message_id"]), str(context.facts.get("review_status", "reviewed")), None
            )
            if not updated:
                return WorkflowTurn(
                    message="That queue entry no longer exists.", status="failed", finished=True
                )
            return WorkflowTurn(
                message=(
                    f"Marked as {context.facts.get('review_status')}. Indexing the document that answers it "
                    "is still a separate, deliberate step."
                ),
                status="completed", allowed_actions=["unanswered queue"], finished=True,
            )
        if action == "queue":
            items = await service.unanswered_queue(days=30, status="pending", limit=10, skip=0)
            if not items:
                return WorkflowTurn(
                    message="The unanswered queue is empty.", status="completed", finished=True
                )
            rows = "\n".join(f"- {str(item.get('question', ''))[:90]}" for item in items)
            return WorkflowTurn(
                message=f"**Unanswered questions** ({len(items)} pending)\n\n{rows}",
                status="completed", allowed_actions=["mark one reviewed"], finished=True,
            )

        board = await service.dashboard(days=30)
        gaps = board.get("knowledge_gaps") or []
        failed = board.get("failed_queries") or []
        low = board.get("low_confidence_responses") or []
        return WorkflowTurn(
            message=(
                "**Admin dashboard** (last 30 days)\n"
                f"- Knowledge gaps: {len(gaps)}\n"
                f"- Failed queries: {len(failed)}\n"
                f"- Low-confidence answers: {len(low)}\n"
                f"- Missing documents flagged: {len(board.get('missing_documents') or [])}\n\n"
                "Ask for the unanswered queue to work through them one at a time."
            ),
            status="completed",
            allowed_actions=["unanswered queue", "knowledge base status"],
            finished=True,
        )


@register_workflow
class AdminSourceGovernanceWorkflow(ChatWorkflow):
    """The legal-source registry, evaluation runs and the audit log."""

    name = "admin_sources"
    title = "legal-source governance"
    required_role = Role.admin

    def matches_intent(self, message: str, language: str) -> float:
        return intents.score(intents.ADMIN_SOURCES, message)

    def needs_confirmation(self, context: WorkflowContext) -> bool:
        return str(context.facts.get("action", "")) in {"review", "evaluate"}

    async def extract_facts(self, context: WorkflowContext) -> dict[str, Any]:
        facts: dict[str, Any] = {}
        lowered = context.message.lower()
        if not context.facts.get("action"):
            if re.search(r"\baudit\s*log", lowered):
                facts["action"] = "audit"
            elif re.search(r"\bevaluation|\bbenchmark", lowered):
                facts["action"] = "evaluate"
            elif re.search(r"\bverify\b|\breject\b|\breview\b", lowered):
                facts["action"] = "review"
            else:
                facts["action"] = "list"
        action = str(facts.get("action") or context.facts.get("action") or "")

        if action == "review":
            if not context.facts.get("decision"):
                if "reject" in lowered:
                    facts["decision"] = "rejected"
                elif "verify" in lowered or "verified" in lowered:
                    facts["decision"] = "verified"
                else:
                    facts["decision"] = "pending_review"
            decision = str(facts.get("decision") or context.facts.get("decision") or "")
            if decision == "verified" and not context.facts.get("evidence_url"):
                found = _URL.search(context.message)
                if found:
                    facts["evidence_url"] = found.group(0)
            if not context.facts.get("source_id"):
                parked = selection.from_dicts(list(context.facts.get("_choices") or []))
                if parked:
                    chosen = selection.resolve(context.message, parked)
                    if chosen is not None:
                        facts["source_id"] = chosen.key
                        facts["source_label"] = chosen.label
                else:
                    sources = await LegalUpdateService().list()
                    if sources:
                        facts["_choices"] = [
                            selection.Choice(
                                key=source.source_id,
                                label=f"{source.act_name} {source.section_number}".strip(),
                                detail=source.verification_status,
                            ).as_dict()
                            for source in sources[:20]
                        ]
        return facts

    def required_fields(self, context: WorkflowContext) -> list[str]:
        if str(context.facts.get("action", "")) != "review":
            return ["action"]
        required = ["action", "source_id", "decision"]
        # A verification claim without evidence is not recorded at all.
        if str(context.facts.get("decision", "")) == "verified":
            required.append("evidence_url")
        return required

    def next_question(self, context: WorkflowContext, missing: list[str]) -> str:
        field_name = missing[0]
        if field_name == "source_id":
            choices = selection.from_dicts(list(context.facts.get("_choices") or []))
            if choices:
                return "Which source are you reviewing?\n\n" + selection.render(choices)
            return "No legal sources are registered yet."
        if field_name == "evidence_url":
            return (
                "Marking a source verified records that you compared it against the issuing authority's own "
                "text. Paste the URL of that text and I will store it with your decision."
            )
        return "Would you like the source registry, an evaluation run, or the audit log?"

    def summarize_for_confirmation(self, context: WorkflowContext) -> str:
        action = str(context.facts.get("action", ""))
        if action == "evaluate":
            return (
                "- Action: run the multilingual evaluation benchmark\n"
                "- Effect: read-only; records a run in the evaluation history"
            )
        return (
            f"- Source: {context.facts.get('source_label', context.facts.get('source_id'))}\n"
            f"- Decision: {context.facts.get('decision')}\n"
            f"- Evidence: {context.facts.get('evidence_url') or 'not applicable for this decision'}\n"
            "- Effect: recorded against your admin account, with today's date, in the source registry"
        )

    def idempotency_key(self, context: WorkflowContext) -> str:
        return (
            f"{self.name}:{context.facts.get('action')}:{context.facts.get('source_id', '')}"
            f":{context.facts.get('decision', '')}"
        )

    async def execute(self, context: WorkflowContext) -> WorkflowTurn:
        action = str(context.facts.get("action", "list"))
        service = LegalUpdateService()

        if action == "audit":
            events = await AuditService().audit.find_recent(20)
            if not events:
                return WorkflowTurn(message="No audit events recorded yet.", status="completed", finished=True)
            rows = "\n".join(
                f"- `{event.get('created_at')}` **{event.get('action')}** ({event.get('actor_role', '—')})"
                for event in events
            )
            return WorkflowTurn(
                message=f"**Audit log** (most recent first)\n\n{rows}\n\n_Audit entries are append-only._",
                status="completed", finished=True,
            )
        if action == "evaluate":
            run = await EvaluationService().run()
            return WorkflowTurn(
                message=(
                    f"**Evaluation run** `{run.run_id}` — {run.status}\n"
                    f"- Cases: {run.cases} · failures: {len(run.failures)}\n"
                    + "".join(f"- {name}: {value:.2f}\n" for name, value in sorted(run.scores.items()))
                ),
                status="completed", finished=True,
            )
        if action == "review":
            decision = str(context.facts["decision"])
            updated = await service.review(
                str(context.facts["source_id"]),
                str(context.authenticated_user_id or ""),
                SourceReviewRequest(
                    verification_status=decision,  # type: ignore[arg-type]
                    last_verified_date=clock.today(),
                    evidence_url=str(context.facts.get("evidence_url", "")),
                    review_notes=f"Recorded through chat by {context.authenticated_user_id or 'an administrator'}.",
                ),
            )
            return WorkflowTurn(
                message=(
                    f"**{updated.act_name} {updated.section_number}** is now `{updated.verification_status}`."
                    + (f" Evidence: {updated.evidence_url}" if updated.evidence_url else "")
                ),
                status="completed", allowed_actions=["legal sources"], finished=True,
            )

        sources = await service.list()
        if not sources:
            return WorkflowTurn(
                message="The legal-source registry is empty.", status="completed", finished=True
            )
        stale = [source for source in sources if source.stale]
        unverified = [source for source in sources if source.verification_status != "verified"]
        rows = "\n".join(
            f"- **{source.act_name} {source.section_number}**".rstrip()
            + f" — {source.verification_status}"
            + (" · stale" if source.stale else "")
            for source in sources[:10]
        )
        more = f"\n\n…and {len(sources) - 10} more." if len(sources) > 10 else ""
        return WorkflowTurn(
            message=(
                f"**Legal sources** ({len(sources)}; {len(unverified)} not verified, {len(stale)} stale)\n\n"
                f"{rows}{more}\n\n"
                "Nothing here is verified automatically — say \"verify\" with the authority's URL to record a review."
            ),
            status="completed",
            allowed_actions=["review a source", "run the evaluation"],
            finished=True,
        )
