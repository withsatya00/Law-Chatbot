"""Schema/index setup for the E-Notarization collections.

Migration strategy for this module, stated plainly:

MongoDB is schemaless, so there is no table to create and no backfill to run
-- every collection here is NEW, and no existing collection or document is
altered by this feature. Installing it is therefore additive and reversible:
an older build simply ignores the new collections.

What DOES need applying is indexes, and one of them is a correctness
constraint rather than a performance tweak:

* `notarization_documents.verification_token` is UNIQUE and sparse. Unique
  because a verification token must identify exactly one document -- a
  collision would let one document's QR resolve to another's record. Sparse
  because the field is null until notarization, and a plain unique index
  would reject every second un-notarized document.
* `notary_accounts.registration_number` is UNIQUE, so one registration number
  cannot be claimed by two accounts.

Run at startup (idempotent) via `ensure_notarization_indexes()`. Applying it
twice is a no-op; applying it to a populated database that already violates a
unique constraint will fail loudly, which is the correct outcome -- silently
dropping the constraint would be worse than a failed startup.

ROLLBACK: drop the five collections named in `app/models/collections.py`
under "E-Notarization". Nothing outside this module reads them.
"""

import structlog

from app.database.mongodb import mongodb
from app.models.collections import (
    ESIGN_SESSIONS,
    NOTARIZATION_AUDIT_EVENTS,
    NOTARIZATION_DOCUMENTS,
    NOTARIZATION_REQUESTS,
    NOTARY_ACCOUNTS,
)

log = structlog.get_logger(__name__)

# (collection, keys, name, unique, sparse)
_INDEXES: tuple[tuple[str, list[tuple[str, int]], str, bool, bool], ...] = (
    # Correctness constraints.
    (NOTARIZATION_DOCUMENTS, [("verification_token", 1)], "verification_token_unique", True, True),
    (NOTARY_ACCOUNTS, [("registration_number", 1)], "registration_number_unique", True, False),
    # Lookup paths the service actually uses.
    (NOTARIZATION_DOCUMENTS, [("source_draft_id", 1), ("document_version", -1)], "draft_version", False, False),
    (NOTARIZATION_DOCUMENTS, [("owner_user_id", 1), ("created_at", -1)], "owner_recent", False, False),
    (NOTARY_ACCOUNTS, [("user_id", 1)], "user_id_lookup", False, False),
    (NOTARIZATION_REQUESTS, [("assigned_notary_id", 1), ("review_status", 1)], "notary_queue", False, False),
    (NOTARIZATION_REQUESTS, [("requested_by_user_id", 1), ("created_at", -1)], "requester_recent", False, False),
    (NOTARIZATION_REQUESTS, [("document_id", 1)], "request_document", False, False),
    (ESIGN_SESSIONS, [("provider", 1), ("provider_reference", 1)], "provider_reference", False, False),
    (ESIGN_SESSIONS, [("document_id", 1), ("created_at", -1)], "session_document", False, False),
    (ESIGN_SESSIONS, [("status", 1), ("created_at", -1)], "session_status", False, False),
    # Audit reads: per document (timeline) and recent-first (admin log).
    (NOTARIZATION_AUDIT_EVENTS, [("document_id", 1), ("occurred_at", 1)], "audit_document_timeline", False, False),
    (NOTARIZATION_AUDIT_EVENTS, [("action", 1), ("occurred_at", -1)], "audit_action_recent", False, False),
)


async def ensure_notarization_indexes() -> list[str]:
    """Creates the module's indexes. Idempotent; returns the names applied.

    Deliberately NOT wrapped in a try/except that swallows failures: a unique
    index that cannot be built means the data already violates a constraint
    this module's correctness depends on, and that must surface.
    """
    applied: list[str] = []
    for collection_name, keys, index_name, unique, sparse in _INDEXES:
        collection = mongodb.db[collection_name]
        existing = await collection.index_information()
        if index_name in existing:
            continue
        await collection.create_index(keys, name=index_name, unique=unique, sparse=sparse)
        applied.append(f"{collection_name}.{index_name}")
    if applied:
        log.info("notarization_indexes_applied", indexes=applied)
    return applied
