"""Workflow implementations.

Importing this package registers every workflow. `app.chatops.orchestrator`
imports it, so registration happens exactly once, at the point the
orchestrator is first used -- no import-order surprises and no manual list to
keep in sync.
"""

from app.chatops.workflows import (
    account,
    admin,
    cases,
    documents,
    drafts,
    legal,
    notarization,
)

__all__ = ["account", "admin", "cases", "documents", "drafts", "legal", "notarization"]
