"""Conversational workflow orchestration.

The chat window is the product's only user-facing surface. Every capability
the application has -- drafting, document analysis, cyber-fraud triage,
jurisdiction help, notarization preparation, notary review, admin actions --
is reachable by talking to it, in any of the languages this app supports.

This package is the layer that makes that true. It is deliberately NOT an
"LLM decides what to do" layer:

* Intent detection is pattern-first (`intents.py`), with the existing
  `ConversationIntentClassifier` as a fallback. A regex that matched is a
  fact; a model that guessed is not.
* Workflow state is a deterministic machine held in conversation memory
  (`state.py`), never re-derived from the transcript by a model.
* Authorization is enforced by the existing FastAPI/service dependencies.
  A workflow declares `required_role`; the guard checks it against the
  authenticated claims. The model is never asked whether someone may act.
* Every irreversible or outward-facing action passes through an explicit
  confirmation turn (`confirm.py`).

Adding a capability to chat means writing one `ChatWorkflow` subclass and
registering it. It must NOT mean reimplementing legal logic -- workflows are
adapters over the existing services.
"""

from app.chatops.base import ChatWorkflow, WorkflowContext, WorkflowTurn
from app.chatops.registry import WORKFLOWS, get_workflow, register_workflow

__all__ = [
    "WORKFLOWS",
    "ChatWorkflow",
    "WorkflowContext",
    "WorkflowTurn",
    "get_workflow",
    "register_workflow",
]
