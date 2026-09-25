"""No production-callable function may raise a *setup* `NotImplementedError`.

Phase 1 removed `app/rag/hybrid_legal_retriever.py`, whose
`get_hybrid_legal_retriever()` was documented as a FastAPI dependency provider
("reference via `Depends(...)`") but raised
`NotImplementedError("wire real dense_client/sparse_client instances")`. Wired
into a route, that is a hard 500 on every request to it; left unwired, it is a
trap for the next person looking for the real retrieval path. Either way it has
no business shipping.

The distinction this module enforces is between:

  * an ABSTRACT METHOD -- an empty base-class body whose whole purpose is to be
    overridden (`app/llm/base.py`, `app/rag/vector_store.py`). Legitimate, and
    unreachable through its subclasses.
  * a SETUP STUB -- a concrete, non-abstract function that raises because
    somebody has not finished wiring it up yet.

Only the second is a defect, so the check keys on the shape of the enclosing
function rather than on the exception alone.
"""

import ast
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent / "app"


def _raises_not_implemented(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for statement in node.body:
        if not isinstance(statement, ast.Raise) or statement.exc is None:
            continue
        raised = statement.exc
        name = raised.func if isinstance(raised, ast.Call) else raised
        if isinstance(name, ast.Name) and name.id == "NotImplementedError":
            return True
    return False


def _is_abstract_shape(node: ast.FunctionDef | ast.AsyncFunctionDef, class_bases: list[str]) -> bool:
    """An override-me base method: decorated `@abstractmethod`, or a body of
    nothing but a docstring and the bare raise, inside a class that is itself a
    declared interface (Protocol/ABC) or whose name reads as a base."""
    decorators = {
        decorator.attr if isinstance(decorator, ast.Attribute) else getattr(decorator, "id", "")
        for decorator in node.decorator_list
    }
    if "abstractmethod" in decorators:
        return True
    if not class_bases:
        return False
    if any(base in {"ABC", "Protocol"} for base in class_bases):
        return True
    body = [item for item in node.body if not (isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant))]
    return len(body) == 1 and isinstance(body[0], ast.Raise)


def _collect_setup_stubs(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []

    def walk(node: ast.AST, class_bases: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                bases = [base.id for base in child.bases if isinstance(base, ast.Name)]
                walk(child, bases)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                if _raises_not_implemented(child) and not _is_abstract_shape(child, class_bases):
                    offenders.append(f"{path.relative_to(APP_ROOT.parent)}::{child.name} (line {child.lineno})")
                walk(child, class_bases)
            else:
                walk(child, class_bases)

    walk(tree, [])
    return offenders


def test_no_production_callable_setup_stub_remains() -> None:
    offenders: list[str] = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        offenders.extend(_collect_setup_stubs(path))
    assert offenders == [], (
        "These functions raise NotImplementedError but are not abstract base methods. "
        "Either implement them or delete them -- a half-wired provider is a 500 waiting "
        f"to be routed: {offenders}"
    )


def test_the_removed_hybrid_retriever_module_is_gone() -> None:
    """Its four stages all exist on the real path; a second, unwired copy of
    them (with a cross-encoder that silently returned fabricated scores when
    sentence-transformers was missing) was a correctness hazard, not a feature.
    """
    assert not (APP_ROOT / "rag" / "hybrid_legal_retriever.py").exists()


def test_the_abstract_bases_it_replaced_are_still_abstract() -> None:
    """Guards the check above from being satisfied by deleting the legitimate
    abstract methods instead of the stubs."""
    from app.llm.base import LLMProvider
    from app.rag.vector_store import VectorStore

    for base, method in ((LLMProvider, "chat"), (VectorStore, "search")):
        assert _raises_not_implemented_in_source(base, method)


def _raises_not_implemented_in_source(cls: type, method: str) -> bool:
    import inspect
    import textwrap

    # `getsource` keeps the method's original class-body indentation, which is
    # not parseable on its own -- dedent before parsing.
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(cls, method))))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef)
    return _raises_not_implemented(function)
