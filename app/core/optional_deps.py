"""Phase 1 item 8: keep document-format dependencies out of the import graph
of chat and drafting.

`app/rag/loader.py` and `app/rag/ocr.py` imported pypdf, python-docx, odfpy,
striprtf, BeautifulSoup, Pillow and pytesseract at module scope. Both are
reachable from `ChatService` (chat_service -> DocumentService -> rag.pipeline
-> rag.loader -> rag.ocr), so a host missing ANY ONE of them -- pytesseract in
particular, which most deployments skip because it also needs an OS-level
Tesseract install -- took down plain legal chat and drafting at import time,
features that never touch a PDF at all.

The fix is not to drop the dependencies but to bind them at the moment the
format is actually requested. A missing library then degrades to "PDF upload
is unavailable, here is what to install" on the one endpoint that needs it,
while every other feature keeps working.

`load()` caches both hits and misses, so the repeated attribute lookup costs
nothing after the first call and a missing module isn't re-probed on every
page of a batch import.
"""

import importlib
import importlib.util
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class MissingOptionalDependencyError(RuntimeError):
    """Raised when a feature is used whose optional library isn't installed.

    Carries the pip install line so the API error a user sees names the fix
    rather than a bare ImportError traceback.
    """

    def __init__(self, feature: str, module: str, install_hint: str) -> None:
        self.feature = feature
        self.module = module
        self.install_hint = install_hint
        super().__init__(
            f"{feature} is unavailable on this server because the optional dependency "
            f"'{module}' is not installed. Install it with: {install_hint}"
        )


# Every optional import this app makes, with the install line for each. Kept
# as one table so "what does this deployment actually need?" is answerable
# from a single place (see `missing_optional_dependencies()`).
_OPTIONAL: dict[str, tuple[str, str]] = {
    "pypdf": ("PDF text extraction", "pip install pypdf"),
    "docx": ("DOCX reading and export", "pip install python-docx"),
    "odf": ("ODT reading", "pip install odfpy"),
    "striprtf": ("RTF reading", "pip install striprtf"),
    "bs4": ("HTML document loading", "pip install beautifulsoup4"),
    "PIL": ("image handling for OCR", "pip install pillow"),
    "pytesseract": ("OCR of scanned documents", "pip install pytesseract (plus the Tesseract OS package)"),
    "pdf2image": ("OCR of scanned PDFs", "pip install pdf2image (plus the Poppler OS package)"),
    "weasyprint": ("PDF export", "pip install weasyprint (plus the GTK3/Pango native libraries)"),
    "pyhanko": ("digital signing of exported PDFs", "pip install pyHanko"),
    "segno": ("notarization verification QR codes", "pip install segno"),
}

_cache: dict[str, Any] = {}
_failures: dict[str, Exception] = {}


def load(module: str, *, feature: str | None = None) -> Any:
    """Imports `module`, raising `MissingOptionalDependencyError` if absent.

    `OSError` is caught alongside `ImportError` deliberately: a native-library
    dependency that is pip-installed but whose shared objects are missing
    (WeasyPrint without GTK, Pillow without its codecs) fails that way, and
    from the caller's point of view it is the same "this format isn't
    available here" condition.
    """
    if module in _cache:
        return _cache[module]
    if module in _failures:
        raise _build_error(module, feature) from _failures[module]
    try:
        imported = importlib.import_module(module)
    except (ImportError, OSError) as exc:
        _failures[module] = exc
        log.warning("optional_dependency_unavailable", module=module, error=str(exc))
        raise _build_error(module, feature) from exc
    _cache[module] = imported
    return imported


def load_attr(module: str, attribute: str, *, feature: str | None = None) -> Any:
    """`load()` plus one attribute access -- `load_attr("pypdf", "PdfReader")`."""
    return getattr(load(module, feature=feature), attribute)


def is_available(module: str) -> bool:
    """Whether `module` is INSTALLED, without importing it.

    Deliberately `find_spec` rather than a real import. A capability probe
    must be side-effect-free, and some of these packages are not: importing
    WeasyPrint loads the GTK/Pango/Cairo native stack, which on a host with a
    partial GTK install aborts the interpreter outright rather than raising --
    confirmed on this machine, where a probe that imported it took the whole
    process down. A health check must never be able to do that.

    The trade-off is that "installed but its native libraries are broken"
    reads as available here. That case still fails loudly and correctly at
    `load()` time, on the one request that needs it, with the install hint
    attached -- which is where it belongs.
    """
    if module in _cache:
        return True
    if module in _failures:
        return False
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        # A namespace-package edge case or a parent package that itself fails
        # to import -- either way the module is not usable here.
        return False


def missing_optional_dependencies() -> dict[str, str]:
    """`{module: "<feature> -- <install hint>"}` for everything unavailable,
    for the health endpoint to report. Probing is cheap after the first call
    (both hits and misses are cached)."""
    missing: dict[str, str] = {}
    for module, (feature, hint) in _OPTIONAL.items():
        if not is_available(module):
            missing[module] = f"{feature} -- {hint}"
    return missing


def _build_error(module: str, feature: str | None) -> MissingOptionalDependencyError:
    known_feature, hint = _OPTIONAL.get(module, (feature or module, f"pip install {module}"))
    return MissingOptionalDependencyError(feature or known_feature, module, hint)
