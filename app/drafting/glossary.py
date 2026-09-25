import json
from functools import lru_cache
from pathlib import Path

from app.schemas.drafting import LegalTermEntry

_DATA_PATH = Path(__file__).parent / "data" / "legal_terms_hi.json"


@lru_cache(maxsize=1)
def _load_terms() -> list[LegalTermEntry]:
    raw = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return [LegalTermEntry(**entry) for entry in raw]


class LegalTermsLibrary:
    """Legal Hindi Knowledge Library: explains legal terms in simple language.

    Adding a term is a data-only change: append an entry to
    `app/drafting/data/legal_terms_hi.json` with the same shape.
    """

    async def all_terms(self) -> list[LegalTermEntry]:
        return list(_load_terms())

    async def lookup(self, query: str | None = None) -> list[LegalTermEntry]:
        terms = _load_terms()
        if not query or not query.strip():
            return list(terms)
        needle = query.strip().lower()
        return [
            entry
            for entry in terms
            if needle in entry.term.lower()
            or needle in entry.english_equivalent.lower()
            or needle in entry.meaning.lower()
        ]

    async def get(self, term: str) -> LegalTermEntry | None:
        needle = term.strip().lower()
        for entry in _load_terms():
            if entry.term.lower() == needle or entry.english_equivalent.lower() == needle:
                return entry
        return None
