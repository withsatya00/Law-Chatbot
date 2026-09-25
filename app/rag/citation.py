import re
from typing import Any

from app.core.constants import INSUFFICIENT_CONTEXT_MESSAGE
from app.schemas.common import RetrievedChunk, SourceCitation


def _looks_like_bns_family(act_name: str) -> bool:
    """Recognize the code family for jurisdiction routing, not Act identity."""
    collapsed = re.sub(r"\s+", "", act_name).lower()
    return any(
        token in collapsed
        for token in ("nyayasanhita", "nagariksuraksha", "sakshyaadhiniyam")
    )

# ---------------------------------------------------------------------------
# Act-name sanity, added by post-Phase-3 hardening (Phase 2, milestone B).
#
# `act_name` is extracted from document text during ingestion, and on this
# corpus the extraction sometimes picks up a sentence fragment or a truncated
# generic. Live, on a real cheque-bounce question, the answer's cited
# provisions came back as:
#
#     Negotiable Instruments Act - Section 138        (correct)
#     Notwithstanding anything contained in the Code - Section 138
#     THE CODE - Section 138
#     Repealing and Amending Act - Section 138.
#
# The middle two are not citations. They are a subordinate clause and a
# truncated heading, rendered in the citation slot, in the `applicable_law`
# field, and in the reader-facing source list -- indistinguishable, in that
# position, from the real one above them.
#
# So a name that cannot be an Act's name is DROPPED rather than shown. The
# citation keeps its section and its source document, which is the honest
# statement: this is where the passage came from, and the Act was not
# identified. Nothing is guessed at or substituted.
#
# Deliberately conservative -- rejecting a real Act name would remove a correct
# citation, which is the worse error. A name is rejected only when it opens
# with a word that cannot begin an Act's title, or when it carries no
# statute-designating noun at all, or when it is one of the bare generics.
# ---------------------------------------------------------------------------

# Words that begin a subordinate clause, never a statute's title.
_CLAUSE_OPENERS = re.compile(
    r"^\s*(notwithstanding|provided|subject\s+to|save\s+as|except|unless|whereas|where\s+any|"
    r"in\s+the\s+case|for\s+the\s+purposes?|nothing\s+in|any\s+person|no\s+court)\b",
    re.IGNORECASE,
)
# A statute's title says what kind of instrument it is.
_STATUTE_NOUNS = re.compile(
    r"\b(act|code|sanhita|adhiniyam|samhita|constitution|rules?|regulations?|ordinance|"
    r"order|scheme|bye-?laws?|notification|amendment)\b",
    re.IGNORECASE,
)
# Names that contain a statute noun and nothing that identifies WHICH statute.
_BARE_GENERICS = frozenset(
    {
        "act", "the act", "this act", "the said act", "said act",
        "code", "the code", "this code", "the said code",
        "rules", "the rules", "regulations", "the regulations",
        "constitution", "the constitution", "ordinance", "the ordinance",
        "amendment act", "the amendment act", "principal act", "the principal act",
    }
)
# Long enough that a fragment usually fails; short enough for the shortest real
# titles ("RTI Act").
_MIN_ACT_NAME_CHARACTERS = 6
_MAX_ACT_NAME_CHARACTERS = 160

# Cosmetic-only OCR/extraction garbling confirmed live in this corpus's own
# `act_name` metadata: "BHARATIYA" split into "BHARA TIY A" (405+ chunks
# under this one spelling alone) and, for BNS specifically, "NYAYA" further
# split into "NY A Y A". Fixing ONLY the spelling/spacing here, never the
# Act identity the name claims. Correcting a misidentified Act requires
# source-specific evidence; section numbers alone cannot establish it.
_GARBLED_SPELLING_FIXES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bBHARA\s*TIY\s*A\b", re.IGNORECASE), "BHARATIYA"),
    (re.compile(r"\bNY\s*A\s*Y\s*A\b", re.IGNORECASE), "NYAYA"),
)


def _fix_garbled_spelling(name: str) -> str:
    for pattern, replacement in _GARBLED_SPELLING_FIXES:
        # Uppercase replacement, then re-title-cased when the surrounding
        # name isn't itself shouting in all-caps -- avoids swapping a
        # garbled ALL-CAPS fragment for a jarring mixed-case one sitting
        # next to otherwise-uppercase text ("THE BHARA TIY A NAGARIK..."
        # must become "THE BHARATIYA NAGARIK...", not "THE Bharatiya
        # NAGARIK...").
        name = pattern.sub(
            replacement if name.isupper() else replacement.title(), name
        )
    return name


def usable_act_name(raw: object) -> str | None:
    """`raw` if it can be an Act's name, else `None`.

    "The Constitution of India" and "Bharatiya Nyaya Sanhita, 2023" pass.
    "Notwithstanding anything contained in the Code" and "THE CODE" do not.
    """
    if not isinstance(raw, str):
        return None
    name = _fix_garbled_spelling(" ".join(raw.split()))
    if not name:
        return None
    if not (_MIN_ACT_NAME_CHARACTERS <= len(name) <= _MAX_ACT_NAME_CHARACTERS):
        return None
    if name.casefold().strip(" .,") in _BARE_GENERICS:
        return None
    if _CLAUSE_OPENERS.search(name):
        return None
    if not _STATUTE_NOUNS.search(name):
        return None
    return name


# Live-verified failure (QA session 2026-09-24): a generated answer named
# "Bharatiya Nyaya Sanhita ... Section 303(2)" as the punishment provision
# for theft, with `no_verified_context=False` (i.e. presented as verified),
# while the actual retrieved `citations` were Bombay Habitual Offenders Act
# s.20 and Bharatiya Nagarik Suraksha Sanhita (the procedural code, not the
# one named) ss.229/395 -- none of them BNS, none of them section 303.
# `validate_grounding` below only checked that *some* non-empty citations
# existed, never that the specific section the prose leans on is among them,
# so a real Act name attached to a specific-sounding section number that the
# retrieval step never actually found sailed through as "grounded". This
# extracts every "Section N"/"धारा N"/"U/S N" the answer names and requires
# at least one to match a section actually present in `citations` --
# deliberately permissive (any overlap passes) so an answer that legitimately
# discusses a few related/background sections alongside its main citation
# is not penalised for mentioning one that wasn't independently retrieved.
_SECTION_MENTION_RE = re.compile(r"(?:Section|Sec\.?|धारा|U/S)\s*[:\-]?\s*(\d+[A-Za-z]?)", re.IGNORECASE)


def _mentioned_sections(answer: str) -> set[str]:
    return {match.group(1) for match in _SECTION_MENTION_RE.finditer(answer)}


_ACT_REFERENCE_STOPWORDS = {"the", "act", "of", "and", "code", "sanhita"}


def _act_referenced_in_answer(answer: str, act_name: str) -> bool:
    """Whether `answer`'s own text names this Act -- mirrors `app.services.
    safe_decline._act_key_terms`'s exact substring-match reasoning (kept as
    a separate small copy here rather than a shared import, since the two
    modules don't otherwise depend on each other and this is a two-line
    function): every significant word of the Act's name must appear in the
    answer, so a short/generic name ("the Act") can't spuriously match by
    matching nothing meaningful at all.
    """
    terms = [word for word in re.findall(r"[a-z]{4,}", act_name.lower()) if word not in _ACT_REFERENCE_STOPWORDS]
    if not terms:
        return False
    answer_lower = answer.lower()
    return all(term in answer_lower for term in terms)


def _clean_provision(raw: object) -> str | None:
    """A section/article number without the trailing punctuation the heading it
    was lifted from happened to carry -- "138." and "138" are one provision,
    and printing both makes them look like two."""
    if raw is None:
        return None
    cleaned = str(raw).strip().rstrip(".,;:")
    return cleaned or None


def _page_evidence(metadata: dict[str, Any]) -> dict[str, Any]:
    """Page fields from chunk metadata, omitting anything absent.

    Returns only keys that are actually present, so a chunk indexed before
    Phase 2 (none of the 2079 existing ones carry page data) produces a
    citation with page evidence left at `None` rather than a fabricated page 1.
    """
    evidence: dict[str, Any] = {}
    for source_key, target_key in (
        ("page_number", "page_number"),
        ("page_start", "page_start"),
        ("page_end", "page_end"),
        ("extraction_method", "extraction_method"),
    ):
        value = metadata.get(source_key)
        if value is not None:
            evidence[target_key] = value
    return evidence


def citation_from_metadata(metadata: dict[str, Any], *, source_document: str) -> SourceCitation:
    """One `SourceCitation` from one chunk's metadata.

    The single place chunk metadata becomes a citation. Post-Phase-3 hardening
    (Phase 2, milestone B) extracted it because there were two: this mapping,
    and a second, narrower one inlined in `ChatService._citation_from_chunk`
    that built the citations actually returned to chat users. That copy carried
    the act, section, article, chapter, URL and page evidence but silently
    dropped every governance field -- `verification_status`,
    `amendment_status`, `source_version`, `effective_date` and
    `last_verified_date`.

    Three things followed from that, all of them visible to a reader:

    * `SourceCitation.verification_status` was `None` on every chat citation,
      so a UI could not distinguish a reviewed source from an unreviewed one,
      and a genuinely verified source was still announced as unverified.
    * `SourceCitation`'s own validator appends "· REPEALED"/"· SUPERSEDED" to
      the citation label from `amendment_status`. With the field dropped, a
      repealed provision was cited with a label that said nothing about it.
    * `_confidence_fields` counts `verification_status == "verified"` sources
      to score grounding, and that count was structurally always zero.
    """
    section = _clean_provision(metadata.get("section_number"))
    # Section numbers are local to each Act. A query-expansion table cannot
    # establish source identity or repair mislabelled ingestion metadata.
    act_name = usable_act_name(metadata.get("act_name"))
    return SourceCitation(
        act_name=act_name,
        section=section,
        article=_clean_provision(metadata.get("article_number")),
        chapter=metadata.get("chapter"),
        source_document=source_document,
        government_source=metadata.get("government_source"),
        url=metadata.get("url"),
        source_version=metadata.get("source_version"),
        effective_date=str(metadata.get("effective_date")) if metadata.get("effective_date") else None,
        amendment_status=metadata.get("amendment_status"),
        last_verified_date=str(metadata.get("last_verified_date")) if metadata.get("last_verified_date") else None,
        verification_status=metadata.get("verification_status"),
        current_as_of=str(metadata.get("last_verified_date")) if metadata.get("last_verified_date") else None,
        **_page_evidence(metadata),
    )


class LegalCitationEngine:
    def citations_from_chunks(self, chunks: list[RetrievedChunk]) -> list[SourceCitation]:
        citations: list[SourceCitation] = []
        seen: set[tuple[str, str | None, str | None]] = set()
        for chunk in chunks:
            metadata = chunk.metadata
            source = metadata.get("source_document") or metadata.get("source")
            if not source:
                continue
            key = (str(source), metadata.get("section_number"), metadata.get("article_number"))
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation_from_metadata(metadata, source_document=str(source)))
        return citations

    def validate_grounding(self, answer: str, citations: list[SourceCitation], chunks: list[RetrievedChunk]) -> tuple[bool, str | None]:
        if INSUFFICIENT_CONTEXT_MESSAGE in answer:
            return True, None
        if not chunks:
            return False, "No retrieved chunks are available."
        if not citations:
            return False, "No source citations are available."
        cited_sources = {citation.source_document for citation in citations}
        if not cited_sources:
            return False, "Citations do not include source documents."
        mentioned_sections = _mentioned_sections(answer)
        # QA pass 2026-09-24 (T026/T027/T028 multi-turn grounding): pooling
        # EVERY cited chunk's section number together, regardless of which
        # Act each one belongs to, treats a genuinely unrelated retrieved
        # Act's real section number as if it were valid counter-evidence
        # against an answer that never discusses that Act at all. Confirmed
        # live via direct pipeline replication: a correct, well-grounded
        # answer citing "Section 18" of the Bombay Rents, Hotel and Lodging
        # House Rates Control Act, 1947 (a real provision -- confirmed
        # directly in that chunk's own raw retrieved text, penalizing
        # illegal premium/deposit) was rejected as ungrounded purely because
        # TWO OTHER, unrelated retrieved Acts in the same candidate pool
        # (an Odisha Stamp Act, a Transfer of Property Act chunk) happened
        # to carry real `section_number` metadata (26, 44) while the Bombay
        # Act chunk that actually supports "Section 18" had none tagged (a
        # metadata-extraction gap, not a citation error) -- {"18"} vs
        # {"26", "44"} read as "cites a section that matches nothing",
        # when the correct comparison is against the Act the answer is
        # ACTUALLY citing. Scoped to citations whose Act is named in the
        # answer text (same `_act_key_terms`-style substring match already
        # used by `safe_decline.prune_unreferenced_sources` for the same
        # "only count what the answer actually references" reasoning);
        # falls back to the full, unfiltered pool when the answer names no
        # Act clearly enough to narrow it, preserving the original guardrail
        # exactly for that case.
        relevant_citations = [
            citation for citation in citations
            if citation.act_name and _act_referenced_in_answer(answer, citation.act_name)
        ] or citations
        cited_sections = {citation.section for citation in relevant_citations if citation.section}
        if mentioned_sections and cited_sections and mentioned_sections.isdisjoint(cited_sections):
            return False, (
                "The answer cites a section number that does not match any retrieved/verified source."
            )
        return True, None
