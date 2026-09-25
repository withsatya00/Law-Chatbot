import re
from typing import Any

# A real section heading in Indian statute drafting is "<number>. <Title>.<dash><body>"
# on its own line, e.g. "173. Information in cognizable cases.—(1) Every information...".
# Anchoring to line-start and requiring the em/en-dash body separator within a short
# lookahead excludes two common false positives that a bare "Section N" search matches:
# footnote/amendment annotations ("1. Subs. by Act 10 of 2009, s. 38, for section 77
# (w.e.f. 27-10-2009).") and mid-sentence cross-references to a *different* act's section
# ("...fails to record any information given to him under sub-section (1) of section 173
# of the Bharatiya Nagarik Suraksha Sanhita...", found inside a Bharatiya Nyaya Sanhita
# chunk -- neither has a heading-style dash and both previously got misread as the
# chunk's own section number).
SECTION_HEADING_RE = re.compile(r"(?m)^[ \t]*(\d{1,4}[A-Z]{0,2})\.\s*[A-Z][^\n]{0,140}?[—–]")
# Some official sources (confirmed live: BNSS_2023_Official_Gazette.pdf's own
# text for section 173, the FIR-registration provision) print no title/dash
# at all before the operative text -- the line reads straight
# "173. (1) Every information relating to..." with the marginal title
# omitted from the extracted text entirely. SECTION_HEADING_RE structurally
# cannot match that (there is no dash within the lookahead), so every chunk
# of that document fell through to SECTION_LOOSE_RE below, which has no
# defence against cross-references -- confirmed live, 11 chunks whose real
# text was sections 173/181/223/397/etc. all got stamped "64" because a
# cross-reference to a DIFFERENT act's section 64 happened to be the first
# bare "section N" mention in the packed buffer. A section that opens
# directly on subsection (1) is exactly as structurally anchored as the
# title+dash form -- line-start, followed immediately by the literal "(1)"
# rather than by explanatory prose -- so it is trusted at the same
# "heading" provenance, not folded into the last-resort fallback.
SECTION_HEADING_NO_TITLE_RE = re.compile(r"(?m)^[ \t]*(\d{1,4}[A-Z]{0,2})\.\s*\(1\)")
# Handbook/commentary style sources cite sections as bracketed asides, e.g.
# "Changes with respect to FIR [Sec. 173 BNSS]" -- no heading line exists to anchor on,
# so this is the fallback for that document shape.
SECTION_BRACKET_RE = re.compile(r"\[\s*Sec\.?\s*(\d{1,4}[A-Z]{0,2})", re.IGNORECASE)
# Last-resort fallback: kept only for text that matches neither pattern above so
# metadata degrades to "less precise" rather than "absent". Previously
# `re.IGNORECASE` applied to the WHOLE pattern, which silently widened the captured
# group's `[0-9A-Z]` character class to match lowercase letters too -- so a mid-
# sentence phrase like "...under this Section shall be construed..." captured the
# literal word "shall" as a "section number" (same for "and"/"do"/"does"/"or").
# The trailing `[0-9A-Z()./-]*` was equally permissive, sweeping trailing
# punctuation into the stored value ("Section 147." -> "147.", "Section 356(2)"
# -> "356(2)"). Confirmed live: these exact junk values ("shall", "and", "147.",
# "356(2)", "141)", ...) were present in embeddings_metadata.section_number across
# BNS/BNSS/BSA. Fixed by (a) making only the "Section"/"Sec." keyword
# case-insensitive via an explicit [Ss] alternation instead of a pattern-wide flag,
# and (b) constraining the captured group to a real section identifier shape
# (1-4 digits, optional 1-2 letter suffix -- the same grammar SECTION_HEADING_RE
# and SECTION_BRACKET_RE already use), so "356(2)" still matches and correctly
# yields "356" (the subsection "(2)" stays in the body text, never in this field),
# while non-numeric words never match at all.
SECTION_LOOSE_RE = re.compile(r"\b(?:[Ss]ection|[Ss]ec\.?)\s+(\d{1,4}[A-Z]{0,2})\b")
CONSTITUTION_DOCUMENT_RE = re.compile(
    r"\bTHE\s+CONSTITUTION\s+OF\s+INDIA\b|भारत\s+का\s+संविधान",
    re.IGNORECASE,
)

# Which of the three patterns above actually produced `section_number` matters:
# a match from the section's own heading line is authoritative, a bracketed
# handbook-style citation ("[Sec. 173 BNSS]") is a cross-reference TO a
# section rather than that section's own text, and the loose last-resort
# pattern can match a bare mid-sentence mention ("...as provided under
# section 302...") inside a chunk that isn't about section 302 at all. Stored
# alongside `section_number` so retrieval/reranking can prefer a heading
# match over a cross-reference over a fallback match when several chunks
# claim the same section number (see `LegalReranker`'s
# `_SECTION_PROVENANCE_BONUS`). A chunk indexed before this field existed
# simply has no `section_number_provenance` key -- treated as lowest
# priority, same as `"fallback"`, everywhere this is read.
SECTION_NUMBER_PROVENANCE_HEADING = "heading"
SECTION_NUMBER_PROVENANCE_CROSS_REFERENCE = "cross_reference"
SECTION_NUMBER_PROVENANCE_FALLBACK = "fallback"
_SECTION_NUMBER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (SECTION_HEADING_RE, SECTION_NUMBER_PROVENANCE_HEADING),
    (SECTION_HEADING_NO_TITLE_RE, SECTION_NUMBER_PROVENANCE_HEADING),
    (SECTION_BRACKET_RE, SECTION_NUMBER_PROVENANCE_CROSS_REFERENCE),
    (SECTION_LOOSE_RE, SECTION_NUMBER_PROVENANCE_FALLBACK),
)

# Document-shape signatures used to classify *source_type*, so retrieval can rank a
# statute's own text above commentary/case-law discussing the same provision. Checked
# against the document's own first ~5000 chars (the same window used for act_name),
# since the shape of a source -- judgment vs. FAQ vs. bare act -- is a document-wide
# property, not something that varies chunk to chunk.
# J\s+U\s+D\s+G... (one-or-more whitespace required between each letter) matches only
# the letter-spaced "J U D G M E N T" heading Indian judgments use on their own line --
# NOT the ordinary word "judgment", which turns up harmlessly in plain statute prose
# (e.g. a BNS table-of-contents entry like "judgment stating that it is doubtful of
# which offence..." or a BSA section title "Judgments of Courts when relevant"). Using
# \s* here previously matched the bare word and misclassified those two acts as case law.
CASE_LAW_RE = re.compile(
    r"IN THE (?:SUPREME COURT OF INDIA|HIGH COURT OF)|J\s+U\s+D\s+G\s+M\s+E\s+N\s+T|"
    r"CRIMINAL APPEAL NO|WRIT PETITION|SLP\s*\(|\bPetitioner\(s\)|\bRespondent\(s\)",
    re.IGNORECASE,
)
FAQ_RE = re.compile(r"\bFAQs?\b|Frequently Asked Questions", re.IGNORECASE)
# India Code PDFs' text layer sometimes has stray spaces inside words from font
# kerning ("A CT N O . 43 OF  1954" for "ACT NO. 43 OF 1954", "T he" for "The") --
# \s* between each letter of ACT/NO absorbs that without risking false positives,
# since the full "(ACT NO <digits> OF <year>)" sequence is specific enough on its own.
STATUTE_RE = re.compile(r"\(\s*A\s*C\s*T\s*N\s*O\s*\.?\s*\d+\s*OF\s*\d{4}\s*\)|BE it enacted by Parliament", re.IGNORECASE)
# A statute's own "definitions" clause -- e.g. '"cognizable offence" means...' -- is
# the single highest-priority source for a definition query, distinct from the rest
# of the same act's operative/procedural text. Checked per chunk, not per document,
# since only a small part of any given act is its Section 2 definitions clause.
DEFINITION_CLAUSE_RE = re.compile(r'"[^"\n]{2,60}"\s+(?:means|includes)\b')

SOURCE_TYPE_STATUTE = "statute"
SOURCE_TYPE_DEFINITION = "definition"
SOURCE_TYPE_CASE_LAW = "case_law"
SOURCE_TYPE_FAQ = "faq"
SOURCE_TYPE_COMMENTARY = "commentary"
# Phase 7 "TOC Contamination": every India-Code-style Act PDF prints a
# chapter/part index ("ARRANGEMENT OF SECTIONS" -> "SECTIONS" -> a run of
# bare "<N>. <Title>." lines) before the operative text actually starts.
# `SectionAwareChunker._split_on_legal_boundaries` correctly refuses to split
# on these bare entries (no heading-pattern dash), by design -- but that
# means the whole index block still ends up AS a chunk (via `_split_large`'s
# plain character-window slicing once it's too long for one chunk), with no
# distinguishing metadata of its own. Confirmed live: one such chunk's own
# cross-reference text ("...document described in section 337 or section
# 338...") still satisfies `SECTION_LOOSE_RE` below, stamping the ENTIRE
# multi-chapter index block with a single arbitrary `section_number` that
# then wins exact-match SECTION_LOOKUP acceptance over the real section's own
# heading-anchored chunk. A corpus-wide scan (looking for the literal
# "SECTIONS" header line every genuine index carries, not a generic
# many-numbered-lines heuristic -- that also matches legitimate
# footnote-heavy operative chunks) found 51 such chunks across 9 documents,
# 19 of them carrying a spurious `section_number`.
SOURCE_TYPE_INDEX = "index"


# `[A-Za-z ]+` (space, not \s) can't cross a newline -- PDF cover/TOC pages
# routinely wrap a multi-word act title one or two words per line ("The
# Bharatiya Nagarik\nSuraksha Sanhita"), so the un-collapsed search silently
# matches only the trailing line-fragment that happens to end in a suffix
# keyword ("Suraksha Sanhita"), dropping the rest -- confirmed baked into
# every chunk of Delhi_Police_Academy_FAQs_New_Criminal_Laws.pdf via
# `setdefault`. Matched against a whitespace-collapsed copy instead (`text`
# itself is left untouched -- SECTION_HEADING_RE depends on real line breaks).
# Requiring every word to be Title-Case and capping repetitions non-greedily
# (stop at the FIRST suffix keyword) keeps this from instead over-matching
# into surrounding prose once newlines no longer block it.
# Three compounding bugs beyond the generic-phrase one below, all confirmed
# live against real newly-ingested Acts: (1) the suffix keywords were
# matched case-SENSITIVELY ("Act", never "ACT"), so an all-caps gazette
# cover/title line -- exactly where a document's real name usually appears
# -- never matched the pattern at all; scoped inline `(?i:...)` fixes just
# the suffix without loosening the "must look like a capitalized phrase"
# check the rest of the pattern relies on to avoid matching ordinary prose.
# (2) requiring EVERY word (including connectors) to start with a capital
# letter meant an ordinary lowercase connector inside a real title ("Right
# TO Information Act" written in prose as "Right to Information Act") broke
# the run, truncating the capture to whatever came after the last connector
# ("Information Act") -- a small, closed set of the connectors Indian Act
# titles actually use is now allowed inside the repeated group without
# needing to start capitalized. (3) a per-word pattern of `[A-Z][A-Za-z]*`
# has no allowance for an internal hyphen, so a hyphenated official title
# ("Income-tax Act, 2025" -- confirmed the Act's OWN short-title clause
# spells it exactly this way, not "Income Tax") broke mid-word at the
# hyphen the same way a newline used to (see the un-collapsed-search comment
# above): matching from "Income" died needing `\s+` right where a literal
# "-" sat instead, so the whole attempt silently restarted past the hyphen
# and captured only the fragment after it ("TAX ACT"). `(?:-[A-Za-z]+)*`
# lets a word absorb any number of hyphenated segments as part of itself.
_ACT_NAME_CONNECTOR_WORDS = r"(?:to|of|and|for|the|in|on|by|from|at|with)"
_ACT_NAME_WORD = r"[A-Z][A-Za-z]*(?:-[A-Za-z]+)*"
ACT_NAME_RE = re.compile(
    rf"\b({_ACT_NAME_WORD}"
    rf"(?:\s+(?:{_ACT_NAME_WORD}|{_ACT_NAME_CONNECTOR_WORDS})){{0,6}}?"
    r"\s+(?:(?i:Act|Sanhita|Adhiniyam|Code)))\b"
)

# "Code of Civil/Criminal Procedure" (CPC/CrPC) structurally can never match
# ACT_NAME_RE above: that pattern requires the anchor keyword (Act/Sanhita/
# Adhiniyam/Code) to be the LAST word of the title, but these two Codes are
# named with "Code" FIRST -- "Code of Civil Procedure", not "Civil Procedure
# Code". Confirmed live: every chunk of CPC_1908_Official.pdf got `act_name`
# stamped "THE CODE" -- a bare-act self-reference ("...hereinafter referred
# to as the Code") that happened to be the first (and only) phrase in the
# document's own text that ends in the word "Code", since the real title
# never had a chance to match at all. Checked before the general pattern
# below since it's a precise, known-good shape rather than a heuristic guess.
_CODE_OF_PROCEDURE_RE = re.compile(r"\bCode\s+of\s+(Civil|Criminal)\s+Procedure(?:,?\s*(\d{4}))?", re.IGNORECASE)

# Indian bare Acts print every amended provision with a footnote citing
# WHICH act inserted/substituted/omitted/repealed/renumbered it -- "1. Added
# by Act 2 of 1885, s. 4.", "Rep. by the Repealing and Amending Act, 1891
# (12 of 1891), s. 2 and Schedule I." -- and these footnotes are everywhere
# (one per amended/repealed section) in exactly the kind of operative-text
# chunk this function scans. `ACT_NAME_RE` cannot tell "the Act this
# footnote is CITING" from "the Act this document IS", so without a guard a
# chunk whose only "...Act"-shaped phrase is such a footnote gets `act_name`
# stamped with the wrong, unrelated amending Act's name -- confirmed live
# against real chunks of NEGOTIABLE_INSTRUMENTS_ACT_Official.pdf, in two
# different shapes:
#
# * The footnote's own verb IS the match's first captured word -- "1. Added
#   by Act 2 of 1885, s. 4." matches as "Added by Act" (Act/Sanhita/
#   Adhiniyam/Code is only required as the LAST word; nothing stops the
#   match starting mid-footnote). Caught by `_is_amendment_footnote_leader`
#   below, which checks the candidate's own first word.
# * The footnote's verb precedes the match, which starts at a REAL (but
#   unrelated) Act's name -- "Rep. by the Repealing and Amending Act, 1891"
#   matches as "Repealing and Amending Act" (itself a genuine, real Act --
#   `_is_generic_act_name` correctly does NOT reject it), with "Rep. by
#   the " sitting just before it. Caught by `_AMENDMENT_FOOTNOTE_CUE_RE`
#   against the ~40 characters immediately before the candidate match --
#   comfortably wider than "Ins. by the "/"Subs. by the " but short enough
#   to never reach back into the previous sentence's own text -- with an
#   optional trailing article since the cue and the cited Act's name are
#   almost never adjacent ("by the ___", "by an ___").
_AMENDMENT_FOOTNOTE_VERBS = r"ins(?:erted)?|subs(?:tituted)?|add(?:ed)?|om(?:itted)?|rep(?:ealed)?|amend(?:ed)?|re-?numbered"
_AMENDMENT_FOOTNOTE_LEADER_WORDS = frozenset({
    "added", "ins", "inserted", "subs", "substituted", "omitted", "om",
    "repealed", "rep", "amended", "renumbered",
})
_AMENDMENT_FOOTNOTE_CUE_RE = re.compile(
    rf"\b(?:{_AMENDMENT_FOOTNOTE_VERBS})\.?\s+by(?:\s+(?:the|an?))?\s*$", re.IGNORECASE,
)
_AMENDMENT_FOOTNOTE_LOOKBACK_CHARS = 40


def _is_amendment_footnote_leader(name: str) -> bool:
    first_word = name.split()[0].rstrip(".").lower()
    return first_word in _AMENDMENT_FOOTNOTE_LEADER_WORDS

# Non-greedy `{0,6}?` means `ACT_NAME_RE` always prefers the SHORTEST valid
# match -- and standard Indian legal drafting opens almost every Act's own
# preamble with the literal two-word phrase "An Act to [purpose]...", while
# the operative text is full of two-word backward-references like "under
# this Act"/"violates that Act". Both satisfy the pattern trivially (one
# capitalized filler word + the suffix keyword) and are shorter than any
# real title ("The Right to Information Act" = 5 words), so a plain
# `.search()` latches onto whichever generic phrase happens to appear
# first in the chunk instead of the document's actual name. Confirmed
# live: EVERY chunk of three separate newly-ingested Acts (RTI, CGST, IT
# Act) got `act_name` stamped "An Act" this way -- not a one-document
# fluke, a systemic mismatch between this regex's non-greedy preference and
# how Indian statutes are actually worded. `_is_generic_act_name` rejects
# exactly this shape (a single filler word, not a real multi-word title) so
# `extract()` below can keep searching the same chunk for a match that
# isn't one, rather than accepting the first (usually wrong) hit.
_GENERIC_ACT_NAME_LEADERS = {
    "an", "this", "that", "any", "such", "every", "no", "said", "which", "whose", "same", "aforesaid",
}


def _is_generic_act_name(name: str) -> bool:
    words = name[4:].split() if name.lower().startswith("the ") else name.split()
    core_words = words[:-1]  # drop the trailing Act/Sanhita/Adhiniyam/Code keyword itself
    if not core_words:
        # "The Act"/"The Code"/"The Sanhita"/"The Adhiniyam" alone -- exactly
        # as generic as "An Act" below, just reached by stripping a leading
        # "the " first, which left nothing between it and the suffix keyword
        # for the length-1 check to ever see. Confirmed live: this exact gap
        # is why a stray "...hereinafter referred to as the Code" self-
        # reference could pass as a real title for a document whose actual
        # name never matched (see `_CODE_OF_PROCEDURE_RE`'s docstring).
        return True
    return len(core_words) == 1 and core_words[0].lower() in _GENERIC_ACT_NAME_LEADERS

# The literal index-page fingerprint (see SOURCE_TYPE_INDEX above). PDF text
# extraction sometimes kerns the header apart ("SEC TIONS"), so the pattern
# tolerates optional whitespace mid-word rather than requiring it whole.
_TOC_HEADER_RE = re.compile(r"(?m)^\s*(?:ARRANGEMENT\s+OF\s+)?SEC\s*TIONS\s*$", re.IGNORECASE)
# A bare index entry: "<number>. <Title ending in a period>" alone on its own
# line, with no heading-pattern dash. Reuses the same number/title shape as
# `SECTION_HEADING_RE` minus the dash requirement, since that's exactly the
# one structural difference between a TOC entry and a real heading.
_TOC_LIST_LINE_RE = re.compile(r"(?m)^\d{1,4}[A-Z]{0,2}\.\s+[A-Z][^\n]{0,140}\.\s*$")
# Requires the header PLUS several list lines, not the header alone -- a
# chunk that merely trails off the tail end of a chapter's operative text
# into the next chapter's "SECTIONS" header (with few or no list lines
# actually captured in this chunk) shouldn't be swept in.
_TOC_MIN_LIST_LINES = 3


def is_toc_chunk(text: str) -> bool:
    if not _TOC_HEADER_RE.search(text):
        return False
    return len(_TOC_LIST_LINE_RE.findall(text)) >= _TOC_MIN_LIST_LINES


class MetadataExtractor:
    async def extract(self, text: str, base_metadata: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(base_metadata)
        if CONSTITUTION_DOCUMENT_RE.search(text):
            metadata["instrument_type"] = "constitution"
        toc = is_toc_chunk(text)
        collapsed = re.sub(r"\s+", " ", text)
        code_of_procedure_match = _CODE_OF_PROCEDURE_RE.search(collapsed)
        if code_of_procedure_match:
            kind, year = code_of_procedure_match.group(1), code_of_procedure_match.group(2)
            act_name: str | None = f"Code of {kind.title()} Procedure" + (f", {year}" if year else "")
        else:
            act_match = next(
                (
                    m for m in ACT_NAME_RE.finditer(collapsed)
                    if not _is_generic_act_name(m.group(1))
                    and not _is_amendment_footnote_leader(m.group(1))
                    and not _AMENDMENT_FOOTNOTE_CUE_RE.search(
                        collapsed[max(0, m.start() - _AMENDMENT_FOOTNOTE_LOOKBACK_CHARS):m.start()]
                    )
                ),
                None,
            )
            act_name = act_match.group(1).strip() if act_match else None
        # A chapter/part index chunk isn't "about" any single section -- it's
        # a listing OF many -- so it must never claim one via the same
        # heading/bracket/loose patterns real operative text uses (see
        # SOURCE_TYPE_INDEX above for the exact failure this prevents).
        section_result = None if toc else self._section_number(text)
        chapter_match = re.search(r"\bChapter\s+([IVXLCDM0-9A-Z -]+)", text, re.IGNORECASE)
        if act_name:
            metadata.setdefault("act_name", act_name)
        if section_result:
            section_number, provenance = section_result
            # The official Constitution uses the same bare "21. Title.—"
            # typography as Acts use for sections.  Once the document-level
            # signature has identified the instrument, record that heading as
            # an Article so an Article lookup cannot be mixed with Section 21
            # from unrelated Acts.
            provision_key = "article_number" if metadata.get("instrument_type") == "constitution" else "section_number"
            metadata.setdefault(provision_key, section_number)
            metadata.setdefault("section_number_provenance", provenance)
        if chapter_match:
            metadata.setdefault("chapter", chapter_match.group(1).strip())
        metadata.setdefault("source", metadata.get("source_document", "uploaded_document"))
        metadata.setdefault("legal_category", self._category(text))
        if toc:
            # Not a `setdefault` -- an index chunk's own shape always wins
            # here regardless of what the generic classifier below would
            # have guessed (it would say "statute", same as any real section,
            # since STATUTE_RE only looks at document-wide front matter).
            metadata["source_type"] = SOURCE_TYPE_INDEX
            metadata["is_toc"] = True
            return metadata
        metadata.setdefault("source_type", self._document_source_type(text))
        # Overrides the document-wide source_type only for the specific chunk that IS
        # the definitions clause -- everything else in a statute keeps "statute".
        if metadata.get("source_type") == SOURCE_TYPE_STATUTE and DEFINITION_CLAUSE_RE.search(text):
            metadata["source_type"] = SOURCE_TYPE_DEFINITION
        return metadata

    def _document_source_type(self, text: str) -> str:
        if CASE_LAW_RE.search(text):
            return SOURCE_TYPE_CASE_LAW
        if FAQ_RE.search(text[:1000]):
            return SOURCE_TYPE_FAQ
        if STATUTE_RE.search(text):
            return SOURCE_TYPE_STATUTE
        return SOURCE_TYPE_COMMENTARY

    def _section_number(self, text: str) -> tuple[str, str] | None:
        for pattern, provenance in _SECTION_NUMBER_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip(), provenance
        return None

    def _category(self, text: str) -> str:
        lowered = text.lower()
        if any(term in lowered for term in ["salary", "employment", "labour", "wages"]):
            return "Employment Law"
        if any(term in lowered for term in ["fir", "offence", "bail", "police"]):
            return "Criminal Law"
        if any(term in lowered for term in ["property", "deed", "tenant", "landlord"]):
            return "Property Law"
        if any(term in lowered for term in ["consumer", "refund", "defect"]):
            return "Consumer Law"
        return "General Law"
