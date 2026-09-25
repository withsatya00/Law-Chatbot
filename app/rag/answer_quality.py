from typing import Any

"""Phase 1 item 3: the gate every grounded legal answer passes before it is
returned.

Retrieval quality, prompt compliance and provider health are all checked
independently elsewhere, but nothing looked at the FINISHED answer and asked
"is this a legal answer at all?". The failures that reached users were exactly
the ones only a final inspection catches:

* a provider outage message rendered as if it were legal content ("Gemini API
  is unreachable...", followed by the legal disclaimer);
* an internal prompt label quoted at the reader ("Source 2 ke mutabiq...")
  and, in one case, a raw `{context}` placeholder;
* an answer written in a different language from the question;
* a fluent answer with no usable citation behind it -- indistinguishable, to a
  reader, from a properly grounded one.

The gate is deliberately conservative. Rejecting a good answer costs the user
a real answer, so every check below either matches something that could not
appear in a genuine reply (a provider error phrase, a prompt placeholder) or
requires a wide margin before it fires. Anything uncertain returns `PASS` with
a recorded note rather than downgrading.

Severity is separated from detection on purpose:

* `REJECT` -- the answer must not be shown. It is replaced by the same safe
  verified-context fallback used when retrieval finds nothing, at confidence
  0. Reserved for answers that are unsupported, artefactual, or not an answer.
* `RETRY_LANGUAGE` -- the answer may well be correct but is in the wrong
  language. Discarding correct legal information over a presentation problem
  helps nobody, so the caller regenerates once with a language reinforcement
  and, if that also misses, keeps the answer with its confidence capped and
  the reason recorded.

Known limitation on the streaming endpoint: tokens reach the client as they
are produced, so a rejected answer has already been displayed by the time this
runs. `answer_stream` still applies a REJECT -- it replaces the text recorded
as the settled answer in the "done" payload, conversation memory, chat history
and analytics -- but cannot un-send what was streamed. It deliberately does
NOT act on RETRY_LANGUAGE, because regenerating after the fact would swap out
text the user already watched appear. The non-streaming `/chat` endpoint has
neither limitation.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

from app.rag.query_rewriter import IPC_TO_BNS_CROSSWALK

# Scripts by Unicode block start, for the language-match check. Only the
# scripts this product actually replies in are listed; anything else counts as
# "other", which no check keys off.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("odia", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("perso_arabic", 0x0600, 0x06FF),
    ("ol_chiki", 0x1C50, 0x1C7F),
    ("meetei_mayek", 0xABC0, 0xABFF),
)

# The script a reply in each language is expected to be written in. Languages
# absent from this map (and "english"/"hinglish", handled separately) skip the
# language check entirely rather than guess.
_LANGUAGE_SCRIPT: dict[str, str] = {
    "hindi": "devanagari", "marathi": "devanagari", "nepali": "devanagari",
    "konkani": "devanagari", "maithili": "devanagari", "dogri": "devanagari",
    "bodo": "devanagari", "sanskrit": "devanagari",
    "bengali": "bengali", "assamese": "bengali",
    "punjabi": "gurmukhi", "gujarati": "gujarati", "odia": "odia",
    "tamil": "tamil", "telugu": "telugu", "kannada": "kannada", "malayalam": "malayalam",
    "urdu": "perso_arabic", "kashmiri": "perso_arabic", "sindhi": "perso_arabic",
    "santali": "ol_chiki", "manipuri": "meetei_mayek",
}
_LATIN_LANGUAGES = {"english", "hinglish"}

# Phrasings a provider yields INSTEAD of an answer when a call fails. Shared in
# spirit with `chat_service._LLM_ERROR_PHRASES` (the streaming-path check),
# extended with the shapes that only show up once an error has been formatted
# into prose or a traceback has leaked.
_PROVIDER_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(is unreachable|is currently unavailable|is not configured|was rejected)\b", re.IGNORECASE),
    re.compile(r"\b(rate limit or quota|failed unexpectedly|please start ollama)\b", re.IGNORECASE),
    re.compile(r"\b(traceback \(most recent call last\)|httpx\.|openai\.|anthropic\.|requests\.exceptions)", re.IGNORECASE),
    re.compile(r"\b(connection refused|connection reset|read timed out|ssl(?:error| certificate))\b", re.IGNORECASE),
    re.compile(r"\bhttp\s*(?:error\s*)?(?:4\d\d|5\d\d)\b|\bstatus[_ ]code[=: ]\s*(?:4\d\d|5\d\d)\b", re.IGNORECASE),
    re.compile(r"\b(api[_ ]?key|bearer token|authorization header)\b", re.IGNORECASE),
    re.compile(r"localhost:\d+|127\.0\.0\.1:\d+", re.IGNORECASE),
)

# Fragments of this system's own plumbing that must never reach a reader.
_INTERNAL_ARTEFACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # The prompt's internal source index. `chat_service._replace_source_ordinals`
    # rewrites these into real citations first; anything still here after that
    # means the rewrite could not resolve it.
    re.compile(r"\[?\s*(?:source|स्रोत|सोर्स)\s*[-#:]?\s*\d{1,2}\s*\]?", re.IGNORECASE),
    # An unrendered prompt placeholder.
    re.compile(r"\{(?:context|question|language|intent|entities|statutory_currency_note)\}"),
    re.compile(r"</?context>", re.IGNORECASE),
    # The model narrating its own instructions instead of answering.
    re.compile(r"\bas an ai (?:language )?model\b", re.IGNORECASE),
    re.compile(r"^\s*(?:system|assistant|user)\s*:", re.IGNORECASE | re.MULTILINE),
)

# Words too generic to prove an answer is on topic (a superset of the ordinary
# stopword idea, extended with the boilerplate every legal answer carries).
_GENERIC_TERMS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "as", "at", "by",
    "this", "that", "it", "you", "your", "with", "from", "can", "may", "will", "should", "under",
    "legal", "law", "act", "section", "information", "provided", "educational", "purposes", "only",
    "advice", "consult", "qualified", "advocate", "situation", "specific", "please", "considered",
})

# A NARROWER, additional exclusion used only by `_section_attribution_
# mismatch`, never by `_on_topic`: an Act's own name (and "person") IS
# legitimate topical-overlap evidence for "is this answer about the same
# corpus at all" (`_on_topic`'s job -- confirmed by an existing test this
# broke when tried as part of `_GENERIC_TERMS` above: an otherwise-clearly-
# grounded cheating answer shared only "cheating" plus "Bharatiya Nyaya
# Sanhita" with its one retrieved chunk, and needed the Act-name overlap to
# clear `_on_topic`'s 2-shared-term bar). But it is USELESS for "does THIS
# claim belong to THIS specific section" (`_section_attribution_mismatch`'s
# narrower job): the Act's own name appears in nearly every sentence that
# names any of its sections, and "person" is near-universal across legal
# text generally, so neither discriminates one section's topic from
# another's within the same Act -- confirmed live, this let a claim about
# FIR-refusal falsely appear "grounded" in an unrelated arrest-procedure
# section purely via the shared words "person"/"Sanhita".
_SECTION_ATTRIBUTION_GENERIC_TERMS = _GENERIC_TERMS | frozenset({
    "person", "bharatiya", "nagarik", "suraksha", "sanhita", "nyaya", "adhiniyam", "sakshya",
    "bns", "bnss", "bsa",
    # Confirmed live: widening the check to a 3-sentence window (see below)
    # to catch a citation and its explanation split across a sentence
    # boundary reintroduced a spurious match through "officer" -- present
    # in an EARLIER, unrelated sentence ("officer-in-charge... mana kar
    # dete hain") and, separately and coincidentally, in Section 64's own
    # genuinely retrieved (but unrelated) text ("officer... serving...").
    # "officer"/"police"/"station" are exactly as cross-cutting across
    # nearly every BNSS procedural provision as "person" already is above,
    # so none of them discriminate one section's specific claim from
    # another's either.
    "officer", "police", "station",
})


def _section_attribution_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if token not in _SECTION_ATTRIBUTION_GENERIC_TERMS
    }

# Short enough that a terse-but-real answer ("An FIR records a cognizable
# offence.") survives, long enough to catch the failures this is for: an
# empty completion, or a bare "OK."/"Yes." carrying the legal disclaimer.
_MIN_ANSWER_CHARACTERS = 25
# Below this fraction of the expected script's characters, a reply in a
# non-Latin language has effectively been written in another language.
# Deliberately low: a legitimate Hindi answer is full of English statute names
# ("Bharatiya Nagarik Suraksha Sanhita", "Section 173", "FIR"), so requiring a
# high proportion of Devanagari would reject correct answers.
_MIN_EXPECTED_SCRIPT_RATIO = 0.10
# Above this fraction of non-Latin characters, a reply that was supposed to be
# English/Hinglish is not.
_MAX_FOREIGN_SCRIPT_RATIO_FOR_LATIN = 0.50


class Severity(str, Enum):
    PASS = "pass"
    REJECT = "reject"
    RETRY_LANGUAGE = "retry_language"
    RETRY_SECTION_ATTRIBUTION = "retry_section_attribution"


@dataclass(frozen=True)
class QualityVerdict:
    severity: Severity
    failed_check: str | None = None
    reason: str | None = None
    # Checks that ran and passed, for the routing log -- makes "the gate saw
    # this and was happy" distinguishable from "the gate never ran".
    checks_run: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.severity is Severity.PASS


def _script_counts(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for char in text:
        if not unicodedata.category(char).startswith(("L", "M")):
            continue
        code = ord(char)
        if code < 0x0250:
            counts["latin"] = counts.get("latin", 0) + 1
            continue
        for name, start, end in _SCRIPT_RANGES:
            if start <= code <= end:
                counts[name] = counts.get(name, 0) + 1
                break
        else:
            counts["other"] = counts.get("other", 0) + 1
    return counts


def _significant_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if token not in _GENERIC_TERMS
    }


def evaluate(
    answer: str,
    *,
    question: str,
    language: str,
    sources: list[Any],
    ranked_chunks: list[Any],
    disclaimer: str = "",
) -> QualityVerdict:
    """Judges a finished, grounded answer. `sources` are `SourceCitation`s and
    `ranked_chunks` the `RetrievedChunk`s they came from.

    `disclaimer` is stripped before the substance checks: it is appended to
    every answer, so leaving it in would make an otherwise-empty reply look
    substantial and would contribute its own vocabulary to the topical check.
    """
    checks: list[str] = []
    body = (answer or "").replace(disclaimer, "").strip() if disclaimer else (answer or "").strip()

    checks.append("non_empty")
    if len(body) < _MIN_ANSWER_CHARACTERS:
        return QualityVerdict(
            Severity.REJECT, "non_empty",
            f"The generated answer was empty or too short to be a legal answer ({len(body)} characters).",
            tuple(checks),
        )

    checks.append("no_provider_error")
    for pattern in _PROVIDER_ERROR_PATTERNS:
        match = pattern.search(body)
        if match:
            return QualityVerdict(
                Severity.REJECT, "no_provider_error",
                # The matched text itself is deliberately NOT echoed: a leaked
                # error can carry a host name, an API key fragment or a file
                # path, and this reason string is logged and returned to the
                # client (Phase 1 item 7).
                "The generated answer contained provider or network error text rather than legal content.",
                tuple(checks),
            )

    checks.append("no_internal_artefacts")
    for pattern in _INTERNAL_ARTEFACT_PATTERNS:
        if pattern.search(body):
            return QualityVerdict(
                Severity.REJECT, "no_internal_artefacts",
                "The generated answer exposed internal prompt scaffolding instead of a citable source.",
                tuple(checks),
            )

    checks.append("grounded")
    if not ranked_chunks:
        return QualityVerdict(
            Severity.REJECT, "grounded",
            "No retrieved chunk supports this answer.",
            tuple(checks),
        )
    if not any(getattr(source, "is_identifiable", False) for source in sources):
        return QualityVerdict(
            Severity.REJECT, "grounded",
            "No retrieved source carries an Act, section, article or official URL, so the answer cannot be "
            "attributed to a verifiable provision.",
            tuple(checks),
        )

    checks.append("section_grounding")
    claimed_sections = set(re.findall(r"\bsections?\s+(\d{1,4}[A-Za-z]?)\b", body, re.IGNORECASE))
    supported_sections: set[str] = set()
    for chunk in ranked_chunks:
        metadata = getattr(chunk, "metadata", None) or {}
        section_number = str(metadata.get("section_number") or "").strip().rstrip(".")
        if section_number:
            supported_sections.add(section_number.lower())
        chunk_text = getattr(chunk, "text", "") or ""
        supported_sections.update(
            number.lower()
            for number in re.findall(r"\bsections?\s+(\d{1,4}[A-Za-z]?)\b", chunk_text, re.IGNORECASE)
        )
    # Confirmed live: an answer entirely grounded in BNS Section 303 (theft)
    # was rejected wholesale for ALSO citing "Section 379" -- the old,
    # pre-July-2024 IPC number for the exact same offence, which the model
    # volunteers unprompted because it is extremely well-known public
    # knowledge, not a fabrication. `IPC_TO_BNS_CROSSWALK` already records
    # which old numbers map to which current ones, each verified against the
    # Act's own section heading (see its own docstring) -- an old number is
    # accepted here ONLY when its crosswalked current number is itself
    # genuinely supported above, so this never launders an unrelated
    # invented citation, only the same true provision's other, historical
    # name.
    supported_sections.update(
        old for old, (new, _topic) in IPC_TO_BNS_CROSSWALK.items() if new.lower() in supported_sections
    )
    unsupported_sections = sorted(
        section for section in claimed_sections if section.lower() not in supported_sections
    )
    if unsupported_sections:
        return QualityVerdict(
            Severity.REJECT,
            "section_grounding",
            "The generated answer cited section(s) absent from the retrieved source material: "
            + ", ".join(unsupported_sections),
            tuple(checks),
        )

    checks.append("section_attribution")
    attribution_reason = _section_attribution_mismatch(body, ranked_chunks)
    if attribution_reason:
        return QualityVerdict(Severity.RETRY_SECTION_ATTRIBUTION, "section_attribution", attribution_reason, tuple(checks))

    checks.append("on_topic")
    topical = _on_topic(body, question=question, chunks=ranked_chunks, language=language)
    if topical is False:
        return QualityVerdict(
            Severity.REJECT, "on_topic",
            "The generated answer shares no substantive vocabulary with either the question or the retrieved "
            "sources.",
            tuple(checks),
        )

    checks.append("language_match")
    language_reason = _language_mismatch_reason(body, language)
    if language_reason:
        return QualityVerdict(Severity.RETRY_LANGUAGE, "language_match", language_reason, tuple(checks))

    return QualityVerdict(Severity.PASS, checks_run=tuple(checks))


def _on_topic(body: str, *, question: str, chunks: list[Any], language: str) -> bool | None:
    """`True` on topic, `False` clearly off topic, `None` when not checkable.

    Only runs when the answer is written in Latin script. For a Hindi or Tamil
    answer over English source text there is no shared vocabulary to measure --
    that is the normal, correct case, not evidence of anything, and treating
    the absence of overlap as a failure would reject every non-English answer
    the product exists to produce.
    """
    counts = _script_counts(body)
    letters = sum(counts.values())
    if not letters or counts.get("latin", 0) / letters < 0.6:
        return None
    answer_terms = _significant_terms(body)
    if not answer_terms:
        return None
    reference_terms = _significant_terms(question)
    for chunk in chunks[:6]:
        reference_terms |= _significant_terms(getattr(chunk, "text", "")[:2000])
        reference_terms |= _significant_terms(str(getattr(chunk, "metadata", "")))
    if not reference_terms:
        return None
    shared = answer_terms & reference_terms
    # A single shared word proves very little -- an answer about mortgages and
    # a chunk about FIR registration both contain "registered". With six
    # chunks' worth of reference vocabulary available, a genuinely on-topic
    # answer shares many terms, so requiring two is a wide margin rather than
    # a tight one. A terse answer (a one-line definition) is held to one,
    # since it may not have two distinctive terms to spare.
    required = 1 if len(answer_terms) < 8 else 2
    return len(shared) >= required


# Confirmed live: an otherwise-good, well-grounded FIR-procedure answer
# claimed "the right to escalate to the Superintendent of Police when an
# officer refuses to register your FIR is given in Section 64" -- Section
# 64 (rape's definition) was genuinely retrieved and IS a real, grounded
# section for this corpus, so `section_grounding` above passed it; the
# SP-escalation content itself is real law but was not actually present in
# any approved chunk this corpus retrieved (confirmed by inspecting the
# corpus directly), so the model attached a real claim it could not ground
# to a real but entirely unrelated section number rather than the section
# the claim actually concerns. `section_grounding` only checks that a cited
# number appears SOMEWHERE in the retrieved pool, never that the specific
# claim next to it belongs to that specific section -- this catches that
# narrower, harder failure shape as a bounded, self-correctable RETRY
# rather than a REJECT, matching `RETRY_LANGUAGE`'s own reasoning: the
# answer is very likely mostly correct, and asking the model to either cite
# the right section or drop the specific number for this one claim is a
# fix it can reliably make when told plainly.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")
_SECTION_MENTION_RE = re.compile(r"\bsections?\s+(\d{1,4}[A-Za-z]?)\b", re.IGNORECASE)
# A sentence needs several of its OWN substantive words before its
# section-number attribution is worth checking at all -- a bare aside like
# "(see also Section 64)" has nothing to compare and would only ever
# produce a spurious zero-overlap "mismatch".
_MIN_SENTENCE_TERMS_FOR_ATTRIBUTION_CHECK = 4


def _section_attribution_mismatch(body: str, ranked_chunks: list[Any]) -> str | None:
    """A sentence attributing a substantive claim to Section N, where N has
    its OWN retrieved chunk(s) but that chunk's text shares no vocabulary
    at all with the claim -- the shape of citing a real, grounded section
    number for a claim that section does not actually make.

    Deliberately narrow, matching this gate's own "wide margin" philosophy:
    only sections with their OWN metadata-tagged chunk(s) are checked (a
    section only ever mentioned in passing inside another chunk's running
    text has no chunk of its own to compare against, so is left alone,
    same as `_on_topic`'s "not checkable" case); a single shared
    significant word is enough to pass, since this targets a
    hallucination-shaped total mismatch, not a paraphrase-strictness bar.
    """
    section_chunks: dict[str, list[Any]] = {}
    for chunk in ranked_chunks:
        metadata = getattr(chunk, "metadata", None) or {}
        section_number = str(metadata.get("section_number") or "").strip().rstrip(".").lower()
        if section_number:
            section_chunks.setdefault(section_number, []).append(chunk)

    sentences = _SENTENCE_SPLIT_RE.split(body)
    for index, sentence in enumerate(sentences):
        mentions = _SECTION_MENTION_RE.findall(sentence)
        if not mentions:
            continue
        # Confirmed live: a citation and the claim it supports are often
        # split across a sentence boundary ("Ye process Section 64(4) mein
        # di gayi hai. Isme saaf hai ki agar police FIR darj karne se mana
        # kare...") -- the mention's OWN sentence can be nothing but the
        # bare citation, with the actual substantive claim in the sentence
        # right after (or, just as often, stated first with the citation
        # trailing it). A 3-sentence window centred on the mention -- prior,
        # own, and next -- catches both orderings without needing to parse
        # which side of the boundary the claim actually sits on.
        window = " ".join(sentences[max(0, index - 1):index + 2])
        sentence_terms = _section_attribution_terms(_SECTION_MENTION_RE.sub("", window))
        if len(sentence_terms) < _MIN_SENTENCE_TERMS_FOR_ATTRIBUTION_CHECK:
            continue
        for number in mentions:
            chunks_for_section = section_chunks.get(number.lower())
            if not chunks_for_section:
                continue
            reference_terms: set[str] = set()
            for chunk in chunks_for_section:
                reference_terms |= _section_attribution_terms(getattr(chunk, "text", "")[:2000])
            # A single shared word proves very little -- confirmed live,
            # "post" alone (meaning "by mail" in the FIR-refusal claim, but
            # "registered post" in a completely unrelated summons-service
            # clause of Section 64's own text) coincidentally overlapped
            # two genuinely unrelated passages. `_on_topic` above faces the
            # identical coincidental-word risk and requires 2 for the same
            # reason -- this window is guaranteed at least `_MIN_SENTENCE_
            # TERMS_FOR_ATTRIBUTION_CHECK` (4) terms by the check above, so
            # requiring 2 of them to genuinely recur in the section's own
            # (usually much longer) text is a wide margin, not a tight one.
            if not reference_terms or len(sentence_terms & reference_terms) >= 2:
                continue
            return (
                f"The generated answer attaches a specific claim to Section {number}, but that "
                "section's own retrieved text shares no vocabulary with the claim -- likely "
                "attributed to the wrong (but genuinely retrieved) section."
            )
    return None


def _language_mismatch_reason(body: str, language: str) -> str | None:
    resolved = (language or "").strip().lower()
    counts = _script_counts(body)
    letters = sum(counts.values())
    if letters < 20:
        return None
    latin = counts.get("latin", 0)

    if resolved in _LATIN_LANGUAGES:
        foreign = letters - latin
        if foreign / letters > _MAX_FOREIGN_SCRIPT_RATIO_FOR_LATIN:
            return (
                f"The question was asked in {resolved}, but the answer was written predominantly in another "
                "script."
            )
        return None

    expected = _LANGUAGE_SCRIPT.get(resolved)
    if expected is None:
        return None
    if counts.get(expected, 0) / letters < _MIN_EXPECTED_SCRIPT_RATIO:
        return f"The question was asked in {resolved}, but the answer contains almost no {resolved} text."
    return None
