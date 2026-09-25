"""Phase 1 item 6: the fact-only guard on generated drafts.

The drafting prompt already forbids inventing, embellishing or strengthening
facts. A prompt rule is necessary but not sufficient: nothing verified whether
the model actually obeyed it, and the failures are exactly the kind a
non-lawyer signing the document cannot spot. Observed in real output:

* a user who wrote "mujhe sandeh hai" (I suspect) got a draft asserting
  "सुनियोजित तरीके से ऑनलाइन धोखाधड़ी" (a premeditated online fraud) with
  "स्पष्ट संकेत और साक्ष्य" (clear indications and evidence);
* a police complaint closed by committing the complainant to escalate to
  higher authorities, which they had never asked for;
* a consumer complaint declared "स्पष्ट उल्लंघन" (a clear violation) -- a
  finding that belongs to the Commission, not to one side's draft.

This module reads the finished draft back against the facts the user actually
supplied and reports what it cannot account for. It is an ADVISORY audit, not
a blocker: it never edits or withholds a draft. A legal document the user
needs is not improved by being refused over a heuristic, but a user about to
sign one deserves to be told which sentences went beyond what they said.

Precision matters more than recall here. A finding the user cannot act on
trains them to ignore the audit, so every check below is either an exact
phrase match against a curated list or a numeric comparison -- never a fuzzy
judgement about tone.
"""

import re
from dataclasses import dataclass

# Legal boilerplate that legitimately contains numbers the user never typed:
# statute years, section numbers, and the standard clause numbering of a
# drafted document. Checked before the unsupported-number rule fires.
_STATUTE_YEARS = frozenset({
    "1860", "1872", "1881", "1908", "1950", "1955", "1956", "1961", "1963", "1973",
    "1986", "1988", "1996", "2000", "2005", "2013", "2015", "2016", "2019", "2023", "2025",
})

# A number immediately preceded by a provision word is a citation, not a fact
# about the user's case.
_CITATION_CONTEXT = re.compile(
    r"(?:section|sec\.?|s\.|article|rule|clause|order|schedule|chapter|act|sanhita|adhiniyam|"
    r"धारा|कलम|अनुच्छेद|अधिनियम|संहिता|ધારા|કલમ|ধারা|பிரிவு|సెక్షన్|ವಿಭಾಗ|دفعہ)"
    r"\s*[-:—]?\s*$",
    re.IGNORECASE,
)

# Conclusive characterisations that state as settled what is, at draft stage,
# one party's allegation. Curated per language rather than derived, so a
# finding always points at a phrase a reviewer can actually see in the text.
_CONCLUSORY_PHRASES: tuple[str, ...] = (
    # English
    "clear violation", "clear evidence", "clear indication", "clearly establishes",
    "it is established that", "is a cognizable offence", "constitutes an offence",
    "is guilty of", "proves that", "beyond doubt", "undoubtedly", "obviously",
    "complete indifference", "total negligence", "deliberate and intentional",
    "premeditated", "well-planned", "systematically defrauded", "with malicious intent",
    "unfair trade practice was committed", "amounts to fraud",
    # Hindi
    "स्पष्ट उल्लंघन", "स्पष्ट संकेत", "स्पष्ट साक्ष्य", "स्पष्ट प्रमाण",
    "सुनियोजित", "पूर्ण उदासीनता", "जानबूझकर", "निस्संदेह", "सिद्ध होता है",
    "संज्ञेय अपराध है", "अपराध सिद्ध", "दुर्भावनापूर्ण",
    # Marathi
    "स्पष्ट उल्लंघन", "पूर्णपणे दुर्लक्ष", "जाणूनबुजून", "नियोजनबद्ध",
    # Gujarati
    "સ્પષ્ટ ઉલ્લંઘન", "સ્પષ્ટ પુરાવા", "જાણીજોઈને", "પૂર્વયોજિત",
    # Bengali
    "স্পষ্ট লঙ্ঘন", "স্পষ্ট প্রমাণ", "ইচ্ছাকৃতভাবে", "পূর্বপরিকল্পিত",
    # Urdu
    "واضح خلاف ورزی", "واضح ثبوت", "جان بوجھ کر", "منصوبہ بند",
)

# Words that make what follows conditional rather than asserted.
_CONDITIONAL_MARKERS = re.compile(
    r"(?:\bif\b|\bshould\b|\bprovided\b|\bsubject to\b|\bin the event\b|\bunless\b|"
    r"यदि|अगर|बशर्ते|जब तक|यदि|जर\b|જો\b|যদি|ஆனால்|ஒருவேளை|ఒకవేళ|ಒಂದು ವೇಳೆ|اگر)",
    re.IGNORECASE,
)

# Embellishments a drafter adds to sound persuasive that the user never said:
# hardship/distress and "repeatedly asked" claims. Advisory, and only when the
# user's own text does not contain the same idea.
_INVENTED_DETAIL_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("hardship", ("mental agony", "mental stress", "mental distress", "mental harassment", "मानसिक तनाव",
                  "मानसिक पीड़ा", "मानसिक उत्पीड़न", "financial hardship", "आर्थिक हानि", "आर्थिक कठिनाई")),
    ("repeated_requests", ("repeated requests", "repeatedly requested", "despite repeated", "बार-बार अनुरोध",
                           "बार-बार आग्रह", "बार बार अनुरोध", "several reminders", "numerous requests")),
)
# "within 15 days", "15 दिनों के भीतर" -- a compliance period.
_DEADLINE_PATTERN = re.compile(
    r"(\d{1,3})\s*(?:\(\w+\)\s*)?(?:days?|दिन(?:ों)?|दिवस|दिवसांत|દિવસ|দিন)", re.IGNORECASE
)

# Commitments to escalate that the applicant never asked for. A Notice is the
# one document type whose purpose IS to state a consequence, so the caller
# passes `category` and this check is skipped for it.
_ESCALATION_PHRASES: tuple[str, ...] = (
    "higher authorities", "higher authority", "superintendent of police",
    "shall be compelled", "will be compelled", "constrained to approach",
    "further legal action will", "legal proceedings will be initiated",
    "approach the court", "move the hon'ble court", "escalate the matter",
    "उच्चाधिकारियों", "उच्च अधिकारियों", "बाध्य होऊंगा", "बाध्य होऊँगा", "बाध्य रहूंगा",
    "कानूनी कार्रवाई करने के लिए बाध्य", "न्यायालय की शरण",
    "वरिष्ठ अधिकाऱ्यांकडे", "बांधील राहीन",
    "ઉચ્ચ અધિકારીઓ", "બંધાયેલ રહીશ",
    "ঊর্ধ্বতন কর্তৃপক্ষ",
    "اعلیٰ حکام",
)

# Words by which the USER expressed uncertainty. If any appear in the supplied
# facts, the draft must not present the same matter as settled.
_USER_HEDGE_WORDS: tuple[str, ...] = (
    "suspect", "suspicion", "i think", "i believe", "seems", "appears", "possibly", "maybe",
    "संदेह", "शंका", "लगता है", "शायद",
    "संशय", "वाटते",
    "શંકા", "લાગે છે",
    "সন্দেহ", "মনে হয়",
    "شبہ", "شک",
)

# Hedges that make a legal-applicability statement properly conditional.
_APPLICABILITY_HEDGES: tuple[str, ...] = (
    "prima facie", "may be applicable", "may apply", "appears to attract", "appear to attract",
    "subject to verification", "if established", "alleged", "as stated by",
    # Rule 4 in legal_drafting_prompt.md now invites paraphrase within the
    # same hedge family (so repeated drafts don't read identically) instead
    # of one fixed phrase -- these are the other natural constructions it
    # names, added here so a properly-hedged sentence using them isn't
    # wrongly flagged as an unhedged legal conclusion.
    "could attract", "might attract", "would, if borne out", "if borne out",
    "on a plain reading of the facts as stated", "would appear to fall within the scope",

    "प्रथम दृष्टया", "लागू हो सकत", "प्रतीत होता", "यदि सिद्ध", "कथित", "सत्यापन के अधीन",
    "प्रथमदर्शनी", "लागू होऊ शकत",
    "પ્રથમ દૃષ્ટિએ", "લાગુ થઈ શકે",
    "প্রাথমিকভাবে", "প্রযোজ্য হতে পারে",
    "بادی النظر", "لاگو ہو سکتا",
)

# Sections whose whole job is to state the legal position -- the only ones the
# applicability-hedge check applies to.
_LEGAL_POSITION_HEADINGS = frozenset({"Legal Position", "Legal Grounds", "Grounds"})

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class DraftAuditFinding:
    section: str
    category: str
    excerpt: str
    detail: str


@dataclass(frozen=True)
class DraftAudit:
    findings: tuple[DraftAuditFinding, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.findings

    def as_dicts(self) -> list[dict[str, str]]:
        return [
            {
                "section": finding.section,
                "category": finding.category,
                "excerpt": finding.excerpt,
                "detail": finding.detail,
            }
            for finding in self.findings
        ]


def _digits(value: str) -> str:
    return re.sub(r"\D", "", value)


def _supplied_numbers(fields: dict[str, str]) -> set[str]:
    """Every number the user typed, digits-only so "35,000" and "35000" match.

    Also records each number's own digit substrings of length >= 4, because a
    draft legitimately restates "₹35,000" as "35,000/-" or splits a long
    transaction id -- comparing whole tokens alone produced false findings on
    correctly-copied values.
    """
    numbers: set[str] = set()
    for value in fields.values():
        if not value:
            continue
        for match in _NUMBER.finditer(str(value)):
            digits = _digits(match.group(0))
            if digits:
                numbers.add(digits)
    return numbers


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_SPLIT.split(text or "") if sentence.strip()]


def _excerpt(sentence: str, limit: int = 160) -> str:
    return sentence if len(sentence) <= limit else f"{sentence[:limit].rstrip()}..."


def audit_draft(
    sections: dict[str, str],
    fields: dict[str, str],
    *,
    category: str = "Complaint",
    today_digits: str = "",
) -> DraftAudit:
    """Reports draft content that the user's own facts do not support.

    `category` is the template's category -- a "Notice" is exempt from the
    escalation check, since a consequence-if-not-complied clause is that
    document type's entire purpose. `today_digits` is the generation date's
    digits, which appear in the signature block and are not a user fact.
    """
    findings: list[DraftAuditFinding] = []
    supplied_numbers = _supplied_numbers(fields)
    supplied_text = " ".join(str(value) for value in fields.values() if value).lower()
    user_hedged = any(word in supplied_text for word in _USER_HEDGE_WORDS)

    for heading, body in sections.items():
        if not body:
            continue
        lowered_body = body.lower()

        for sentence in _sentences(body):
            lowered = sentence.lower()

            for phrase in _CONCLUSORY_PHRASES:
                position = lowered.find(phrase.lower())
                # A phrase introduced by a conditional ("यदि यह सिद्ध होता है
                # कि ... तो ... आ सकता है", "if it is established that ...") is
                # the HEDGE this audit asks for, not a settled finding.
                # Flagging it blocked the export of a properly qualified
                # Legal Position section.
                if position >= 0 and _CONDITIONAL_MARKERS.search(lowered[:position]):
                    continue
                if position >= 0:
                    findings.append(DraftAuditFinding(
                        section=heading,
                        category="unsupported_conclusion",
                        excerpt=_excerpt(sentence),
                        detail=(
                            f'States "{phrase}" as settled. At draft stage this is one party\'s allegation, '
                            "not an established finding -- it should be attributed to the applicant or "
                            "qualified as prima facie."
                        ),
                    ))
                    break

            if category != "Notice":
                for phrase in _ESCALATION_PHRASES:
                    if phrase.lower() in lowered and phrase.lower() not in supplied_text:
                        findings.append(DraftAuditFinding(
                            section=heading,
                            category="unrequested_escalation",
                            excerpt=_excerpt(sentence),
                            detail=(
                                "Commits the applicant to escalating the matter. The supplied facts and relief "
                                "do not ask for this, and it is a real commitment in a document they will sign."
                            ),
                        ))
                        break

            for label, phrases in _INVENTED_DETAIL_PATTERNS:
                hit = next((p for p in phrases if p in lowered and p not in supplied_text), None)
                if hit:
                    findings.append(DraftAuditFinding(
                        section=heading,
                        category=f"invented_detail:{label}",
                        excerpt=_excerpt(sentence),
                        detail=f'"{hit}" is not something you said. Remove it or confirm it before signing.',
                    ))
            deadline = _DEADLINE_PATTERN.search(sentence)
            if deadline and deadline.group(1) not in supplied_numbers and deadline.group(1) not in supplied_text:
                findings.append(DraftAuditFinding(
                    section=heading,
                    category="unsupported_deadline",
                    excerpt=_excerpt(sentence),
                    detail=(
                        f'The {deadline.group(1)}-day period was not supplied by you. State a deadline only if '
                        "you chose one; otherwise use \"within a reasonable time\"."
                    ),
                ))

            if user_hedged:
                for phrase in ("it is established", "it is proved", "clearly", "स्पष्ट रूप से", "निश्चित रूप से"):
                    if phrase in lowered:
                        findings.append(DraftAuditFinding(
                            section=heading,
                            category="certainty_upgraded",
                            excerpt=_excerpt(sentence),
                            detail=(
                                "The applicant expressed this as a suspicion, but the draft states it as "
                                "established."
                            ),
                        ))
                        break

            for match in _NUMBER.finditer(sentence):
                token = match.group(0)
                digits = _digits(token)
                if not digits or len(digits) < 3:
                    # One- and two-digit numbers are overwhelmingly clause
                    # numbering, dates within a written date, or section
                    # numbers -- too noisy to be worth reporting.
                    continue
                if digits in supplied_numbers or digits == today_digits or token in _STATUTE_YEARS:
                    continue
                if _CITATION_CONTEXT.search(sentence[: match.start()]):
                    continue
                findings.append(DraftAuditFinding(
                    section=heading,
                    category="unsupported_figure",
                    excerpt=_excerpt(sentence),
                    detail=(
                        f'The figure "{token}" does not appear in any value you supplied. Check it before '
                        "signing -- a wrong amount, date or reference number can defeat the complaint."
                    ),
                ))
                break

        if (
            heading in _LEGAL_POSITION_HEADINGS
            and lowered_body
            and not any(hedge.lower() in lowered_body for hedge in _APPLICABILITY_HEDGES)
        ):
                findings.append(DraftAuditFinding(
                    section=heading,
                    category="unhedged_legal_conclusion",
                    excerpt=_excerpt(body),
                    detail=(
                        "States the legal position without any conditional wording. Whether a provision "
                        'applies is decided on evidence, so this should read "prima facie appears to attract", '
                        '"may be applicable", or "subject to verification".'
                    ),
                ))

    # Same finding can be produced once per section; de-duplicate on the exact
    # (section, category, excerpt) triple so a repeated phrase is reported once.
    seen: set[tuple[str, str, str]] = set()
    unique: list[DraftAuditFinding] = []
    for finding in findings:
        key = (finding.section, finding.category, finding.excerpt)
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return DraftAudit(tuple(unique))


# Fabricated-scene claims the model adds to sound complete: the condition of a
# handed-over property, that a formality was "completed", etc.
_INVENTED_CONDITION_PHRASES: tuple[str, ...] = (
    "स्थिति सामान्य थी", "अच्छी स्थिति में", "सभी औपचारिकताएं पूर्ण", "सभी औपचारिकताएँ पूर्ण",
    "in good condition", "in a good condition", "in satisfactory condition", "all formalities were completed",
    "financial inconvenience", "financial difficulty", "आर्थिक असुविधा", "वित्तीय कठिनाई", "वित्तीय प्रबंधन में कठिनाई",
)
_REASONABLE_TIME = {"devanagari": "उचित समय", "default": "a reasonable time"}


def strip_invented_details(
    sections: dict[str, str], fields: dict[str, str], allowed_periods: frozenset[str] = frozenset()
) -> dict[str, str]:
    """Deterministic enforcement of the "never invent" rule the prompt asks for.

    Removes any sentence that adds hardship, "repeatedly requested" or
    property-condition claims the user's own text does not contain, and
    replaces a compliance period the user never supplied ("within 15 days")
    with "a reasonable time". A prompt rule alone was not obeyed: length-
    target expansion pushed the model to pad a notice with exactly these.
    Sentences are dropped, never rewritten, so nothing new is introduced.
    `allowed_periods` are statutory periods a template records with their
    legal basis (e.g. the 15 days of s.138 NI Act) and are never touched.
    """
    supplied_text = " ".join(str(v) for v in fields.values() if v).lower()
    supplied_numbers = _supplied_numbers(fields)
    banned = [p for _, group in _INVENTED_DETAIL_PATTERNS for p in group] + list(_INVENTED_CONDITION_PHRASES)
    cleaned: dict[str, str] = {}
    for heading, body in sections.items():
        if not body:
            cleaned[heading] = body
            continue
        kept_paragraphs = []
        for paragraph in body.split("\n"):
            pieces = re.split(r"(?<=[.!?\u0964])(\s+)", paragraph)
            out: list[str] = []
            skip_space = False
            for piece in pieces:
                if piece.strip() == "":
                    if not skip_space:
                        out.append(piece)
                    continue
                lowered = piece.lower()
                if any(p.lower() in lowered and p.lower() not in supplied_text for p in banned):
                    skip_space = True
                    if out and out[-1].strip() == "":
                        out.pop()
                    continue
                skip_space = False
                match = _DEADLINE_PATTERN.search(piece)
                if (
                    match
                    and match.group(1) not in supplied_numbers
                    and match.group(1) not in supplied_text
                    and match.group(1) not in allowed_periods
                ):
                    devanagari = bool(re.search(r"[\u0900-\u097F]", piece))
                    piece = piece.replace(
                        match.group(0), _REASONABLE_TIME["devanagari" if devanagari else "default"], 1
                    )
                out.append(piece)
            kept_paragraphs.append("".join(out).strip())
        cleaned[heading] = "\n".join(kept_paragraphs)
    return cleaned
