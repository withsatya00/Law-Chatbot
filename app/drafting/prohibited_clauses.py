"""Post-Phase-3 hardening (Phase 2, milestone C): statements a generated draft
must not make unless the user actually supplied the fact behind them.

Why a scanner and not just careful wording
------------------------------------------
The deterministic fallback's wording is fixed and has been corrected in
`officialese.py` and `fallback_phrases.py`. The LLM path is not fixed: it is
asked for an advocate-style document, and "a failure to reply will be treated
as an admission" / "we will initiate criminal proceedings" / "you are liable
for the mental agony caused" are exactly the sentences a model reaches for when
told to sound like a legal notice. Correcting the templates closes one door;
this closes the other, and it keeps closing it as prompts and models change.

What this is and is not
-----------------------
It is an **advisory scan**, surfaced through the same `audit_findings` channel
as the fact-only audit -- the user sees, at the moment they read the draft,
that a specific sentence asserts something they never said. It does not block
generation or export: a person who needs a document is not helped by being
refused one over a regex, and they ARE helped by being told which sentence to
delete before they sign it.

Each rule names the exact harm, because these are not stylistic preferences:

* `silence_as_admission` -- telling a recipient that not replying admits the
  facts is a statement about the legal effect of their silence, and it is not
  one this system has any source for. It also pressures a lay recipient into
  responding to a claim they may have every right to ignore.
* `automatic_criminal_proceedings` -- threatening prosecution to obtain money
  is a serious allegation to put in a private letter, and whether any offence
  is even made out is not something a drafting tool can know.
* `unclaimed_mental_distress` -- a head of damage the user never pleaded. If
  they did suffer it, they must say so themselves, in their own words.
* `invented_money_claim` -- interest, costs and damages are amounts, and an
  amount nobody supplied is a fabricated figure.
* `evidence_not_supplied` -- asserting the sender "holds" or "encloses"
  documents when the user listed none tells the recipient (and later, a court)
  that evidence exists.

A rule fires only when the corresponding user field is ABSENT. A notice that
demands interest because the user asked for interest is the user's document
saying what the user wants, and nothing here interferes with that.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ProhibitedClause:
    key: str
    pattern: re.Pattern[str]
    # Field keys whose non-blank value makes this statement the user's own.
    licensed_by: tuple[str, ...]
    explanation: str
    # When true, the rule also stands down if the SAME pattern matches
    # something the user typed. This is the licence that matters in practice:
    # naming a field key does not work for `expected_relief` or `facts`, which
    # are filled on essentially every draft, so keying off their presence would
    # disable the rule everywhere. Keying off their CONTENT is exact -- a user
    # who wrote "I want interest and costs" has asked for interest and costs,
    # and the draft is then repeating them, not inventing them.
    self_licensing: bool = True


_RULES: tuple[ProhibitedClause, ...] = (
    ProhibitedClause(
        "silence_as_admission",
        re.compile(
            r"(failure|omission)\s+to\s+(respond|reply|answer)[^.]{0,120}\badmission\b"
            r"|\bdeemed\s+to\s+have\s+admitted\b"
            r"|\btreated\s+as\s+an\s+admission\b"
            r"|उत्तर\s+न\s+देने[^।]{0,80}स्वीकृत"
            r"|मौन[^।]{0,60}स्वीकृति",
            re.IGNORECASE,
        ),
        (),
        "This tells the recipient that not replying admits your account. Nothing in the material "
        "behind this draft supports that, and it is not a consequence you can create by stating it.",
        # Never licensed, not even by the user asking for it: this is a claim
        # about the legal effect of another person's silence, which is wrong
        # however it got into the document. Advisory, so a note is the whole
        # cost of being firm here.
        self_licensing=False,
    ),
    ProhibitedClause(
        "automatic_criminal_proceedings",
        re.compile(
            r"\bcriminal\s+(proceedings?|prosecution|case|complaint|action)\b"
            r"|\bcivil\s+and\s*/?\s*or\s+criminal\b"
            r"|\bprosecut(e|ed|ion)\b"
            r"|आपराधिक\s+(कार्यवाही|मुकदमा|अभियोजन)",
            re.IGNORECASE,
        ),
        (),
        "This threatens a criminal case. Whether any offence is made out is for the police or a court "
        "to decide on the facts, and this draft has no basis recorded for saying one is.",
    ),
    ProhibitedClause(
        "unclaimed_mental_distress",
        re.compile(
            r"\bmental\s+(distress|agony|harassment|torture|trauma|anguish)\b"
            r"|\bemotional\s+distress\b"
            r"|मानसिक\s+(क्लेश|पीड़ा|प्रताड़ना|उत्पीड़न|आघात)",
            re.IGNORECASE,
        ),
        ("loss_suffered", "damages_claimed"),
        "This claims mental distress. If you did suffer it, say so in your own words in the facts, "
        "so the document records what you actually experienced rather than a standard phrase.",
    ),
    ProhibitedClause(
        "invented_money_claim",
        re.compile(
            r"\b(claim|recover|demand)\s+(?:for\s+)?interest,?\s*(?:and\s+)?(?:costs?|damages)\b"
            r"|\binterest,\s*costs?\s+and\s+damages\b"
            r"|\btogether\s+with\s+interest\s+at\s+\d"
            r"|ब्याज,\s*व्यय\s+एवं\s+क्षतिपूर्ति",
            re.IGNORECASE,
        ),
        ("interest_claimed",),
        "This claims interest, costs or damages that you did not ask for. An amount nobody supplied "
        "is a figure this draft would be inventing on your behalf.",
    ),
    ProhibitedClause(
        "evidence_not_supplied",
        re.compile(
            r"\b(enclosed|annexed|attached)\s+herewith\b"
            r"|\bcopies\s+of\s+the\s+(documents|receipts|invoices)\s+are\s+(enclosed|annexed|attached)\b"
            r"|संलग्न\s+(है|हैं|किए\s+जा\s+रहे)",
            re.IGNORECASE,
        ),
        ("available_documents", "annexures"),
        "This says documents are enclosed with the draft. You have not listed any, so nothing would "
        "actually be attached to it.",
    ),
)


def scan(
    sections: dict[str, str],
    fields: dict[str, str],
    permitted: "tuple[str, ...] | frozenset[str]" = (),
) -> list[dict[str, str]]:
    """Advisory findings for `sections`, in the same `section`/`category`/
    `excerpt`/`detail` shape `FactAudit.as_dicts` produces, so both audits reach
    the user through one `audit_findings` list.

    A rule stands down in three cases:

    * one of its `licensed_by` fields carries a value -- the fact is the
      user's;
    * it is `self_licensing` and the user's own typed text makes the same
      statement -- again, the draft is repeating them, not inventing;
    * the template declares the statement permitted, in its YAML
      `permitted_statements`, with a `statutory_basis` recorded beside it.
      That last one exists for exactly one situation so far: a statutory notice
      under s.138 of the Negotiable Instruments Act MUST state that failure to
      pay will lead to a complaint under s.138 read with s.142 -- that is what
      makes it a s.138 notice. Suppressing it would break the document. The
      permission is per template and declared as data, never inferred.
    """
    permitted_keys = frozenset(permitted)
    user_text = "\n".join(value for value in fields.values() if value)
    findings: list[dict[str, str]] = []
    for rule in _RULES:
        if rule.key in permitted_keys:
            continue
        if any((fields.get(key) or "").strip() for key in rule.licensed_by):
            continue
        if rule.self_licensing and rule.pattern.search(user_text):
            continue
        for heading, text in sections.items():
            match = rule.pattern.search(text or "")
            if not match:
                continue
            findings.append(
                {
                    "section": heading,
                    "category": f"unsupported_statement:{rule.key}",
                    "excerpt": match.group(0).strip(),
                    "detail": rule.explanation,
                }
            )
            break
    return findings
