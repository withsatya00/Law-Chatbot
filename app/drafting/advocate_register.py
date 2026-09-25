"""Advocate-register redesign: the structural elements missing between this
app's deterministic drafts and a real advocate's drafting convention.

Confirmed against a side-by-side comparison of a lawyer's own "FIR
registration application" and this app's equivalent output: the lawyer's
draft (1) opens with a sworn identification -- "I, Sunita Devi, W/o Ramesh
Sharma, aged about 35 years, residing at ..., and holding Mobile Number ...,
do hereby state as under:-" -- instead of a plain name/address/mobile list;
(2) cites the exact relevant sections in the Subject line; (3) uses classical
prayer phrasing ("It is therefore most respectfully prayed that this
Honorable Authority may kindly be pleased to ..."); and (4) lists supporting
documents under "Enclosures". None of this touches the user's own narrative
-- every sentence here is built from structured fields (name, address,
mobile, age, father's/husband's name) and static per-language text, the same
class of input `officialese.py` already safely uses.

The sworn-opening/classical-prayer register is scoped to Complaint (always)
and Application (per-template, via `DraftTemplateDefinition.advocate_register`)
-- see that field's own docstring in `templates/base.py` for why an RTI
application or a certificate request wants this register and a resignation
letter or job application does not. Notice keeps its own, deliberately
different register (no sworn declaration, no "receiving authority" -- a
notice's author is a private party writing to another private party, not a
petitioner before an authority) and none of that is touched here.

Subject-line citation is the one piece Notice DOES share: confirmed against
real Indian legal-notice formats (cheque-bounce notices under s.138 NI Act,
money-recovery notices) that the converging convention is a Subject line
naming the exact section being invoked ("Legal notice under Section 138 of
the Negotiable Instruments Act...") -- see `supports_citation`, which is
broader than `supports_advocate_register` for exactly this reason. What real
notice templates do NOT do, confirmed by the same research, is claim to be
issued by an advocate on the sender's "instructions and authority" -- this
app has no advocate to attribute a notice to, so that convention is
deliberately not replicated; the sender speaks for themselves throughout.

FLOWING-LETTER FORMAT (`FLOWING_LETTER_DRAFTS`): a second, deeper finding
from the same side-by-side comparison -- the lawyer's FIR-registration
application has almost NO internal bold headings at all. "To,"/the SHO's
address is a plain paragraph, "Subject:" is an inline prefix not a labelled
section, and everything from "Respected Sir/Madam," through the sworn
opening, the numbered facts, and "It is therefore, most respectfully
prayed..." reads as ONE continuous letter -- no "Facts of the Case" heading,
no "Legal Position" heading, and critically no "Prayer" heading; the prayer
clause is just the letter's closing paragraph. Confirmed against a second,
independent real complaint template (advocategandhi.com) showing the same
pattern for the request/prayer specifically. This is scoped to the 9
"SHO representation" Complaint templates (`FLOWING_LETTER_DRAFTS`, the same
set `_police_request_block` already treats differently in engine.py) --
NOT the 3 consumer-forum complaints (`consumer_complaint`,
`ecommerce_complaint`, `service_complaint`), which research confirmed use
the OPPOSITE convention: a real District Consumer Disputes Redressal
Forum/Commission complaint is even MORE formally headed than this app's
current output (a full cause title -- "IN RE: COMPLAINT No. ___ of 20__ ...
VERSUS ..." -- plus labelled INTRODUCTION/TRANSACTION/DEFECT-DEFICIENCY/
RECTIFICATION/EVIDENCE/JURISDICTION/PRAYER CLAUSE sections and a separate
VERIFICATION), because it is a pleading before a quasi-judicial tribunal,
not a letter to an office. Left untouched for that reason.
"""

import re

from app.drafting.fallback_phrases import phrase
from app.drafting.officialese import SCAFFOLDED_LANGUAGES
from app.drafting.templates.base import DraftTemplateDefinition

REGISTER_CATEGORIES = frozenset({"Complaint", "Application"})
CITATION_CATEGORIES = frozenset({"Complaint", "Application", "Notice"})

# The 9 "SHO representation" Complaint templates -- same set
# `LegalDraftEngine._police_request_block` already treats differently
# (heading "Request" not "Prayer"). These render as one continuous flowing
# letter (see module docstring); every other category/template keeps its
# existing headed layout.
FLOWING_LETTER_DRAFTS = frozenset({
    "bank_fraud_complaint", "cyber_crime_complaint", "missing_person_report",
    "mobile_theft_complaint", "online_fraud_complaint", "police_complaint",
    "sp_complaint", "threat_complaint", "vehicle_theft_complaint",
})

# The order a flowing letter's body paragraphs read in -- confirmed against
# the real lawyer draft: salutation, then the sworn identification, then the
# numbered facts, then the legal grounding, then the prayer, then
# verification. NOT the order these sit in `COMPLAINT_SECTIONS` (which puts
# "Complainant Details" before "Introduction" -- correct for the headed
# layout's own visual hierarchy, wrong for one continuous letter). Any
# section present but not listed here (should not happen for this category,
# but never silently dropped) is appended after, in whatever order it was
# already in.
FLOWING_BODY_ORDER = (
    "Introduction", "Complainant Details", "Facts of the Case",
    "Legal Position", "Consequences", "Prayer", "Request", "Verification",
)
# Rendered as their own plain (unheaded) paragraphs, before the flowing body.
FLOWING_LEAD_HEADINGS = ("Recipient", "Subject")
# Still gets a small heading, same as today -- the lawyer's own draft sets
# its Enclosures list apart from the letter body the same way.
FLOWING_KEEP_HEADING = ("Enclosures", "Annexures")


def supports_advocate_register(language: str, template: DraftTemplateDefinition) -> bool:
    return (
        template.category in REGISTER_CATEGORIES
        and template.advocate_register
        and language in SCAFFOLDED_LANGUAGES
    )


def supports_citation(language: str, template: DraftTemplateDefinition) -> bool:
    """Whether the Subject line should cite the template's own
    `applicable_sections_hint`, when it has any. Deliberately broader than
    `supports_advocate_register` -- see the module docstring for the
    research backing this: a Notice's subject line conventionally names the
    section being invoked even though Notice gets none of the rest of the
    advocate register (sworn opening, classical prayer, Enclosures).
    """
    return template.category in CITATION_CATEGORIES and language in SCAFFOLDED_LANGUAGES


def sworn_opening_block(fields: dict[str, str], language: str) -> str:
    """"I, {name}, {S/o,D/o,W/o clause}, {aged clause}, residing at
    {address}, and holding Mobile Number {mobile}, do hereby state as
    under:-"

    Every clause but the name is optional and degrades grammatically when
    blank -- true for every existing draft today, since `applicant_age`/
    `applicant_father_name` are brand-new fields nobody has filled in yet,
    and true even for `applicant_address` on at least one template
    (`bonafide_certificate_application`, addressed to the student's own
    institution, never asks for one). Never a placeholder blank -- the
    clause is simply omitted, not rendered as "aged about ___ years".
    """
    name = fields.get("applicant_name", "").strip()
    if not name:
        return phrase("not_provided", language, "Not provided.")
    relation = fields.get("applicant_father_name", "").strip()
    age = fields.get("applicant_age", "").strip()
    address = fields.get("applicant_address", "").strip()
    mobile = fields.get("applicant_mobile", "").strip()

    if language == "hindi":
        clauses = [
            phrase("son_daughter_wife_of", "hindi", "S/o, D/o, W/o {name}", name=relation) if relation else "",
            f"आयु लगभग {age} वर्ष" if age else "",
            f"निवासी {address}" if address else "",
            f"तथा मोबाइल नंबर {mobile} धारक" if mobile else "",
        ]
        joined = ", ".join(clause for clause in clauses if clause)
        return f"मैं, {name}{', ' + joined if joined else ''}, निम्नानुसार कथन करता/करती हूँ:-"

    clauses = [
        phrase("son_daughter_wife_of", language, "S/o, D/o, W/o {name}", name=relation) if relation else "",
        f"aged about {age} years" if age else "",
        f"residing at {address}" if address else "",
        f"holding Mobile Number {mobile}" if mobile else "",
    ]
    clauses = [clause for clause in clauses if clause]
    if not clauses:
        joined = ""
    elif len(clauses) == 1:
        joined = f", {clauses[0]}"
    else:
        joined = ", " + ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    return f"I, {name}{joined}, do hereby state as under:-"


def classical_prayer_intro(
    language: str, applicable_sections_hint: list[str], applicable_acts_hint: list[str]
) -> str | None:
    """The classical "It is therefore most respectfully prayed that this
    Honorable Authority may kindly be pleased to..." introductory clause,
    only when the template actually names a section to cite -- never
    invents a citation a template doesn't have. Returns `None` (caller
    falls back to the existing generic `prayer_intro` phrase) when
    `applicable_sections_hint` is empty.
    """
    if not applicable_sections_hint:
        return None
    citation = ", ".join(applicable_sections_hint)
    if language == "hindi":
        citation = re.sub(r"\bSection\s+(\d+[A-Z]?)\b", r"धारा \1", citation, flags=re.IGNORECASE)
        return (
            f"अतः यह अत्यंत सम्मानपूर्वक प्रार्थना की जाती है कि यह सक्षम प्राधिकारी {citation} के अंतर्गत "
            "उपरोक्त विषय पर विचार करने तथा निम्नलिखित हेतु कृपा करे:"
        )
    return (
        f"It is therefore most respectfully prayed that this Honorable Authority may kindly be pleased to "
        f"consider the above under {citation}, and to:"
    )


def append_citation(
    subject: str, language: str, applicable_sections_hint: list[str], applicable_acts_hint: list[str]
) -> str:
    """Appends a "under Section X of Y" citation clause to an already-built
    Subject line, using only per-template data that already exists
    (`applicable_sections_hint`) -- never a citation invented from the
    user's own free text.

    A no-op when the subject text already names the section: some
    `subject_template`s (e.g. cheque_bounce_notice's own "Statutory notice
    under Section 138, Negotiable Instruments Act, 1881 -- dishonour of
    cheque no. {cheque_number}") already cite it directly -- confirmed live,
    appending unconditionally produced "...under Section 138... under
    Section 138, Section 142", the same citation stated twice in one line.
    """
    if not applicable_sections_hint or not subject:
        return subject
    if any(hint.lower() in subject.lower() for hint in applicable_sections_hint):
        return subject
    citation = ", ".join(applicable_sections_hint)
    if language == "hindi":
        return f"{subject.rstrip('.')} ({citation} के अंतर्गत)"
    return f"{subject.rstrip('.')} under {citation}"


def enclosures_block(fields: dict[str, str], template: DraftTemplateDefinition, language: str) -> str | None:
    """Two-tier: prefers the user's own confirmed `available_documents`
    verbatim (identical to how "Annexures" already works elsewhere in this
    app); falls back to a HEDGED suggestion built from the template's
    `typical_supporting_documents` only when the user supplied nothing of
    their own. Returns `None` (no section at all) when neither exists --
    never a claim that something is attached when it is not.

    The hedge is deliberately phrased "should be enclosed where available"
    (prescriptive/conditional), never "enclosed herewith"/"is enclosed"
    (an assertive past-tense claim) -- the exact phrasing
    `prohibited_clauses.py`'s `evidence_not_supplied` rule exists to catch.
    See `tests/test_prohibited_clauses.py` for the regression test
    confirming this sentence never trips that rule.
    """
    available = fields.get("available_documents", "").strip()
    if available:
        return available
    if not template.typical_supporting_documents:
        return None
    items = "\n".join(f"{index}. {doc}" for index, doc in enumerate(template.typical_supporting_documents, start=1))
    if language == "hindi":
        return (
            "इस प्रकार के आवेदन/शिकायत हेतु सामान्यतः निम्नलिखित दस्तावेज़ आवश्यक होते हैं, तथा जहाँ उपलब्ध हों "
            f"वहाँ उनकी प्रतियाँ संलग्न की जानी चाहिए:\n{items}"
        )
    return (
        "Documents ordinarily required to support a filing of this kind include the following, and copies "
        f"should be enclosed where available:\n{items}"
    )


def classical_request_intro(
    language: str, applicable_sections_hint: list[str], applicable_acts_hint: list[str]
) -> str | None:
    """`classical_prayer_intro`'s counterpart for a representation to the
    police (an application to an SHO). That is a REQUEST to an officer, not a
    prayer to a court: "most respectfully prayed ... Honorable Authority
    may kindly be pleased" reads like a court pleading and was flagged in QA
    as the wrong register for a police application. Same rule as the
    original: only when the template names a section, never invents one.
    """
    if not applicable_sections_hint:
        return None
    citation = ", ".join(applicable_sections_hint)
    if language == "hindi":
        citation = re.sub(r"\bSection\s+(\d+[A-Z]?)\b", r"धारा \1", citation, flags=re.IGNORECASE)
        return f"अतः विनम्र अनुरोध है कि {citation} के अंतर्गत उपरोक्त शिकायत पर कार्यवाही करते हुए निम्नलिखित करने की कृपा करें:"
    return f"It is therefore respectfully requested that you kindly take action on the above under {citation}, and:"
