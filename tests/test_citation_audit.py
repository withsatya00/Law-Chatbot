"""Citation verification: a generated draft that cites a specific statute
section is checked against the template's own curated
`applicable_sections_hint`, catching a substituted or hallucinated section
number before the user files a document citing it. See
`app/drafting/citation_audit.py` for why this is scoped to section numbers
only, and only when a template actually has curated hints to check against.
"""

from app.drafting import citation_audit
from app.drafting.templates import get_template


def test_a_hallucinated_section_number_is_flagged() -> None:
    sections = {"Legal Position": "This notice is issued under Section 420 of the applicable law."}
    findings = citation_audit.verify_citations(sections, ["Section 138", "Section 142"])
    assert [finding["category"] for finding in findings] == ["unverified_citation"]
    assert "420" in findings[0]["detail"]


def test_a_correctly_cited_section_is_not_flagged() -> None:
    sections = {"Legal Position": "This notice is issued under Section 138 of the Negotiable Instruments Act."}
    assert citation_audit.verify_citations(sections, ["Section 138", "Section 142"]) == []


def test_a_sub_lettered_citation_of_a_hinted_number_is_not_flagged() -> None:
    """"66C" is a real sub-section of the hinted "Section 66" -- this check
    verifies the core number, not letter-perfect sub-clause matching."""
    sections = {"Legal Position": "The accused is liable under Section 66C of the Information Technology Act."}
    assert citation_audit.verify_citations(sections, ["Section 66"]) == []


def test_a_hindi_citation_is_recognized() -> None:
    sections = {"Legal Position": "यह सूचना नकारात्मक लिखत अधिनियम की धारा 138 के तहत जारी की जाती है।"}
    assert citation_audit.verify_citations(sections, ["Section 138"]) == []
    sections_wrong = {"Legal Position": "यह सूचना धारा 420 के तहत जारी की जाती है।"}
    findings = citation_audit.verify_citations(sections_wrong, ["Section 138"])
    assert len(findings) == 1


def test_a_template_with_no_curated_hints_is_never_checked() -> None:
    """Contract-category templates leave `applicable_sections_hint` empty and
    number their own clauses ("Section 5" of the agreement) -- that must never
    be misread as an uncited statute reference."""
    sections = {"Terms and Conditions": "Section 5 of this Agreement governs termination."}
    assert citation_audit.verify_citations(sections, []) == []


def test_a_citation_naming_a_number_absent_from_a_template_with_no_number_hints_is_flagged() -> None:
    """`bank_fraud_complaint.yaml` hints at an RBI circular, not a specific
    criminal section -- if a draft still cites one, that section number
    cannot have come from this template's own guidance."""
    sections = {"Legal Position": "The bank is also liable under Section 66C of the IT Act."}
    findings = citation_audit.verify_citations(sections, ["RBI Limited Liability of Customers circular"])
    assert len(findings) == 1


def test_real_templates_produce_findings_consistent_with_their_own_hints() -> None:
    cheque = get_template("cheque_bounce_notice")
    rent_agreement = get_template("rent_agreement")
    assert cheque is not None and rent_agreement is not None

    correct = {"Legal Position": "Proceedings under Section 138 read with Section 142 will follow."}
    assert citation_audit.verify_citations(correct, cheque.applicable_sections_hint) == []

    wrong = {"Legal Position": "Proceedings under Section 420 will follow."}
    assert len(citation_audit.verify_citations(wrong, cheque.applicable_sections_hint)) == 1

    # rent_agreement.yaml's own numbered clauses must never be flagged.
    own_clauses = {"Terms and Conditions": "Section 3 fixes the security deposit refund timeline."}
    assert citation_audit.verify_citations(own_clauses, rent_agreement.applicable_sections_hint) == []
