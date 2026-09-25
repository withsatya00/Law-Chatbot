"""Native high-value legal document grammars.

These schemas are structural configuration, not prose templates.  They are
available for resolution/validation immediately while existing YAML-backed
generation continues through the compatibility adapter until each workflow's
fact collection is explicitly mapped.
"""

from app.drafting.document_grammar import (
    ComponentType as C,
)
from app.drafting.document_grammar import (
    DocumentGrammar,
    document_schema_registry,
)
from app.drafting.document_grammar import (
    HeadingMode as H,
)
from app.drafting.document_grammar import (
    PaginationPolicy as P,
)
from app.drafting.document_grammar import (
    SectionSpec as S,
)


def _s(
    section_id: str,
    heading: str,
    component: C,
    *,
    required: bool = True,
    mode: H = H.VISIBLE,
    repeatable: bool = False,
    page_break: bool = False,
) -> S:
    protected = component in {C.SIGNATURE, C.VERIFICATION, C.WITNESS_BLOCK}
    return S(
        section_id, heading, component, mode, required=required, repeatable=repeatable,
        pagination=P(
            keep_together=protected, keep_with_next=not protected,
            allow_split=not protected, page_break_before=page_break,
        ),
    )


NATIVE_GRAMMARS = (
    DocumentGrammar("advocate_legal_notice", "advocate_notice", "notice", (
        _s("advocate_header", "Advocate Header", C.ADVOCATE_HEADER, required=False),
        _s("recipient", "Recipient", C.ADDRESS_BLOCK), _s("subject", "Subject", C.SUBJECT_BLOCK),
        _s("facts", "Facts", C.NUMBERED_PARAGRAPHS, mode=H.HIDDEN, repeatable=True),
        _s("legal_position", "Legal Position", C.LEGAL_GROUND, mode=H.HIDDEN, repeatable=True),
        _s("demand", "Demand", C.DEMAND, mode=H.HIDDEN), _s("signature", "Signature", C.SIGNATURE),
    ), formatting_profile="advocate_letter"),
    DocumentGrammar("reply_legal_notice", "reply_notice", "notice", (
        _s("reference", "Reference", C.REFERENCE_BLOCK), _s("recipient", "Recipient", C.ADDRESS_BLOCK),
        _s("preliminary_reply", "Preliminary Reply", C.TEXT, mode=H.HIDDEN),
        _s("paragraph_reply", "Paragraph-wise Reply", C.NUMBERED_PARAGRAPHS, repeatable=True),
        _s("response", "Response", C.DEMAND, mode=H.HIDDEN), _s("signature", "Signature", C.SIGNATURE),
    ), formatting_profile="advocate_letter"),
    DocumentGrammar("police_fir_application", "police_application", "application", (
        _s("authority", "To", C.ADDRESS_BLOCK), _s("subject", "Subject", C.SUBJECT_BLOCK),
        _s("complainant", "Complainant Details", C.PARTY_BLOCK),
        _s("facts", "Facts", C.NUMBERED_PARAGRAPHS, repeatable=True),
        _s("police_request", "Request", C.POLICE_REQUEST), _s("enclosures", "Enclosures", C.ANNEXURE_LIST, required=False),
        _s("signature", "Signature", C.SIGNATURE),
    )),
    DocumentGrammar("administrative_application", "administrative_application", "application", (
        _s("authority", "To", C.ADDRESS_BLOCK), _s("subject", "Subject", C.SUBJECT_BLOCK),
        _s("applicant", "Applicant Details", C.PARTY_BLOCK), _s("request", "Request", C.ADMINISTRATIVE_REQUEST),
        _s("enclosures", "Enclosures", C.ANNEXURE_LIST, required=False), _s("signature", "Signature", C.SIGNATURE),
    )),
    DocumentGrammar("court_application", "court_application", "application", (
        _s("court", "Court", C.COURT_HEADER), _s("case", "Case Reference", C.COURT_CAPTION),
        _s("parties", "Parties", C.PARTY_BLOCK), _s("facts", "Facts", C.NUMBERED_PARAGRAPHS, repeatable=True),
        _s("grounds", "Grounds", C.LEGAL_GROUND, repeatable=True), _s("prayer", "Prayer", C.COURT_PRAYER),
        _s("verification", "Verification", C.VERIFICATION, required=False), _s("signature", "Signature", C.SIGNATURE),
    ), forum_type="court"),
    DocumentGrammar("bail_application", "bail_application", "application", (
        _s("court", "Court", C.COURT_HEADER), _s("case", "FIR / Case Details", C.COURT_CAPTION),
        _s("parties", "Parties", C.PARTY_BLOCK), _s("custody", "Custody and Case Facts", C.NUMBERED_PARAGRAPHS, repeatable=True),
        _s("grounds", "Grounds for Bail", C.LEGAL_GROUND, repeatable=True), _s("prayer", "Prayer", C.COURT_PRAYER),
        _s("signature", "Signature", C.SIGNATURE),
    ), proceeding_type="criminal", proceeding_stage="bail", forum_type="court"),
    DocumentGrammar("civil_interim_injunction_application", "interim_application", "application", (
        _s("court", "Court", C.COURT_HEADER), _s("case", "Case Reference", C.COURT_CAPTION),
        _s("parties", "Parties", C.PARTY_BLOCK), _s("facts", "Facts", C.NUMBERED_PARAGRAPHS, repeatable=True),
        _s("legal_tests", "Legal Tests", C.LEGAL_TEST, repeatable=True), _s("prayer", "Prayer", C.COURT_PRAYER),
        _s("verification", "Verification", C.VERIFICATION), _s("signature", "Signature", C.SIGNATURE),
    ), proceeding_type="civil", proceeding_stage="interim", forum_type="district_court",
       governing_law=("Code of Civil Procedure, 1908",), legal_provisions=("Order XXXIX Rules 1 and 2",)),
    DocumentGrammar("general_affidavit", "affidavit", "affidavit", (
        _s("authority", "Court / Authority", C.COURT_HEADER, required=False),
        _s("deponent", "Deponent Details", C.PARTY_BLOCK), _s("statements", "Statements", C.NUMBERED_PARAGRAPHS, repeatable=True),
        _s("verification", "Verification", C.VERIFICATION), _s("signature", "Deponent Signature", C.SIGNATURE),
    )),
    DocumentGrammar("civil_plaint", "pleading", "plaint", (
        _s("court", "Court", C.COURT_HEADER), _s("parties", "Cause Title", C.PARTY_BLOCK),
        _s("facts", "Facts", C.NUMBERED_PARAGRAPHS, repeatable=True), _s("cause", "Cause of Action", C.CAUSE_OF_ACTION),
        _s("jurisdiction", "Jurisdiction", C.JURISDICTION), _s("limitation", "Limitation", C.LIMITATION),
        _s("valuation", "Valuation and Court Fee", C.VALUATION), _s("prayer", "Prayer", C.COURT_PRAYER),
        _s("verification", "Verification", C.VERIFICATION), _s("signature", "Signature", C.SIGNATURE),
    ), proceeding_type="civil", forum_type="court"),
    DocumentGrammar("civil_written_statement", "written_statement", "written_statement", (
        _s("court", "Court", C.COURT_HEADER), _s("parties", "Cause Title", C.PARTY_BLOCK),
        _s("objections", "Preliminary Objections", C.LEGAL_GROUND, repeatable=True),
        _s("replies", "Reply on Merits", C.NUMBERED_PARAGRAPHS, repeatable=True),
        _s("additional", "Additional Pleas", C.LEGAL_GROUND, required=False, repeatable=True),
        _s("prayer", "Prayer", C.COURT_PRAYER), _s("verification", "Verification", C.VERIFICATION),
        _s("signature", "Signature", C.SIGNATURE),
    ), proceeding_type="civil", forum_type="court"),
    DocumentGrammar("native_consumer_complaint", "consumer_complaint", "complaint", (
        _s("forum", "Commission", C.COURT_HEADER), _s("parties", "Parties", C.PARTY_BLOCK),
        _s("facts", "Facts", C.NUMBERED_PARAGRAPHS, repeatable=True), _s("deficiency", "Defect / Deficiency", C.LEGAL_GROUND),
        _s("jurisdiction", "Jurisdiction", C.JURISDICTION), _s("limitation", "Limitation", C.LIMITATION),
        _s("relief", "Relief Sought", C.RELIEF), _s("verification", "Verification", C.VERIFICATION),
        _s("annexures", "Annexures", C.ANNEXURE_LIST, required=False, page_break=True),
        _s("signature", "Signature", C.SIGNATURE),
    ), forum_type="consumer_commission"),
    DocumentGrammar("agreement_or_deed", "agreement", "agreement", (
        _s("title", "Title", C.TITLE), _s("parties", "Parties", C.PARTY_BLOCK),
        _s("recitals", "Recitals", C.RECITAL, repeatable=True),
        _s("clauses", "Terms and Conditions", C.OPERATIVE_CLAUSE, repeatable=True),
        _s("schedule", "Schedule", C.PROPERTY_SCHEDULE, required=False, page_break=True),
        _s("signatures", "Signatures", C.SIGNATURE), _s("witnesses", "Witnesses", C.WITNESS_BLOCK),
    )),
)


def register_native_grammars() -> None:
    for grammar in NATIVE_GRAMMARS:
        if document_schema_registry.get(grammar.document_id) is None:
            document_schema_registry.register(grammar)


register_native_grammars()

