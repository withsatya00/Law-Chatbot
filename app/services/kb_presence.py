"""Is this Act already in the Knowledge Base?

`GapAutoFetchService.record_gap` used to queue whatever Act an LLM named for a
refused question, without asking whether the KB already held it. Every "gap"
in the 2026-09-21 queue for NI Act, Contract Act, RTI Act, Consumer Protection
Act, CPC, BNS, BNSS and the Payment of Wages Act was a false positive of that
kind: the provisions were indexed, the refusal came from a retrieval/bridge
miss on the question's wording, and the queue then tried to download an Act it
already had.

Identifying an indexed Act from chunk metadata alone is unreliable here --
`document_key` is null for several central Acts and `act_name` often holds a
fragment ("For the purposes of this Act"). So presence is decided by two
independent things that must both hold: the requested title matches a known
alias (below, plus every title in the official-source registry), AND a chunk
of one of that law's known KB files/keys really exists in MongoDB. An alias
match with no chunks (a registered-but-never-ingested Act) is NOT present, so a
genuine gap still queues.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.models.collections import EMBEDDINGS_METADATA
from app.services.kb_official_source_sync import SOURCES

STATUS_PRESENT = "present"
# Text that automation cannot fetch because the only official copy is an image
# scan; only the OCR manual-ingestion path (scripts/ingest_scanned_official_pdf.py)
# followed by human review can supply it. Queueing it for auto-fetch would just
# re-download a PDF the identity check can never read.
STATUS_MANUAL_ONLY = "manual_ingestion_required"

_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_SECTION_REF_RE = re.compile(
    r"\b(?:sections?|sec\.?|s\.)\s*\d+[a-z]?(?:\s*(?:,|and|&|to|-)\s*\d+[a-z]?)*", re.IGNORECASE,
)
# A different instrument that merely contains an Act's title.
_DERIVATIVE_MARKERS = ("amendment", "rules", "regulations", "repeal", "ordinance", "bill")


@dataclass(frozen=True)
class KnownKbLaw:
    key: str
    aliases: tuple[str, ...]  # alphanumeric-lowercase title fragments, year excluded
    years: tuple[int, ...] = ()  # accepted years; empty = any
    source_documents: tuple[str, ...] = ()  # KB `metadata.source_document` values
    document_keys: tuple[str, ...] = ()
    status: str = STATUS_PRESENT
    note: str = ""


# Only entries confirmed against live MongoDB on 2026-09-21. Where two copies of
# one Act are indexed, both file names are listed.
KNOWN_KB_LAWS: tuple[KnownKbLaw, ...] = (
    KnownKbLaw(
        "bns", ("bharatiyanyayasanhita", "bharatiyanyayasecondsanhita"), (2023,),
        ("The_Bhara_Tiy_A_Ny_A_Y_A_Sanhita_2023_5.pdf",), ("bharatiya-nyaya-sanhita-2023",),
    ),
    KnownKbLaw(
        "bnss", ("bharatiyanagariksurakshasanhita", "bharatiyanagariksurakshasecondsanhita"), (2023,),
        ("BNSS_2023_Official_Gazette.pdf", "Bharatiya_Nagarik_Suraksha_Sanhita_2023_Complete_Act.pdf"),
        ("bharatiya-nagarik-suraksha-sanhita-2023-dup",),
    ),
    KnownKbLaw(
        "ni-act", ("negotiableinstrumentsact",), (1881,),
        ("Negotiable_Instruments_Act_1881_Complete_Act.pdf", "Negotiable_Instruments_Act_1881_India_Code.pdf"),
    ),
    KnownKbLaw(
        "contract-act", ("indiancontractact",), (1872,), ("Indian_Contract_Act_1872_CAG_Official.pdf",),
    ),
    KnownKbLaw(
        "rti-act", ("righttoinformationact",), (2005,),
        ("RTI_Act_2005_Official_Amended.pdf", "The_Right_To_Information_Act_2005_2.pdf"),
        ("right-to-information-act-2005-dup",),
    ),
    KnownKbLaw(
        "consumer-protection-act", ("consumerprotectionact",), (2019,),
        ("The_Consumer_Protection_Act_2019_5.pdf", "Consumer_Protection_Act_2019_India_Code.pdf"),
        ("consumer-protection-act-2019",),
    ),
    KnownKbLaw(
        "cpc", ("codeofcivilprocedure",), (1908,), ("CPC_1908_Official.pdf",), ("code-of-civil-procedure-1908",),
    ),
    KnownKbLaw(
        "crpc", ("codeofcriminalprocedure",), (1973, 1974), ("CRPC_1974_Official.pdf",),
        ("code-of-criminal-procedure-1974",),
    ),
    KnownKbLaw(
        "income-tax", ("incometaxact",), (1961, 2025),
        ("INCOME_TAX_ACT_1961_Official.pdf", "income_tax_act_2025.pdf"), ("income-tax-act-1961",),
    ),
    KnownKbLaw(
        "payment-of-wages-act", ("paymentofwagesact",), (1936,),
        ("mh_acts_1936.04_payment-of-wages-act-1936.pdf",), ("mh-1936.04-payment-of-wages-act-1936",),
        note="Maharashtra-hosted copy of the 1936 Act; the central Code on Wages, 2019 is indexed separately.",
    ),
    KnownKbLaw(
        "code-on-wages", ("codeonwages",), (2019,), ("2589gi_P65_6.pdf", "CODE_ON_WAGES_2019_Official.pdf"),
        ("code-on-wages-2019",),
    ),
    KnownKbLaw(
        "it-act-2000", ("informationtechnologyact",), (2000,),
        ("IT_Act_2000_India_Code.pdf",), ("information-technology-act-2000",),
        note="Original 2000 text only; amended provisions are tracked under it-act-2008-amendment.",
    ),
    KnownKbLaw(
        "it-act-2008-amendment", ("informationtechnologyamendmentact",), (2008, 2009),
        status=STATUS_MANUAL_ONLY,
        note="Only official copy is an image-only scan; needs OCR manual ingestion and human review.",
    ),
    KnownKbLaw(
        "rbi-limiting-liability-circular",
        ("limitingliabilityofcustomers", "customerprotectionlimitingliability", "limitingliabilityunauthorised"),
        (2017,), ("RBI_Limiting_Liability_Unauthorised_Electronic_Banking_2017.html",),
        ("rbi-limiting-liability-unauthorised-electronic-banking-2017",),
    ),
)


def normalize_act_title(name: str) -> str:
    """Lowercase alphanumerics only, with any trailing "Section 138"-style
    reference removed so "NI Act, 1881, Section 138" and the bare title agree."""
    return re.sub(r"[^a-z0-9]+", "", _SECTION_REF_RE.sub(" ", name.lower()))


@lru_cache(maxsize=1)
def _catalog() -> tuple[KnownKbLaw, ...]:
    """Explicit entries first, then one derived entry per official-source
    registry title (so every registered Act gets alias matching for free)."""
    derived: list[KnownKbLaw] = []
    for source in SOURCES:
        token = source.identity_tokens[0]
        year_match = re.search(r"((?:1[89]|20)\d{2})$", token)
        derived.append(KnownKbLaw(
            key=f"registry-{source.key.lower()}",
            aliases=(token[: year_match.start()] if year_match else token,),
            years=(int(year_match.group(1)),) if year_match else (),
            source_documents=(source.filename, *source.kb_filenames),
            document_keys=(source.document_key,),
        ))
    return (*KNOWN_KB_LAWS, *derived)


def match_known_laws(act_name: str) -> list[KnownKbLaw]:
    normalized = normalize_act_title(act_name)
    if not normalized:
        return []
    years = {int(year) for year in _YEAR_RE.findall(_SECTION_REF_RE.sub(" ", act_name))}
    matches: list[KnownKbLaw] = []
    for law in _catalog():
        for alias in law.aliases:
            if alias not in normalized:
                continue
            if any(marker in normalized and marker not in alias for marker in _DERIVATIVE_MARKERS):
                continue
            if years and law.years and not years & set(law.years):
                continue
            matches.append(law)
            break
    return matches


@dataclass(frozen=True)
class KbPresence:
    law_key: str
    status: str
    source_documents: tuple[str, ...]
    note: str = ""


async def find_existing_kb_law(db: Any, act_name: str) -> KbPresence | None:
    """`KbPresence` if `act_name` is an Act the KB already holds (or one only
    the manual OCR path can supply); `None` when it is a genuine gap."""
    matches = match_known_laws(act_name)
    for law in matches:
        if law.status == STATUS_MANUAL_ONLY:
            return KbPresence(law.key, law.status, (), law.note)
    if not matches:
        return None
    files = sorted({name for law in matches for name in law.source_documents})
    keys = sorted({key for law in matches for key in law.document_keys})
    clauses: list[dict[str, Any]] = []
    if files:
        clauses.append({"metadata.source_document": {"$in": files}})
    if keys:
        clauses.append({"metadata.document_key": {"$in": keys}})
    hit = await db[EMBEDDINGS_METADATA].find_one({"$or": clauses}, {"metadata.source_document": 1})
    if not hit:
        return None
    found = (hit.get("metadata") or {}).get("source_document")
    return KbPresence(matches[0].key, STATUS_PRESENT, (found,) if found else tuple(files), matches[0].note)
