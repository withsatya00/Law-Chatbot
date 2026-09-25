"""Download official legal PDFs and ingest new bytes through the shared-KB pipeline.

Downloading an official file proves provenance and identity only.  It does not
prove territorial applicability, commencement, or transition rules, so newly
ingested sources deliberately remain ``needs_review`` until a law-specific
machine-verification policy can prove those separate facts.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
from dataclasses import dataclass
from typing import Any

import structlog
from fastapi import UploadFile

from app.core.config import settings
from app.database.mongodb import mongodb
from app.models.collections import EMBEDDINGS_METADATA, UPLOADED_DOCUMENTS
from app.rag.kb_jurisdiction import (
    PROVENANCE_AUTOMATED_OFFICIAL,
    document_metadata_fields,
    normalize_jurisdiction,
)
from app.services.kb_automation import OfficialDownloader
from app.services.kb_ingestion_service import KnowledgeBaseIngestionService
from app.services.kb_machine_verification import normalized_text

log = structlog.get_logger(__name__)

# Bare Acts (especially scanned Gazette PDFs) routinely exceed the 4 MiB cap
# `fetch_official_snapshot` enforces for its own purpose -- bounding a
# snapshot that gets stored as a BSON binary field for change-monitoring
# diffs. This sync only ever writes the bytes to disk
# (`settings.knowledge_base_dir`), so it uses the automation pipeline's own
# `OfficialDownloader` instead, which is sized for full documents.
_document_downloader = OfficialDownloader(max_bytes=100 * 1024 * 1024)


async def fetch_official_document(url: str) -> tuple[bytes, str]:
    body, content_type, _final_url = await _document_downloader.fetch(url)
    return body, content_type


@dataclass(frozen=True)
class OfficialSource:
    key: str
    filename: str
    url: str
    document_key: str
    version_label: str
    source_type: str
    identity_tokens: tuple[str, ...]
    # Other `source_document` names the SAME official text is indexed under,
    # e.g. an admin upload that kept the Gazette's own file name. `coverage()`
    # looked only at `filename` and so reported a manually-ingested Act (Code on
    # Wages, 2019 is indexed as `2589gi_P65_6.pdf`) as never ingested.
    kb_filenames: tuple[str, ...] = ()
    # Image-only official scan: `normalized_text` reads nothing from it, so the
    # identity check in `sync()` could never pass. Such a source is never
    # auto-fetched; `sync()` reports it as needing the OCR manual-ingestion path
    # (`scripts/ingest_scanned_official_pdf.py`) and a human review afterwards.
    requires_ocr: bool = False


SOURCES = (
    OfficialSource(
        "BNS", "BNS_2023_Official_Gazette.pdf",
        "https://www.mha.gov.in/sites/default/files/2024-04/250883_english_01042024.pdf",
        "bharatiya-nyaya-sanhita-2023", "Act 45 of 2023 (official Gazette text)",
        "bare_act", ("bharatiyanyayasanhita2023", "no45of2023", "25thdecember2023"),
        kb_filenames=("The_Bhara_Tiy_A_Ny_A_Y_A_Sanhita_2023_5.pdf",),
    ),
    OfficialSource(
        "BSA", "BSA_2023_Official_Gazette.pdf",
        "https://www.mha.gov.in/sites/default/files/2024-04/250882_english_01042024_0.pdf",
        "bharatiya-sakshya-adhiniyam-2023", "Act 47 of 2023 (official Gazette text)",
        "bare_act", ("bharatiyasakshyaadhiniyam2023", "no47of2023", "25thdecember2023"),
    ),
    OfficialSource(
        "BNSS", "BNSS_2023_Official_Gazette.pdf",
        "https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf",
        "bharatiya-nagarik-suraksha-sanhita-2023", "Act 46 of 2023 (official Gazette text)",
        "bare_act", ("bharatiyanagariksurakshasanhita2023", "no46of2023", "25thdecember2023"),
        kb_filenames=("Bharatiya_Nagarik_Suraksha_Sanhita_2023_Complete_Act.pdf",),
    ),
    OfficialSource(
        "RTI", "RTI_Act_2005_Official_Amended.pdf",
        "https://rti.dopt.gov.in/Writereaddata/RTI%20Act,%202005%20(Amended)-English%20Version.PDF",
        "right-to-information-act-2005", "Official amended English version",
        "bare_act", ("righttoinformationact2005", "no22of2005", "15thjune2005"),
        kb_filenames=("The_Right_To_Information_Act_2005_2.pdf",),
    ),
    # KNOWN GAP (verified 2026-09-21 against the indexed chunks): this URL is the
    # ORIGINAL 2000 text ("ITbill_2000.pdf") -- it has no IT (Amendment) Act 2008
    # provisions, so ss. 43A, 66C (identity theft), 66D (cheating by personation),
    # 66E, 67A/67B and 69A are absent from the KB. cag.gov.in's copy is the same
    # unamended text. The only official amended text found is MeitY's
    # IT_amendment_act2008-1_0.pdf (next entry), an image-only scan.
    OfficialSource(
        "IT_ACT", "IT_Act_2000_India_Code.pdf",
        "https://www.meity.gov.in/static/uploads/2024/03/ITbill_2000.pdf",
        "information-technology-act-2000", "MeitY official text (indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("informationtechnologyact2000", "no21of2000"),
    ),
    # SCANNED, NOT AUTO-SYNCED. HTTP 200 application/pdf, 20 pages, 0 extractable
    # characters (verified 2026-09-21), so `sync()` would hold it on the identity
    # check every run. `requires_ocr=True` makes `sync()` skip it with an explicit
    # reason. Ingest it only via `scripts/ingest_scanned_official_pdf.py`, which
    # OCRs it, records per-page OCR confidence and provenance, and leaves it
    # `needs_review`: a human must check the OCR text against the page images
    # before anything here is marked verified. It is the AMENDING Act (Act 10 of
    # 2009), not a consolidated text of the IT Act.
    OfficialSource(
        "IT_ACT_2008_AMENDMENT", "IT_Amendment_Act_2008_MeitY_Scanned.pdf",
        "https://www.meity.gov.in/static/uploads/2024/03/IT_amendment_act2008-1_0.pdf",
        "information-technology-amendment-act-2008",
        "MeitY official scan of the Information Technology (Amendment) Act, 2008 (image-only; OCR text, "
        "amending Act rather than a consolidated IT Act)",
        "bare_act", ("informationtechnologyamendmentact2008",), requires_ocr=True,
    ),
    OfficialSource(
        "HINDU_MARRIAGE_ACT", "Hindu_Marriage_Act_1955_India_Code.pdf",
        "https://igrsup.gov.in/prernadoc/Adhiniyam/pdfMarriage/hinduMarriageActEnglish/hinduMarriageAct.pdf",
        "hindu-marriage-act-1955", "UP IGRS (Inspector General of Registration & Stamps) official mirror",
        "bare_act", ("hindumarriageact1955", "25of1955"),
    ),
    # Neither MP nor UP has a separate state Police Act -- both operate under
    # this shared central Act, extended to each State by its own adoption
    # law. One copy covers both; `applicability`/`applicable_state_codes` are
    # set to `specific_states: [MP, UP]` at the review step (never
    # `all_india` -- rule 1 in `app/rag/kb_jurisdiction.py`), same as any
    # other State-specific KB entry.
    OfficialSource(
        "POLICE_ACT_1861_MP_UP", "Police_Act_1861_MP_UP_Official.pdf",
        "https://www.mppolice.gov.in/sites/default/files/Act1861_English.pdf",
        "police-act-1861", "Act V of 1861, as extended to Madhya Pradesh and Uttar Pradesh",
        "bare_act", ("policeact1861", "actvof1861"),
    ),
    OfficialSource(
        "NI_ACT", "Negotiable_Instruments_Act_1881_India_Code.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Negotiable%20Instruments%20Act,%201881.pdf",
        "negotiable-instruments-act-1881", "Telangana High Court official mirror (indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("negotiableinstrumentsact1881", "26of1881", "firstdayofmarch1882"),
        kb_filenames=("Negotiable_Instruments_Act_1881_Complete_Act.pdf",),
    ),
    OfficialSource(
        "CONSUMER_PROTECTION_ACT", "Consumer_Protection_Act_2019_India_Code.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/16939/1/a2019-35.pdf",
        "consumer-protection-act-2019", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("consumerprotectionact2019", "35of2019"),
        kb_filenames=("The_Consumer_Protection_Act_2019_5.pdf",),
    ),
    OfficialSource(
        "CONTRACT_ACT", "Indian_Contract_Act_1872_CAG_Official.pdf",
        "https://www.cag.gov.in/uploads/media/Indian-Contract-Act-1872-20200816140128.pdf",
        "indian-contract-act-1872", "CAG compiled text (as amended)",
        "bare_act", ("indiancontractact1872", "no9of1872", "25thapril1872"),
    ),
    # --- Remaining acts from the "most important central laws in India" list.
    # indiacode.nic.in migrated to a client-rendered SPA and no longer serves
    # most /bitstream/ paths as raw PDFs, so these were individually
    # re-sourced (mostly the Telangana High Court's official "Central
    # Governmental Acts" mirror at thc.nic.in, else the administering
    # ministry) and each URL was verified live (HTTP 200, application/pdf,
    # no redirect) before being added here.
    OfficialSource(
        "IPC", "IPC_1860_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/11091/1/the_indian_penal_code%2C_1860.pdf",
        "indian-penal-code-1860", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("indianpenalcode1860", "45of1860"),
    ),
    OfficialSource(
        "CRPC", "CRPC_1974_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Code%20of%20Criminal%20Procedure,%201973.pdf",
        "code-of-criminal-procedure-1974", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("codeofcriminalprocedure1973", "2of1974"),
    ),
    OfficialSource(
        "EVIDENCE_ACT", "EVIDENCE_ACT_1872_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/4218/1/THE-INDIAN-EVIDENCE-ACT-1872.pdf",
        "indian-evidence-act-1872", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("indianevidenceact1872", "1of1872"),
    ),
    OfficialSource(
        "NDPS", "NDPS_1985_Official.pdf",
        "https://dor.gov.in/files/acts_files/Narcotic-Drugs-and-Psychotropic-Substances-Act-1985_0.pdf",
        "narcotic-drugs-and-psychotropic-substances-act-1985", "Department of Revenue official text",
        "bare_act", ("narcoticdrugsandpsychotropicsubstancesact1985", "61of1985"),
    ),
    OfficialSource(
        "PMLA", "PMLA_2003_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Prevention%20of%20Money-Laundering%20Act,%202002..pdf",
        "prevention-of-money-laundering-act-2003", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("preventionofmoneylaunderingact2002", "no15of2003"),
    ),
    OfficialSource(
        "POCSO", "POCSO_2012_Official.pdf",
        "https://bhubaneswarcuttackpolice.gov.in/wp-content/uploads/2020/08/POCSO-ACT.pdf",
        "protection-of-children-from-sexual-offences-act-2012", "Odisha Police (Bhubaneswar-Cuttack) official mirror",
        "bare_act", ("protectionofchildrenfromsexualoffencesact2012", "32of2012"),
    ),
    OfficialSource(
        "POSH_ACT", "POSH_ACT_2013_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/2104/1/A2013-14.pdf",
        "sexual-harassment-of-women-at-workplace-prevention-prohibition-and-redressal-act-2013", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("sexualharassmentofwomenatworkplacepreventionprohibitionandredressalact2013", "14of2013"),
    ),
    OfficialSource(
        "UAPA", "UAPA_1967_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Unlawful%20Activities%20(Prevention)%20Act,%201967.pdf",
        "unlawful-activities-prevention-act-1967", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("unlawfulactivitiespreventionact1967", "no37of1967"),
    ),
    OfficialSource(
        "DOWRY_PROHIBITION", "DOWRY_PROHIBITION_1961_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/5402/1/a1961-28.pdf",
        "dowry-prohibition-act-1961", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("dowryprohibitionact1961", "28of1961"),
    ),
    OfficialSource(
        "ARMS_ACT", "ARMS_ACT_1959_Official.pdf",
        "https://megpolice.gov.in/sites/default/files/arms-act-1959.pdf",
        "arms-act-1959", "Meghalaya Police official mirror. OCR quality is noisy but text is genuinely extractable (thousands of real chars, section numbers present).",
        "bare_act", ("thearmsact1959",),
    ),
    OfficialSource(
        "ARMS_AMENDMENT_2019", "ARMS_AMENDMENT_2019_Official.pdf",
        "https://www.mha.gov.in/sites/default/files/ActAndRuleThe%20ArmsAct_17122019.pdf",
        "arms-amendment-act-2019", "Official government PDF. Short amendment act, not the consolidated Arms Act text.",
        "bare_act", ("armsamendmentact2019", "no48of2019"),
    ),
    OfficialSource(
        "PREVENTION_OF_CORRUPTION", "PREVENTION_OF_CORRUPTION_1988_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/9317/1/corruptiona1988-49.pdf",
        "prevention-of-corruption-act-1988", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("preventionofcorruptionact1988", "49of1988"),
    ),
    OfficialSource(
        "PREVENTION_OF_CORRUPTION_AMENDMENT_2018", "PREVENTION_OF_CORRUPTION_AMENDMENT_2018_Official.pdf",
        "https://cgca.gov.in/pdf/PC_Act_Amendment_2018.pdf",
        "prevention-of-corruption-amendment-act-2018", "Official government PDF. Short amendment act, not the consolidated Prevention of Corruption Act text.",
        "bare_act", ("preventionofcorruptionamendmentact2018", "no16of2018"),
    ),
    OfficialSource(
        "CPC", "CPC_1908_Official.pdf",
        "https://sclsc.gov.in/theme/front/pdf/ACTS%20FINAL/THE%20CODE%20OF%20CIVIL%20PROCEDURE,%201908.pdf",
        "code-of-civil-procedure-1908", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("codeofcivilprocedure1908", "no5of1908"),
    ),
    OfficialSource(
        "TRANSFER_OF_PROPERTY", "TRANSFER_OF_PROPERTY_1882_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/14037/1/transfer_of_property_act8_(1).pdf",
        "transfer-of-property-act-1882", "India Code consolidated text (newer /indiacode/bitstream/ path; previous thc.nic.in URL served a mislabeled JPEG)",
        "bare_act", ("transferofpropertyact1882", "4of1882"),
    ),
    OfficialSource(
        "SPECIFIC_RELIEF_ACT_1963", "SPECIFIC_RELIEF_ACT_1963_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/1583/7/A1963-47.pdf",
        "specific-relief-act-1963", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("specificreliefact1963", "no47of1963"),
    ),
    OfficialSource(
        "LIMITATION_ACT_1963", "LIMITATION_ACT_1963_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/1565/5/A1963-36.pdf",
        "limitation-act-1963", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("limitationact1963", "no36of1963"),
    ),
    OfficialSource(
        "SALE_OF_GOODS_ACT_1930", "SALE_OF_GOODS_ACT_1930_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/2390/1/193003.pdf",
        "sale-of-goods-act-1930", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("saleofgoodsact1930", "no3of1930"),
    ),
    OfficialSource(
        "INDIAN_STAMP_ACT_1899", "INDIAN_STAMP_ACT_1899_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/5746/1/indianstampactenglish_1899_searchable.pdf",
        "indian-stamp-act-1899", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("indianstampact1899", "no2of1899"),
    ),
    OfficialSource(
        "REGISTRATION_ACT_1908", "REGISTRATION_ACT_1908_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/2190/5/A1908-16.pdf",
        "registration-act-1908", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("registrationact1908", "no16of1908"),
    ),
    OfficialSource(
        "RFCTLARR_ACT_2013", "RFCTLARR_ACT_2013_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/19895/1/the_right_to_fair_compensation_and_transparency_in_land_acquisition,_rehabilitation_and_resettlement_act,_2013..pdf",
        "right-to-fair-compensation-and-transparency-in-land-acquisition-rehabilitation-and-resettlement-act-2013", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("righttofaircompensationandtransparencyinlandacquisitionrehabilitationandresettlementact2013", "30of2013"),
    ),
    OfficialSource(
        "HINDU_SUCCESSION_ACT_1956", "HINDU_SUCCESSION_ACT_1956_Official.pdf",
        "https://sclsc.gov.in/theme/front/pdf/ACTS%20FINAL/THE%20HINDU%20SUCCESSION%20ACT,%201956.pdf",
        "hindu-succession-act-1956", "Supreme Court Legal Services Committee official mirror",
        "bare_act", ("hindusuccessionact1956", "30of1956"),
    ),
    OfficialSource(
        "HINDU_MINORITY_GUARDIANSHIP_ACT_1956", "HINDU_MINORITY_GUARDIANSHIP_ACT_1956_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/1649/1/195632.pdf",
        "hindu-minority-and-guardianship-act-1956", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("hinduminorityandguardianshipact1956", "no32of1956"),
    ),
    OfficialSource(
        "SPECIAL_MARRIAGE_ACT_1954", "SPECIAL_MARRIAGE_ACT_1954_Official.pdf",
        "https://sclsc.gov.in/theme/front/pdf/ACTS%20FINAL/THE%20SPECIAL%20MARRIAGE%20ACT,%201954.pdf",
        "special-marriage-act-1954", "Supreme Court Legal Services Committee official mirror",
        "bare_act", ("specialmarriageact1954", "43of1954"),
    ),
    OfficialSource(
        "MUSLIM_PERSONAL_LAW_SHARIAT_ACT_1937", "MUSLIM_PERSONAL_LAW_SHARIAT_ACT_1937_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/2303/1/A1937-26.pdf",
        "muslim-personal-law-shariat-application-act-1937", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("muslimpersonallawshariatapplicationact1937", "no26of1937"),
    ),
    OfficialSource(
        "DISSOLUTION_OF_MUSLIM_MARRIAGES_ACT_1939", "DISSOLUTION_OF_MUSLIM_MARRIAGES_ACT_1939_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/2404/1/193908.pdf",
        "dissolution-of-muslim-marriages-act-1939", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("dissolutionofmuslimmarriagesact1939", "no8of1939"),
    ),
    OfficialSource(
        "MUSLIM_WOMEN_PROTECTION_RIGHTS_DIVORCE_ACT_1986", "MUSLIM_WOMEN_PROTECTION_RIGHTS_DIVORCE_ACT_1986_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/1873/1/198625.pdf",
        "muslim-women-protection-of-rights-on-divorce-act-1986", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("muslimwomenprotectionofrightsondivorceact1986", "no25of1986"),
    ),
    OfficialSource(
        "MUSLIM_WOMEN_PROTECTION_RIGHTS_MARRIAGE_ACT_2019", "MUSLIM_WOMEN_PROTECTION_RIGHTS_MARRIAGE_ACT_2019_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/11564/1/a2019-20.pdf",
        "muslim-women-protection-of-rights-on-marriage-act-2019", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("muslimwomenprotectionofrightsonmarriageact2019", "no20of2019"),
    ),
    OfficialSource(
        "INDIAN_CHRISTIAN_MARRIAGE_ACT_1872", "INDIAN_CHRISTIAN_MARRIAGE_ACT_1872_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/22106/1/the_indian_christian_marriage_act,_1872.pdf",
        "indian-christian-marriage-act-1872", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("indianchristianmarriageact1872", "15of1872"),
    ),
    OfficialSource(
        "INDIAN_DIVORCE_ACT_1869", "INDIAN_DIVORCE_ACT_1869_Official.pdf",
        "https://wcdc.bihar.gov.in/Document/Acts/THEINDIANDIVORCEACT,1869.pdf",
        "indian-divorce-act-1869", "Bihar Women & Child Development Corporation official mirror (ijtr.nic.in has a persistent TLS cert error)",
        "bare_act", ("indiandivorceact1869", "4of1869"),
    ),
    OfficialSource(
        "PARSI_MARRIAGE_DIVORCE_ACT_1936", "PARSI_MARRIAGE_DIVORCE_ACT_1936_Official.pdf",
        "https://www.spniwcd.wcd.gov.in/uploads/pdf/1714715192.pdf",
        "parsi-marriage-and-divorce-act-1936", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("parsimarriageanddivorceact1936", "no3of1936"),
    ),
    OfficialSource(
        "GUARDIANS_AND_WARDS_ACT_1890", "GUARDIANS_AND_WARDS_ACT_1890_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Guardians%20and%20Wards%20Act,%201890.pdf",
        "guardians-and-wards-act-1890", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("guardiansandwardsact1890", "no8of1890"),
    ),
    OfficialSource(
        "HINDU_ADOPTIONS_MAINTENANCE_ACT_1956", "HINDU_ADOPTIONS_MAINTENANCE_ACT_1956_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Hindu%20Adoptions%20and%20Maintenance%20Act,%201956.pdf",
        "hindu-adoptions-and-maintenance-act-1956", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("hinduadoptionsandmaintenanceact1956", "no78of1956"),
    ),
    OfficialSource(
        "PROHIBITION_OF_CHILD_MARRIAGE_ACT_2006", "PROHIBITION_OF_CHILD_MARRIAGE_ACT_2006_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/2055/4/a2007-06.pdf",
        "prohibition-of-child-marriage-act-2007", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("prohibitionofchildmarriageact2006", "6of2007"),
    ),
    OfficialSource(
        "COMPANIES_ACT_2013", "COMPANIES_ACT_2013_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Companies%20Act,%202013.pdf",
        "companies-act-2013", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("companiesact2013", "no18of2013"),
    ),
    OfficialSource(
        "INSOLVENCY_AND_BANKRUPTCY_CODE_2016", "INSOLVENCY_AND_BANKRUPTCY_CODE_2016_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Insolvency%20and%20Bankruptcy%20Code,%202016.pdf",
        "insolvency-and-bankruptcy-code-2016", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("insolvencyandbankruptcycode2016", "no31of2016"),
    ),
    OfficialSource(
        "LIMITED_LIABILITY_PARTNERSHIP_ACT_2008", "LIMITED_LIABILITY_PARTNERSHIP_ACT_2008_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Limited%20Liability%20Partnership%20Act,%202008..pdf",
        "limited-liability-partnership-act-2009", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("limitedliabilitypartnershipact2008", "6of2009"),
    ),
    OfficialSource(
        "SEBI_ACT_1992", "SEBI_ACT_1992_Official.pdf",
        "https://www.sebi.gov.in/sebi_data/attachdocs/1456380272563.pdf",
        "securities-and-exchange-board-of-india-act-1992", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("securitiesandexchangeboardofindiaact1992", "15of1992"),
    ),
    OfficialSource(
        "COMMERCIAL_COURTS_ACT_2015", "COMMERCIAL_COURTS_ACT_2015_Official.pdf",
        "https://cdnbbsr.s3waas.gov.in/s32e45f93088c7db59767efef516b306aa/uploads/2025/04/202504091204980039.pdf",
        "commercial-courts-act-2016", "Ministry of Law and Justice (Legislative Department) official typeset reprint",
        "bare_act", ("commercialcourtsact2015",),
    ),
    OfficialSource(
        "COMPETITION_ACT_2002", "COMPETITION_ACT_2002_Official.pdf",
        "https://www.cci.gov.in/images/legalframeworkact/en/the-competition-act-20021652103427.pdf",
        "competition-act-2003", "Competition Commission of India official text",
        "bare_act", ("competitionact2002", "12of2003"),
    ),
    OfficialSource(
        "FOREIGN_EXCHANGE_MANAGEMENT_ACT_1999", "FOREIGN_EXCHANGE_MANAGEMENT_ACT_1999_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/1988/1/a199942.pdf",
        "foreign-exchange-management-act-1999", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("foreignexchangemanagementact1999", "42of1999"),
    ),
    OfficialSource(
        "BUREAU_OF_INDIAN_STANDARDS_ACT_2016", "BUREAU_OF_INDIAN_STANDARDS_ACT_2016_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/2157/3/A2016-11.pdf",
        "bureau-of-indian-standards-act-2016", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("bureauofindianstandardsact2016", "11of2016"),
    ),
    OfficialSource(
        "ARBITRATION_AND_CONCILIATION_ACT_1996", "ARBITRATION_AND_CONCILIATION_ACT_1996_Official.pdf",
        "https://sclsc.gov.in/theme/front/pdf/ACTS%20FINAL/THE%20ARBITRATION%20AND%20CONCILIATION%20ACT,%201996.pdf",
        "arbitration-and-conciliation-act-1996", "Supreme Court Legal Services Committee official mirror",
        "bare_act", ("arbitrationandconciliationact1996", "26of1996"),
    ),
    OfficialSource(
        "MEDIATION_ACT_2023", "MEDIATION_ACT_2023_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/19637/1/aA2023-32.pdf",
        "mediation-act-2023", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("mediationact2023", "32of2023"),
    ),
    OfficialSource(
        "LEGAL_SERVICES_AUTHORITIES_ACT_1987", "LEGAL_SERVICES_AUTHORITIES_ACT_1987_Official.pdf",
        "https://cdnbbsr.s3waas.gov.in/s32e45f93088c7db59767efef516b306aa/uploads/2025/04/202504081796627129.pdf",
        "legal-services-authorities-act-1987", "NALSA-linked Ministry of Law and Justice official mirror",
        "bare_act", ("legalservicesauthoritiesact1987", "39of1987"),
    ),
    OfficialSource(
        "SARFAESI_ACT_2002", "SARFAESI_ACT_2002_Official.pdf",
        "https://cdn.s3waas.gov.in/s3d840cc5d906c3e9c84374c8919d2074e/uploads/2018/07/2018070569.pdf",
        "securitisation-and-reconstruction-of-financial-assets-and-enforcement-of-security-interest-act-2002", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("securitisationandreconstructionoffinancialassetsandenforcementofsecurityinterestact2002", "no54of2002"),
    ),
    OfficialSource(
        "RDB_ACT_1993", "RDB_ACT_1993_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/1775/1/A1993-51.pdf",
        "recovery-of-debts-and-bankruptcy-act-1993", "India Code consolidated text (newer /indiacode/bitstream/ path). Originally titled Recovery of Debts Due to Banks and Financial Institutions Act, 1993; renamed by 2016 amendment.",
        "bare_act", ("recoveryofdebtsandbankruptcyact1993", "51of1993"),
    ),
    OfficialSource(
        "BANKING_REGULATION_ACT_1949", "BANKING_REGULATION_ACT_1949_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/1885/1/aa1949-10.pdf",
        "banking-regulation-act-1949", "India Code consolidated text (newer /indiacode/bitstream/ path; ifsca.gov.in kept timing out)",
        "bare_act", ("bankingregulationact1949", "10of1949"),
    ),
    OfficialSource(
        "RBI_ACT_1934", "RBI_ACT_1934_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Revenue%20bank%20of%20India%20Act,%201934.pdf",
        "reserve-bank-of-india-act-1934", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("reservebankofindiaact1934", "no2of1934"),
    ),
    OfficialSource(
        "FACTORING_REGULATION_ACT_2011", "FACTORING_REGULATION_ACT_2011_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Factoring%20Regulation%20Act,%202011..pdf",
        "factoring-regulation-act-2012", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("factoringregulationact2011", "no12of2012"),
    ),
    OfficialSource(
        "FUGITIVE_ECONOMIC_OFFENDERS_ACT_2018", "FUGITIVE_ECONOMIC_OFFENDERS_ACT_2018_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Fugitive%20Economic%20Offenders%20Act,%202018.pdf",
        "fugitive-economic-offenders-act-2018", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("fugitiveeconomicoffendersact2018", "no17of2018"),
    ),
    OfficialSource(
        "FOREIGN_TRADE_DR_ACT_1992", "FOREIGN_TRADE_DR_ACT_1992_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Foreign%20Trade%20(Development%20and%20Regulation)%20Act,%201992.pdf",
        "foreign-trade-development-and-regulation-act-1992", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("foreigntradedevelopmentandregulationact1992", "no22of1992"),
    ),
    OfficialSource(
        "TRADE_MARKS_ACT_1999", "TRADE_MARKS_ACT_1999_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/1993/1/a199947.pdf",
        "trade-marks-act-1999", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("trademarksact1999", "47of1999"),
    ),
    OfficialSource(
        "COPYRIGHT_ACT_1957", "COPYRIGHT_ACT_1957_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/1367/1/A195714.pdf",
        "copyright-act-1957", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("copyrightact1957", "14of1957"),
    ),
    OfficialSource(
        "PATENTS_ACT_1970", "PATENTS_ACT_1970_Official.pdf",
        "https://ipindia.gov.in/frontend/pdf/patents/1_113_1_The_Patents_Act__1970___incorporating_all_amendments_till_1-08-2024.pdf",
        "patents-act-1970", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("patentsact1970", "no39of1970"),
    ),
    OfficialSource(
        "DESIGNS_ACT_2000", "DESIGNS_ACT_2000_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/1917/1/A2000-16.pdf",
        "designs-act-2000", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("designsact2000", "16of2000"),
    ),
    OfficialSource(
        "GI_ACT_1999", "GI_ACT_1999_Official.pdf",
        "https://ipindia.gov.in/storage/uploads/docs-operator/5720e089-4967-408c-bc76-031d2a7fd72c.pdf",
        "geographical-indications-of-goods-registration-and-protection-act-1999", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("geographicalindicationsofgoodsregistrationandprotectionact1999", "no48of1999"),
    ),
    OfficialSource(
        "CGST_ACT_2017", "CGST_ACT_2017_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Central%20Goods%20and%20Services%20Tax%20Act,2017.pdf",
        "central-goods-and-services-tax-act-2017", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("centralgoodsandservicestaxact2017", "12of2017"),
    ),
    OfficialSource(
        "IGST_ACT_2017", "IGST_ACT_2017_Official.pdf",
        "https://www.indiacode.nic.in/indiacode/bitstream/123456789/2251/1/A201713.pdf",
        "integrated-goods-and-services-tax-act-2017", "India Code consolidated text (newer /indiacode/bitstream/ path)",
        "bare_act", ("integratedgoodsandservicestaxact2017", "13of2017"),
    ),
    OfficialSource(
        "INCOME_TAX_ACT_1961", "INCOME_TAX_ACT_1961_Official.pdf",
        "https://thc.nic.in/Central%20Governmental%20Acts/Income%20Tax%20Act%201961_.pdf",
        "income-tax-act-1961", "Official government PDF. Reported repealed/replaced by the Income-tax Act, 2025 effective 2026-04-01; ingested as historical text, not current law.",
        "bare_act", ("incometaxact1961", "no43of1961"),
    ),
    OfficialSource(
        "RERA_2016", "RERA_2016_Official.pdf",
        "https://rera.mohua.gov.in/images/pdf_folder/Real_Estate_Act_2016(2).pdf",
        "real-estate-regulation-and-development-act-2016", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("realestateregulationanddevelopmentact2016", "no16of2016"),
    ),
    OfficialSource(
        "CODE_ON_WAGES_2019", "CODE_ON_WAGES_2019_Official.pdf",
        "https://www.labour.gov.in/static/uploads/2025/06/c328da14bbb15fc4ad571dc33e7a4ab3.pdf",
        "code-on-wages-2019", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("codeonwages2019", "no29of2019"),
        # Indexed 2026-08-27 by an admin upload under the Gazette file name, human-verified
        # 2026-09-18; the registry filename below was never used, which is why an
        # earlier audit read this Act as "registered but never ingested".
        kb_filenames=("2589gi_P65_6.pdf",),
    ),
    OfficialSource(
        "INDUSTRIAL_RELATIONS_CODE_2020", "INDUSTRIAL_RELATIONS_CODE_2020_Official.pdf",
        "https://www.labour.gov.in/static/uploads/2025/07/682a44b5426bff2c1f4943ee1b2fd566.pdf",
        "industrial-relations-code-2020", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("industrialrelationscode2020", "no35of2020"),
    ),
    OfficialSource(
        "DPDP_ACT_2023", "DPDP_ACT_2023_Official.pdf",
        "https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf",
        "digital-personal-data-protection-act-2023", "Official government PDF (best available direct source; indiacode.nic.in bitstream unavailable post-migration)",
        "bare_act", ("digitalpersonaldataprotectionact2023", "no22of2023"),
    ),
    # --- Cyber security specific instruments (IT Act 2000 and DPDP Act 2023
    # are already covered above).
    OfficialSource(
        "IT_INTERMEDIARY_RULES_2021", "IT_INTERMEDIARY_RULES_2021_Official.pdf",
        "https://www.meity.gov.in/static/uploads/2024/02/Information-Technology-Intermediary-Guidelines-and-Digital-Media-Ethics-Code-Rules-2021-updated-06.04.2023-.pdf",
        "it-intermediary-guidelines-digital-media-ethics-code-rules-2021", "MeitY official text, updated as on 06.04.2023",
        "rules", ("theinformationtechnologyintermediaryguidelinesanddigitalmediaethicscoderules2021",),
    ),
    OfficialSource(
        "NATIONAL_CYBER_SECURITY_POLICY_2013", "NATIONAL_CYBER_SECURITY_POLICY_2013_Official.pdf",
        "https://www.meity.gov.in/static/uploads/2024/02/National_cyber_security_policy-2013_0.pdf",
        "national-cyber-security-policy-2013", "MeitY official text",
        "unknown", ("nationalcybersecuritypolicy2013",),
    ),
    OfficialSource(
        "IT_CERTIFYING_AUTHORITIES_RULES_2000", "IT_CERTIFYING_AUTHORITIES_RULES_2000_Official.pdf",
        "https://www.meity.gov.in/static/uploads/2024/03/IT-Act-Rules_2000_0.pdf",
        "it-certifying-authorities-rules-2000", "MeitY official text (a combined rules bundle under the IT Act, 2000, including these rules)",
        "rules", ("informationtechnologycertifyingauthoritiesrules2000",),
    ),
)


class OfficialSourceSyncService:
    def __init__(self, fetcher: Any = None, ingestion: Any = None) -> None:
        self.fetcher = fetcher or fetch_official_document
        self.ingestion = ingestion or KnowledgeBaseIngestionService()

    async def run(self) -> dict[str, Any]:
        results = [await self.sync(source) for source in SOURCES]
        return {
            "downloaded_and_indexed": sum(item["status"] == "indexed_needs_review" for item in results),
            "already_present": sum(item["status"] == "already_present" for item in results),
            "held": sum(item["status"] == "held" for item in results),
            "manual_ocr_required": sum(item["status"] == "manual_ocr_required" for item in results),
            "results": results,
        }

    async def coverage(self) -> dict[str, Any]:
        """Report acquisition/index/publication separately; never imply corpus completeness."""
        rows = []
        for source in SOURCES:
            names = [source.filename, *source.kb_filenames]
            # Match on the registry document_key as well as file name: a source
            # ingested by hand keeps whatever name it was uploaded under.
            document = await mongodb.db[UPLOADED_DOCUMENTS].find_one({"$or": [
                {"filename": {"$in": names}}, {"metadata.document_key": source.document_key},
            ]})
            chunk_match = {"$or": [
                {"metadata.source_document": {"$in": names}}, {"metadata.document_key": source.document_key},
            ]}
            chunks = await mongodb.db[EMBEDDINGS_METADATA].count_documents(chunk_match)
            approved = await mongodb.db[EMBEDDINGS_METADATA].count_documents(
                {**chunk_match, "metadata.review_status": "approved"},
            )
            rows.append({
                "law": source.key, "official_url": source.url, "filename": source.filename,
                "downloaded": any((settings.knowledge_base_dir / name).is_file() for name in names),
                "indexed": bool(document), "chunks": chunks, "approved_chunks": approved,
                "review_status": ((document or {}).get("metadata") or {}).get("review_status"),
                "requires_ocr": source.requires_ocr,
            })
        return {
            "scope": "configured priority sources only; this is not all-India legal coverage",
            "sources": rows,
            "next_jurisdictions": ["MP", "UP"],
        }

    async def sync(self, source: OfficialSource) -> dict[str, Any]:
        result: dict[str, Any] = {"law": source.key, "url": source.url, "status": "held"}
        if source.requires_ocr:
            # Not fetched at all: downloading an image-only scan every cycle can
            # never satisfy the text identity check below.
            result.update(
                status="manual_ocr_required",
                reason="Scanned image-only official PDF; run scripts/ingest_scanned_official_pdf.py "
                "and have a human reviewer verify the OCR text before it is approved.",
            )
            return result
        try:
            body, content_type = await self.fetcher(source.url)
            if content_type != "application/pdf" or not body.startswith(b"%PDF"):
                raise ValueError("Official endpoint did not return a PDF.")
            text = normalized_text(body)
            checks = [token in text for token in source.identity_tokens]
            if not all(checks):
                raise ValueError("Official PDF identity checks failed.")
            official_hash = hashlib.sha256(body).hexdigest()
            canonical_path = settings.knowledge_base_dir / source.filename
            if canonical_path.is_file() and hashlib.sha256(canonical_path.read_bytes()).hexdigest() == official_hash:
                result.update(
                    status="already_present", official_sha256=official_hash,
                    generated_filename=source.filename,
                )
                return result
            metadata = document_metadata_fields(normalize_jurisdiction({
                "document_key": source.document_key,
                "issuing_level": "central",
                "applicability": "unknown",
                "jurisdiction_source_type": source.source_type,
                "source_url": source.url,
                "version_label": source.version_label,
                "verification_status": "unverified",
                "metadata_provenance": PROVENANCE_AUTOMATED_OFFICIAL,
                "machine_verification": {
                    "official_url": source.url,
                    "official_sha256": official_hash,
                    "identity_checks": checks,
                    "exact_byte_match": True,
                    "policy_version": "official-source-sync-v1",
                },
            }, provenance=PROVENANCE_AUTOMATED_OFFICIAL))
            upload = UploadFile(file=io.BytesIO(body), filename=source.filename)
            response = await self.ingestion.ingest(upload, jurisdiction_metadata=metadata)
            status = "indexed_needs_review" if response.status == "indexed" else response.status
            if status == "duplicate":
                status = "already_present"
            result.update(
                status=status, official_sha256=official_hash,
                generated_filename=response.generated_filename, document_id=response.document_id,
                chunks_indexed=response.chunks_indexed, review_status=response.review_status,
                reason=(response.review_reasons or None),
            )
        except Exception as exc:  # noqa: BLE001 - every source fails closed and the batch continues
            detail = str(exc).strip() or "No diagnostic message was supplied."
            result["reason"] = f"{type(exc).__name__}: {detail}"
        return result


class OfficialSourceSyncScheduler:
    """Runs `OfficialSourceSyncService.run()` periodically so the registry
    never again goes stale for lack of anyone remembering to invoke
    `scripts/sync_official_kb_sources.py` -- confirmed live 2026-09-22: it had
    not run in 4 days, silently, because nothing was checking it. Mirrors
    `GapAutoFetchScheduler`/`KnowledgeBaseAutomationScheduler` in
    `app/services/kb_gap_autofetch.py` / `kb_automation.py`.
    """

    def __init__(self, service: OfficialSourceSyncService | None = None) -> None:
        self.service = service
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if settings.official_source_sync_enabled and (self.task is None or self.task.done()):
            self.service = self.service or OfficialSourceSyncService()
            self.task = asyncio.create_task(self._loop())

    async def close(self) -> None:
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def _loop(self) -> None:
        if self.service is None:
            raise RuntimeError("Official source sync scheduler started without a service.")
        while True:
            try:
                result = await self.service.run()
                log.info(
                    "official_source_sync_cycle_complete",
                    downloaded_and_indexed=result.get("downloaded_and_indexed"),
                    already_present=result.get("already_present"), held=result.get("held"),
                    manual_ocr_required=result.get("manual_ocr_required"),
                )
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill the scheduler loop
                log.error("official_source_sync_cycle_failed", error=type(exc).__name__)
            await asyncio.sleep(max(3600, settings.official_source_sync_interval_hours * 3600))


official_source_sync_scheduler = OfficialSourceSyncScheduler()
