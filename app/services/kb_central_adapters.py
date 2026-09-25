"""Official Central-law catalogue adapters.

Portals change markup frequently, so the shared adapter extracts semantic link
records and each named adapter supplies only official URLs and inclusion rules.
An empty/changed page becomes adapter telemetry, never a claim of completeness.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from app.rag.source_confidence import classify_change_type, score_candidate
from app.schemas.law_monitoring import official_url
from app.services.kb_automation import SourceCandidate
from app.services.law_monitoring import fetch_official_snapshot

_DATE = re.compile(r"\b(\d{1,2}[-/.](?:\d{1,2}|[A-Za-z]{3,9})[-/.]\d{4}|\d{4}-\d{2}-\d{2})\b")
_YEAR = re.compile(r"\b(18|19|20)\d{2}\b")
_ACT_NUMBER = re.compile(r"\b(?:Act\s+)?(?:No\.?\s*)?(\d{1,3})\s+of\s+(\d{4})\b", re.IGNORECASE)
# A download/view button's own visible text -- confirmed against 5 real state
# portals (Chhattisgarh: "ENG"/"HIN"; Ladakh: "View"; Manipur: "Download
# 926.24 KB"; Nagaland: "Download"; Odisha: "Download(495.23 KB)") -- carries
# no title information at all, even though it is non-empty, so it must be
# treated the SAME as an icon-only anchor (empty text) rather than allowed to
# win over a real title sitting in a sibling cell. Deliberately narrow: only
# the exact whole-string labels actually observed, so a real Act title that
# merely CONTAINS one of these words (e.g. "Hindi Version of ...") is
# unaffected -- `fullmatch` never fires on a longer string.
_GENERIC_ANCHOR_LABEL = re.compile(r"(?:view|download|eng|hin)(?:\s*[(:]?\s*[\d.]+\s*(?:kb|mb))?[.)]?", re.IGNORECASE)


def _nearby_record_text(anchor: Any) -> str:
    """The text of the anchor's nearest semantic container -- normally its
    `<tr>`/`<li>`/`<article>` ancestor, exactly as before.

    Confirmed live 2026-09-22 against `law.cgstate.gov.in/act-details`: a
    real listing that uses none of those three tags at all -- title and
    download button are separate `<div>` children of a shared `<div
    class="act-card">` container, with the download link one level further
    down inside its own `div.act-btn-grp`, so the closest enclosing div adds
    no text beyond the anchor's own. For exactly this case (no `<tr>`/`<li>`/
    `<article>` ancestor anywhere), walk up a SMALL, bounded number of
    `<div>`/`<span>` ancestors and use the first one whose text is both
    meaningfully longer than the anchor's own (so it actually adds new
    content, not just another wrapper around the same button) and not
    implausibly large (so a big outer container spanning multiple listing
    items is never mistaken for one item's own text). Bounded to 4 levels
    and a container div/span only (stops at the first non-div/span ancestor,
    e.g. a `<section>`) so this cannot walk arbitrarily far up the page.
    """
    ancestor = anchor.find_parent(["tr", "li", "article"])
    if ancestor is not None:
        return " ".join(ancestor.get_text(" ", strip=True).split())
    own_text = str(anchor.get_text(" ", strip=True))
    node = anchor.parent
    for _ in range(4):
        if node is None or node.name not in ("div", "span"):
            break
        candidate_text = " ".join(node.get_text(" ", strip=True).split())
        if len(candidate_text) > len(own_text) + 10 and len(candidate_text) <= 600:
            return candidate_text
        node = node.parent
    return own_text


@dataclass(frozen=True)
class CatalogueConfig:
    name: str
    authority: str
    urls: tuple[str, ...]
    document_type: str
    include_pattern: str
    max_items: int = 100
    resolve_pdf_links: bool = False
    # "IN" for a Central source (the historical default); a real State/UT ISO
    # code (see `app/rag/kb_jurisdiction.py::STATE_UT_CODES`) for a state
    # Gazette/legislature adapter.
    jurisdiction_code: str = "IN"
    # Extra states/UTs this SAME portal binds, beyond `jurisdiction_code` --
    # a High Court routinely covers several States/UTs from one portal.
    applicable_state_codes: tuple[str, ...] = ()
    # Domain suffixes this jurisdiction's official portal is expected to use,
    # for confidence scoring (see `app/rag/source_confidence.py`). Empty means
    # "any .gov.in/.nic.in host", the historical Central-adapter behaviour.
    expected_domain_suffixes: tuple[str, ...] = ()


class OfficialCatalogueAdapter:
    def __init__(self, config: CatalogueConfig, fetcher: Any = None) -> None:
        self.config = config
        self.name = config.name
        self.fetcher = fetcher or fetch_official_snapshot
        self._include = re.compile(config.include_pattern, re.IGNORECASE)

    async def discover(self, checkpoint: str | None) -> tuple[list[SourceCandidate], str | None]:
        candidates: list[SourceCandidate] = []
        catalogue_hash = hashlib.sha256()
        successful_catalogues = 0
        errors: list[str] = []
        page_index = int(checkpoint or 0) % len(self.config.urls) if len(self.config.urls) > 1 else 0
        catalogue_urls = (self.config.urls[page_index],) if len(self.config.urls) > 1 else self.config.urls
        for catalogue_url in catalogue_urls:
            try:
                official_url(catalogue_url)
                body, content_type = await self.fetcher(catalogue_url)
                if content_type != "text/html":
                    raise ValueError(f"{self.name} catalogue must return HTML.")
            except Exception as exc:  # noqa: BLE001 - multi-URL ministry adapter can degrade partially
                errors.append(f"{catalogue_url}: {type(exc).__name__}")
                continue
            successful_catalogues += 1
            catalogue_hash.update(body)
            candidates.extend(self._parse(catalogue_url, body))
            if len(candidates) >= self.config.max_items:
                break
        if successful_catalogues == 0:
            raise ValueError("; ".join(errors) or f"{self.name} has no reachable catalogue.")
        unique = list({item.source_key: item for item in candidates}.values())[:self.config.max_items]
        if self.config.resolve_pdf_links:
            resolved: list[SourceCandidate] = []
            for candidate in unique:
                resolved.extend(await self._resolve_pdf_links(candidate))
            unique = resolved
        next_checkpoint = str((page_index + 1) % len(self.config.urls)) if len(self.config.urls) > 1 else catalogue_hash.hexdigest()
        return unique, next_checkpoint

    async def _resolve_pdf_links(self, candidate: SourceCandidate) -> list[SourceCandidate]:
        """Resolve a catalogue detail page to its actual official PDF files."""
        if urlsplit(candidate.url).path.casefold().endswith(".pdf"):
            return [candidate]
        body, content_type = await self.fetcher(candidate.url)
        if content_type != "text/html":
            return []
        soup = BeautifulSoup(body, "html.parser")
        resolved = []
        for anchor in soup.select("a[href]"):
            href = urljoin(candidate.url, str(anchor.get("href") or ""))
            label = " ".join(anchor.get_text(" ", strip=True).split())
            if not (urlsplit(href).path.casefold().endswith(".pdf")
                    or re.search(r"bitstream|showfile|casepdf", href, re.IGNORECASE)):
                continue
            try:
                official_url(href)
            except ValueError:
                continue
            language = "hindi" if re.search(r"hindi|हिन्दी|हिंदी", f"{label} {href}", re.IGNORECASE) else candidate.language
            external_id = hashlib.sha256(href.encode()).hexdigest()[:24]
            resolved.append(replace(
                candidate, external_id=external_id, url=href, language=language,
                filename=f"{self.name}_{external_id}_{language}.pdf",
            ))
        return resolved

    def _parse(self, catalogue_url: str, body: bytes) -> list[SourceCandidate]:
        soup = BeautifulSoup(body, "html.parser")
        rows: list[SourceCandidate] = []
        for anchor in soup.select("a[href]"):
            href = urljoin(catalogue_url, str(anchor.get("href") or "").strip())
            text = " ".join(anchor.get_text(" ", strip=True).split())
            parent_text = _nearby_record_text(anchor)
            record_text = parent_text or text
            # A real government listing row commonly carries an icon-only
            # download link (`<a href="...pdf"><img alt=""/></a>`) with no
            # visible anchor text at all -- confirmed against the live
            # upvidhai.gov.in Ordinances page, where every one of its 31 real
            # PDF links is exactly this shape. Requiring `text` itself to be
            # non-empty silently discarded every one of them; `record_text`
            # (the surrounding row/cell, which still carries the date/title)
            # is the correct thing to require and match the pattern against.
            if not record_text or not self._include.search(f"{text} {href} {record_text}"):
                continue
            # A non-empty anchor text that is itself just a generic download/
            # view button label (see `_GENERIC_ANCHOR_LABEL` above) carries no
            # title information -- treat it the same as an empty/icon-only
            # anchor and fall back to the row text, instead of letting it win.
            title_text = "" if text and _GENERIC_ANCHOR_LABEL.fullmatch(text) else text
            try:
                official_url(href)
            except ValueError:
                continue
            external_id = hashlib.sha256(href.encode()).hexdigest()[:24]
            date_match = _DATE.search(record_text)
            act_match = _ACT_NUMBER.search(record_text)
            year_match = _YEAR.search(record_text)
            is_pdf = urlsplit(href).path.casefold().endswith(".pdf")
            query_looks_like_record = bool(re.search(r"[?&](?:id|actid|mds|documentid)=", href, re.IGNORECASE))
            if not (is_pdf or date_match or year_match or query_looks_like_record):
                continue
            publication_date = date_match.group(1) if date_match else None
            fingerprint = hashlib.sha256(record_text.encode()).hexdigest()[:12]
            version = f"{publication_date or 'catalogue'}-{fingerprint}"
            language = "hindi" if re.search(r"hindi|हिन्दी|हिंदी", record_text, re.IGNORECASE) else "english"
            suffix = ".pdf" if is_pdf else ".html"
            filename = f"{self.name}_{external_id}_{language}{suffix}"
            identity_text = _DATE.sub("", parent_text).casefold()
            identity_text = re.sub(r"hindi|english|हिन्दी|हिंदी", "", identity_text, flags=re.IGNORECASE)
            identity_text = re.sub(r"\s+", " ", identity_text).strip()
            parent_identity = hashlib.sha256(identity_text.encode()).hexdigest()[:24] if identity_text else None
            confidence = score_candidate(
                url=href, title=title_text or record_text, record_text=record_text, is_pdf=is_pdf,
                has_date=date_match is not None, has_act_number=act_match is not None,
                expected_domain_suffixes=self.config.expected_domain_suffixes,
            )
            rows.append(SourceCandidate(
                # An icon-only anchor -- or a generic download/view label,
                # see above -- has no real title of its own: fall back to the
                # row's text so the candidate never carries an empty or
                # meaningless title.
                adapter=self.name, external_id=external_id, title=(title_text or record_text)[:300], url=href,
                jurisdiction_code=self.config.jurisdiction_code, document_type=self.config.document_type,
                version=version, filename=filename,
                act_number=act_match.group(1) if act_match else None,
                enactment_year=int(act_match.group(2)) if act_match else (
                    int(year_match.group(0)) if year_match else None
                ),
                language=language, issuing_authority=self.config.authority,
                publication_date=publication_date,
                parent_external_id=parent_identity,
                applicable_state_codes=self.config.applicable_state_codes,
                change_type=classify_change_type(record_text),
                confidence_score=confidence,
            ))
        return rows


class IndiaCodeAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        urls = tuple(
            f"https://www.indiacode.nic.in/handle/123456789/1362/browse?type=shorttitle&rpp=25&order=ASC&offset={offset}"
            for offset in range(0, 850, 25)
        )
        super().__init__(CatalogueConfig(
            "india_code", "Legislative Department",
            urls, "bare_act", r"view|act|bitstream|casepdf", 25, True,
        ), fetcher)


class LegislativeDepartmentAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            "legislative_department", "Legislative Department",
            ("https://legislative.gov.in/actsofparliamentfromtheyear/",),
            "bare_act", r"act|ordinance|\.pdf", 100, True,
        ), fetcher)


class EGazetteAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            "egazette", "Department of Publication", ("https://egazette.gov.in/",),
            "notification", r"gazette|notification|extraordinary|\.pdf", 100,
        ), fetcher)


class SupremeCourtAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            "supreme_court", "Supreme Court of India",
            ("https://www.sci.gov.in/landmark-judgment-summaries/",),
            "case_law", r"judgment|insc|\.pdf", 100, True,
        ), fetcher)


class RBIAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            "rbi", "Reserve Bank of India", ("https://www.rbi.org.in/Scripts/NotificationUser.aspx",),
            "circular", r"notification|circular|direction|master", 100,
        ), fetcher)


class SEBIAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            "sebi", "Securities and Exchange Board of India", ("https://www.sebi.gov.in/legal.html",),
            "circular", r"act|rule|regulation|order|circular|notification", 100,
        ), fetcher)


class MCAAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            "mca", "Ministry of Corporate Affairs",
            ("https://www.mca.gov.in/content/mca/global/en/acts-rules.html",),
            "rules", r"act|rule|notification|circular|\.pdf", 100,
        ), fetcher)


class IRDAIAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            "irdai", "Insurance Regulatory and Development Authority of India",
            ("https://irdai.gov.in/regulations",), "regulation",
            r"regulation|circular|order|guideline|\.pdf", 100,
        ), fetcher)


class CBDTAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            "cbdt", "Central Board of Direct Taxes",
            ("https://www.incometax.gov.in/iec/foportal/circulars-notifications",),
            "circular", r"circular|notification|order|rule|\.pdf", 100,
        ), fetcher)


class CBICGSTAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            "cbic_gst", "Central Board of Indirect Taxes and Customs",
            ("https://taxinformation.cbic.gov.in/content-page/explore-notification",),
            "notification", r"gst|notification|circular|instruction|order|\.pdf", 100,
        ), fetcher)


class SelectedMinistriesAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            "selected_ministries", "Government of India",
            (
                "https://www.mha.gov.in/en/commoncontent/acts-rules",
                "https://www.meity.gov.in/content/acts-rules",
                "https://consumeraffairs.nic.in/acts-and-rules",
            ),
            "unknown", r"act|rule|regulation|notification|circular|order|\.pdf", 150,
        ), fetcher)


def central_source_adapters() -> list[OfficialCatalogueAdapter]:
    return [
        IndiaCodeAdapter(), LegislativeDepartmentAdapter(), EGazetteAdapter(), SupremeCourtAdapter(),
        RBIAdapter(), SEBIAdapter(), MCAAdapter(), IRDAIAdapter(), CBDTAdapter(), CBICGSTAdapter(),
        SelectedMinistriesAdapter(),
    ]
