r"""Official High Court judgment/notification catalogue adapters.

Same shared engine as `kb_central_adapters.py`'s `SupremeCourtAdapter`
(`OfficialCatalogueAdapter`), generalized for a High Court because one High
Court routinely has jurisdiction over several States/UTs from a single portal
(e.g. Punjab & Haryana; Gauhati over multiple North-Eastern states) --
`CatalogueConfig.applicable_state_codes` carries that full list, while
`jurisdiction_code` stays a single representative code for the candidate's
primary `document_metadata_fields` state binding built in
`KnowledgeBaseAutomationService._process_claimed`.

Only listings verified on the Courts' own domains are registered.  Other High
Courts remain explicit gaps in the Phase-4 source registry; no URL is guessed.
"""

from __future__ import annotations

from typing import Any

from app.services.kb_central_adapters import CatalogueConfig, OfficialCatalogueAdapter


class DelhiHighCourtJudgmentsAdapter(OfficialCatalogueAdapter):
    """Latest judgments published by the High Court of Delhi."""

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="delhi_high_court_judgments", authority="High Court of Delhi",
            urls=("https://delhihighcourt.nic.in/web/hi/judgement/fetch-data",),
            document_type="case_law",
            include_pattern=r"judg(?:e)?ment|download|casepdf|\.pdf",
            max_items=100,
            jurisdiction_code="DL",
            expected_domain_suffixes=("delhihighcourt.nic.in",),
        ), fetcher)


class BombayHighCourtJudgmentsAdapter(OfficialCatalogueAdapter):
    """Recent reported orders/judgments from the Bombay High Court."""

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="bombay_high_court_judgments", authority="High Court of Bombay",
            urls=("https://bombayhighcourt.nic.in/recentorderjudgment.php",),
            document_type="case_law",
            include_pattern=r"judgment|order|generatenewauth|\.pdf",
            max_items=100,
            jurisdiction_code="MH",
            applicable_state_codes=("GA", "DH"),
            expected_domain_suffixes=("bombayhighcourt.nic.in",),
        ), fetcher)


def high_court_source_adapters() -> list[OfficialCatalogueAdapter]:
    return [DelhiHighCourtJudgmentsAdapter(), BombayHighCourtJudgmentsAdapter()]
