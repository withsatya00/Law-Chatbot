from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.schemas.law_monitoring import CatalogueSourceRequest
from app.services.kb_source_onboarding import SourceOnboardingService, adapter_from_record


def request(**changes):
    values = {
        "name": "kerala_law_catalogue",
        "authority": "Government of Kerala Law Department",
        "url": "https://law.kerala.gov.in/acts",
        "jurisdiction_code": "KL",
        "applicable_state_codes": [],
        "document_type": "bare_act",
        "include_pattern": r"act|\.pdf",
        "resolve_pdf_links": False,
        "authority_confirmed": True,
    }
    return CatalogueSourceRequest(**{**values, **changes})


def test_catalogue_source_requires_official_url_and_confirmation() -> None:
    with pytest.raises(ValidationError):
        request(url="https://example.com/acts")
    with pytest.raises(ValidationError, match="Confirm"):
        request(authority_confirmed=False)


def test_record_builds_jurisdiction_aware_adapter() -> None:
    record = request(applicable_state_codes=["TN"]).model_dump(exclude={"authority_confirmed"})
    adapter = adapter_from_record(record, AsyncMock())
    assert adapter.name == "kerala_law_catalogue"
    assert adapter.config.jurisdiction_code == "KL"
    assert adapter.config.applicable_state_codes == ("TN",)
    assert adapter.config.expected_domain_suffixes == ("law.kerala.gov.in",)


def test_activation_is_fail_closed_until_probe_passes() -> None:
    async def run() -> None:
        sources = AsyncMock()
        events = AsyncMock()
        db = {"kb_source_adapters": sources, "operational_events": events}
        service = SourceOnboardingService(db=db)

        sources.update_one.return_value = SimpleNamespace(modified_count=0)
        with pytest.raises(ValueError, match="pass"):
            await service.activate("kerala_law_catalogue", "admin")

        sources.update_one.return_value = SimpleNamespace(modified_count=1)
        result = await service.activate("kerala_law_catalogue", "admin")
        assert result["status"] == "active"
        events.insert_one.assert_awaited_once()

    asyncio.run(run())


def test_admin_routes_cover_register_probe_activate_and_list() -> None:
    from app.api.admin_phase3 import router
    routes = {(route.path, next(iter(route.methods))) for route in router.routes}
    paths = {path for path, _ in routes}
    assert "/admin/phase3/kb-sources" in paths
    assert "/admin/phase3/kb-sources/{name}/probe" in paths
    assert "/admin/phase3/kb-sources/{name}/activate" in paths
