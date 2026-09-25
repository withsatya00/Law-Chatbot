from __future__ import annotations

import asyncio

import pytest


def test_state_coverage_route_is_registered_and_admin_gated() -> None:
    from app.api.admin_phase3 import router
    from app.api.deps import require_admin

    assert any(dependency.dependency is require_admin for dependency in router.dependencies)
    routes = {route.path for route in router.routes}
    assert "/admin/phase3/state-coverage" in routes
    assert "/admin/phase3/kb-rollout/readiness" in routes


def test_state_coverage_composes_matrix_automation_and_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.api.admin_phase3 as module

    fake_coverage_report = {"jurisdictions": [], "complete": False, "manual_access_exceptions": []}
    fake_automation_report = {"enabled": True, "exception_queue": 0, "dead_letter": {"manual_access_required": 1}}
    monkeypatch.setattr(
        module, "KnowledgeBaseCoverageService",
        lambda: type("S", (), {"coverage": staticmethod(lambda: asyncio.sleep(0, result=fake_coverage_report))})(),
    )

    class _FakeAutomation:
        async def coverage(self):
            return fake_automation_report

        async def list_jobs(self, status=None, limit=100):
            assert status == "manual_access_required"
            return [{"_id": "job-1", "candidate": {"title": "x", "url": "https://x.gov.in/x.pdf"}}]

    monkeypatch.setattr(module, "KnowledgeBaseAutomationService", _FakeAutomation)
    fake_cursor_items = [{"_id": "evt-1", "event_type": "kb_adapter_layout_changed", "created_at": "now"}]

    class _FakeCursor:
        def sort(self, *a, **kw):
            return self

        def limit(self, *a, **kw):
            return self

        def __aiter__(self):
            async def gen():
                for item in fake_cursor_items:
                    yield item
            return gen()

    fake_events_collection = type("C", (), {"find": lambda self, *a, **kw: _FakeCursor()})()
    monkeypatch.setattr(
        type(module.mongodb), "db",
        property(lambda self: {"operational_events": fake_events_collection}),
    )

    result = asyncio.run(module.state_coverage())

    assert result["coverage"] == fake_coverage_report
    assert result["automation"] == fake_automation_report
    assert result["manual_access_required"][0]["_id"] == "job-1"
    assert result["layout_change_alerts"][0]["_id"] == "evt-1"
