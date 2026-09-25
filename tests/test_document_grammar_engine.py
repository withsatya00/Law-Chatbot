from fastapi.testclient import TestClient

from app.drafting.document_grammar import (
    ComponentType,
    ConditionEvaluator,
    DocumentGrammarValidator,
    HeadingMode,
    document_schema_registry,
)
from app.drafting.registries import formatting_profile_registry, language_registry
from app.drafting.templates import get_template, list_templates
from app.main import app


def test_restricted_condition_engine_supports_nested_rules() -> None:
    context = {"forum_type": "court", "attachments": ["invoice"], "amount": 150000}
    rule = {
        "and": [
            {"equals": [{"var": "forum_type"}, "court"]},
            {"greater_than": [{"count": {"var": "attachments"}}, 0]},
            {"in": [{"var": "amount"}, [50000, 150000]]},
        ]
    }
    assert ConditionEvaluator().evaluate(rule, context) is True


def test_every_legacy_template_adapts_without_losing_heading_order() -> None:
    for template in list_templates():
        grammar = document_schema_registry.for_template(template)
        assert grammar.document_id == template.draft_id
        assert grammar.generation_headings()
        assert grammar.legacy_adapter is True


def test_notice_semantics_are_separate_from_visible_headings() -> None:
    template = get_template("legal_notice")
    assert template is not None
    grammar = document_schema_registry.for_template(template)
    prayer = next(section for section in grammar.structure if section.source_heading == "Prayer")
    assert prayer.component == ComponentType.DEMAND
    assert prayer.heading_mode == HeadingMode.HIDDEN


def test_affidavit_has_no_prayer_component() -> None:
    template = get_template("affidavit")
    assert template is not None
    grammar = document_schema_registry.for_template(template)
    assert ComponentType.COURT_PRAYER not in {section.component for section in grammar.structure}
    assert ComponentType.DEMAND not in {section.component for section in grammar.structure}


def test_ast_is_ordered_and_round_trips_for_legacy_exporters() -> None:
    template = get_template("rti_application")
    assert template is not None
    grammar = document_schema_registry.for_template(template)
    sections = {heading: f"content:{heading}" for heading in grammar.generation_headings()}
    ast = document_schema_registry.compile_ast(grammar, sections, language="hindi", script="Devanagari")
    assert [block.source_heading for block in ast.blocks] == list(grammar.generation_headings())
    assert ast.to_legacy_sections() == sections
    assert DocumentGrammarValidator().validate(grammar, sections) == []


def test_language_registry_covers_english_and_eighth_schedule_languages() -> None:
    assert len(language_registry.list()) == 23
    assert language_registry.get("ur").direction == "rtl"  # type: ignore[union-attr]
    assert "Devanagari" in language_registry.get("ks").script_variants  # type: ignore[union-attr]
    assert formatting_profile_registry.get("delhi_courts_2022") is not None


def test_new_registry_and_validation_endpoints_are_additive() -> None:
    client = TestClient(app)
    grammar = client.get("/document-grammars/legal_notice")
    assert grammar.status_code == 200
    assert grammar.json()["document_family"] == "notice"
    assert any(item["language_code"] == "hi" for item in client.get("/languages").json())

    headings = document_schema_registry.for_template(get_template("legal_notice")).generation_headings()  # type: ignore[arg-type]
    sections = {heading: "present" for heading in headings}
    validated = client.post(
        "/draft/validate", json={"document_id": "legal_notice", "sections": sections, "context": {}}
    )
    assert validated.status_code == 200
    assert validated.json()["status"] == "PASS"

