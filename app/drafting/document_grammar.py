"""Deterministic document grammar, AST and validation primitives.

This module is deliberately independent from the LLM and the exporters.  It
is the structural boundary between both: templates resolve to a grammar,
generated content is compiled into a :class:`DocumentAST`, and renderers only
receive the validated, ordered result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from app.drafting.templates.base import DraftTemplateDefinition, structure_sections_for


class HeadingMode(StrEnum):
    VISIBLE = "visible"
    HIDDEN = "hidden"
    INLINE = "inline"
    GENERATED = "generated"
    CONTINUATION_ONLY = "continuation_only"


class ComponentType(StrEnum):
    COURT_HEADER = "court_header"
    COURT_CAPTION = "court_caption"
    PARTY_BLOCK = "party_block"
    ADVOCATE_HEADER = "advocate_header"
    DATE_BLOCK = "date_block"
    ADDRESS_BLOCK = "address_block"
    SUBJECT_BLOCK = "subject_block"
    SALUTATION = "salutation"
    REFERENCE_BLOCK = "reference_block"
    TITLE = "title"
    TEXT = "text"
    NUMBERED_PARAGRAPHS = "numbered_paragraphs"
    NUMBERED_LIST = "numbered_list"
    BULLET_LIST = "bullet_list"
    RECITAL = "recital"
    OPERATIVE_CLAUSE = "operative_clause"
    LEGAL_GROUND = "legal_ground"
    LEGAL_TEST = "legal_test"
    CAUSE_OF_ACTION = "cause_of_action"
    JURISDICTION = "jurisdiction"
    LIMITATION = "limitation"
    VALUATION = "valuation"
    COURT_FEE = "court_fee"
    COURT_PRAYER = "court_prayer"
    RELIEF = "relief"
    DEMAND = "demand"
    ADMINISTRATIVE_REQUEST = "administrative_request"
    POLICE_REQUEST = "police_request"
    DECLARATION = "declaration"
    VERIFICATION = "verification"
    AFFIDAVIT = "affidavit"
    SIGNATURE = "signature"
    WITNESS_BLOCK = "witness_block"
    TABLE = "table"
    PROPERTY_SCHEDULE = "property_schedule"
    ANNEXURE_LIST = "annexure_list"
    NOTE = "note"
    PAGE_BREAK = "page_break"


Rule = Mapping[str, Any] | bool | None


class RuleEvaluationError(ValueError):
    """Raised for unsupported or malformed restricted rules."""


class ConditionEvaluator:
    """Small, side-effect-free JSON-Logic style evaluator.

    It intentionally supports only an allow-list.  Rules are configuration,
    never Python expressions, so a lawyer/admin supplied grammar cannot run
    code on the server.
    """

    _OPS: ClassVar[set[str]] = {
        "and", "or", "not", "equals", "contains", "exists", "count",
        "greater_than", "less_than", "in", "any", "all", "var",
    }

    def evaluate(self, rule: Rule, context: Mapping[str, Any]) -> bool:
        if rule is None:
            return False
        if isinstance(rule, bool):
            return rule
        return bool(self._value(rule, context))

    def _value(self, expression: Any, context: Mapping[str, Any]) -> Any:
        if not isinstance(expression, Mapping):
            return expression
        if len(expression) != 1:
            raise RuleEvaluationError("A rule node must contain exactly one operator")
        op, raw = next(iter(expression.items()))
        if op not in self._OPS:
            raise RuleEvaluationError(f"Unsupported rule operator: {op}")
        if op == "var":
            return self._lookup(context, str(raw))
        if op == "exists":
            value = self._value(raw, context)
            return value is not None and value != "" and value != [] and value != {}
        if op == "not":
            return not bool(self._value(raw, context))
        if op in {"and", "all"}:
            return all(bool(self._value(item, context)) for item in self._items(raw))
        if op in {"or", "any"}:
            return any(bool(self._value(item, context)) for item in self._items(raw))
        if op == "count":
            value = self._value(raw, context)
            return len(value) if value is not None and hasattr(value, "__len__") else 0
        left, right = self._binary(raw, context)
        if op == "equals":
            return left == right
        if op == "contains":
            return left is not None and right in left
        if op == "in":
            return right is not None and left in right
        if op == "greater_than":
            return left > right
        if op == "less_than":
            return left < right
        raise RuleEvaluationError(f"Unsupported rule operator: {op}")

    def _binary(self, raw: Any, context: Mapping[str, Any]) -> tuple[Any, Any]:
        items = self._items(raw)
        if len(items) != 2:
            raise RuleEvaluationError("Binary rule operators require exactly two operands")
        return self._value(items[0], context), self._value(items[1], context)

    @staticmethod
    def _items(raw: Any) -> list[Any]:
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise RuleEvaluationError("Rule operands must be a list")
        return list(raw)

    @staticmethod
    def _lookup(context: Mapping[str, Any], path: str) -> Any:
        current: Any = context
        for part in path.split("."):
            if isinstance(current, Mapping):
                current = current.get(part)
            elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and part.isdigit():
                index = int(part)
                current = current[index] if index < len(current) else None
            else:
                return None
        return current


@dataclass(frozen=True)
class PaginationPolicy:
    keep_together: bool = False
    keep_with_next: bool = False
    widow_control: bool = True
    orphan_control: bool = True
    allow_split: bool = True
    page_break_before: bool = False


@dataclass(frozen=True)
class SectionSpec:
    id: str
    source_heading: str
    component: ComponentType = ComponentType.TEXT
    heading_mode: HeadingMode = HeadingMode.VISIBLE
    required: bool = False
    required_when: Rule = None
    include_when: Rule = None
    forbidden_when: Rule = None
    repeatable: bool = False
    children: tuple[str, ...] = ()
    pagination: PaginationPolicy = field(default_factory=PaginationPolicy)

    def is_required(self, context: Mapping[str, Any], evaluator: ConditionEvaluator) -> bool:
        return self.required or evaluator.evaluate(self.required_when, context)

    def is_forbidden(self, context: Mapping[str, Any], evaluator: ConditionEvaluator) -> bool:
        return evaluator.evaluate(self.forbidden_when, context)

    def is_included(self, context: Mapping[str, Any], evaluator: ConditionEvaluator) -> bool:
        return self.include_when is None or evaluator.evaluate(self.include_when, context)


@dataclass(frozen=True)
class DocumentGrammar:
    document_id: str
    document_family: str
    document_type: str
    structure: tuple[SectionSpec, ...]
    version: str = "1.0"
    document_variant: str = "default"
    proceeding_type: str = ""
    proceeding_stage: str = ""
    forum_type: str = ""
    jurisdiction_profile: str = "generic_india"
    formatting_profile: str = "generic_legal_document"
    governing_law: tuple[str, ...] = ()
    legal_provisions: tuple[str, ...] = ()
    source_template_id: str | None = None
    legacy_adapter: bool = False
    active: bool = True

    def active_sections(self, context: Mapping[str, Any]) -> tuple[SectionSpec, ...]:
        evaluator = ConditionEvaluator()
        return tuple(
            section for section in self.structure
            if section.is_included(context, evaluator) and not section.is_forbidden(context, evaluator)
        )

    def generation_headings(self, context: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        return tuple(section.source_heading for section in self.active_sections(context or {}))


@dataclass(frozen=True)
class DocumentBlock:
    section_id: str
    component: ComponentType
    content: str
    source_heading: str
    heading_mode: HeadingMode
    pagination: PaginationPolicy
    children: tuple[DocumentBlock, ...] = ()


@dataclass(frozen=True)
class DocumentAST:
    document_id: str
    grammar_version: str
    language: str
    script: str
    blocks: tuple[DocumentBlock, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_legacy_sections(self) -> dict[str, str]:
        """Compatibility bridge for the existing PDF/DOCX/TXT exporters."""
        return {block.source_heading: block.content for block in self.blocks if block.content.strip()}


@dataclass(frozen=True)
class ValidationIssue:
    error_code: str
    severity: str
    location: str
    message: str
    suggested_fix: str

    def as_dict(self) -> dict[str, str]:
        return {
            "error_code": self.error_code,
            "severity": self.severity,
            "location": self.location,
            "message": self.message,
            "suggested_fix": self.suggested_fix,
        }


class DocumentGrammarValidator:
    def validate(
        self, grammar: DocumentGrammar, sections: Mapping[str, str], context: Mapping[str, Any] | None = None,
    ) -> list[ValidationIssue]:
        context = context or {}
        evaluator = ConditionEvaluator()
        issues: list[ValidationIssue] = []
        known = {spec.source_heading for spec in grammar.structure}
        for spec in grammar.structure:
            content = str(sections.get(spec.source_heading, "")).strip()
            if spec.is_forbidden(context, evaluator) and content:
                issues.append(ValidationIssue(
                    "STRUCTURE_FORBIDDEN_SECTION", "error", spec.id,
                    f"'{spec.source_heading}' is forbidden for this document grammar.",
                    "Remove the section or select the correct document grammar.",
                ))
            elif spec.is_required(context, evaluator) and not content:
                issues.append(ValidationIssue(
                    "STRUCTURE_REQUIRED_SECTION_MISSING", "warning" if grammar.legacy_adapter else "error", spec.id,
                    f"Required section '{spec.source_heading}' is missing.",
                    "Supply the material facts needed for this section.",
                ))
        for heading, content in sections.items():
            if heading not in known and str(content).strip() and heading not in {"Disclaimer", "Annexures", "Enclosures"}:
                issues.append(ValidationIssue(
                    "STRUCTURE_UNKNOWN_SECTION", "warning", heading,
                    f"Section '{heading}' is not declared by grammar '{grammar.document_id}'.",
                    "Declare the section in the grammar or remove it.",
                ))
        return issues


class LegacyTemplateAdapter:
    """Maps every existing YAML template to the new grammar contract."""

    _COMPONENTS: ClassVar[dict[str, ComponentType]] = {
        "Recipient": ComponentType.ADDRESS_BLOCK,
        "To": ComponentType.ADDRESS_BLOCK,
        "Subject": ComponentType.SUBJECT_BLOCK,
        "Facts of the Case": ComponentType.NUMBERED_PARAGRAPHS,
        "Statements": ComponentType.NUMBERED_PARAGRAPHS,
        "Prayer": ComponentType.COURT_PRAYER,
        "Request": ComponentType.ADMINISTRATIVE_REQUEST,
        "Verification": ComponentType.VERIFICATION,
        "Signature": ComponentType.SIGNATURE,
        "Signature Block": ComponentType.SIGNATURE,
        "Signatures": ComponentType.SIGNATURE,
        "Annexures": ComponentType.ANNEXURE_LIST,
        "Enclosures": ComponentType.ANNEXURE_LIST,
    }

    @classmethod
    def adapt(cls, template: DraftTemplateDefinition) -> DocumentGrammar:
        family = template.document_family or template.category.lower()
        specs_list: list[SectionSpec] = []
        for heading in structure_sections_for(template.category):
            component = cls._COMPONENTS.get(heading, ComponentType.TEXT)
            heading_mode = HeadingMode.VISIBLE
            if template.category == "Notice":
                if heading == "Prayer":
                    component = ComponentType.DEMAND
                    heading_mode = HeadingMode.HIDDEN
                elif heading in {"Introduction", "Facts of the Case", "Legal Position", "Consequences"}:
                    heading_mode = HeadingMode.HIDDEN
            elif template.category == "Complaint" and heading == "Prayer":
                component = ComponentType.COURT_PRAYER
            elif template.category == "Application" and heading == "Request":
                component = ComponentType.ADMINISTRATIVE_REQUEST
            specs_list.append(SectionSpec(
                id=heading.lower().replace(" / ", "_").replace(" ", "_"),
                source_heading=heading,
                component=component,
                heading_mode=heading_mode,
                required=True,
                pagination=PaginationPolicy(
                    keep_together=heading in {"Verification", "Signature", "Signature Block", "Signatures"},
                    keep_with_next=heading not in {"Signature", "Signature Block", "Signatures"},
                    allow_split=heading not in {"Verification", "Signature", "Signature Block", "Signatures"},
                ),
            ))
        specs = tuple(specs_list)
        return DocumentGrammar(
            document_id=template.draft_id,
            document_family=family,
            document_type=template.category.lower(),
            structure=specs,
            governing_law=tuple(template.applicable_acts_hint),
            legal_provisions=tuple(template.applicable_sections_hint),
            source_template_id=template.draft_id,
            legacy_adapter=True,
            version=template.template_version,
        )


class DocumentSchemaRegistry:
    def __init__(self) -> None:
        self._grammars: dict[str, DocumentGrammar] = {}

    def register(self, grammar: DocumentGrammar, *, replace: bool = False) -> None:
        if grammar.document_id in self._grammars and not replace:
            raise ValueError(f"Duplicate document grammar: {grammar.document_id}")
        self._grammars[grammar.document_id] = grammar

    def get(self, document_id: str) -> DocumentGrammar | None:
        return self._grammars.get(document_id)

    def for_template(self, template: DraftTemplateDefinition) -> DocumentGrammar:
        grammar = self.get(template.draft_id)
        if grammar is None:
            grammar = LegacyTemplateAdapter.adapt(template)
            self.register(grammar)
        return grammar

    def list(self) -> list[DocumentGrammar]:
        return sorted((g for g in self._grammars.values() if g.active), key=lambda g: g.document_id)

    def compile_ast(
        self,
        grammar: DocumentGrammar,
        sections: Mapping[str, str],
        *,
        language: str = "english",
        script: str = "auto",
        context: Mapping[str, Any] | None = None,
    ) -> DocumentAST:
        blocks = tuple(
            DocumentBlock(
                section_id=spec.id,
                component=spec.component,
                content=str(sections.get(spec.source_heading, "")),
                source_heading=spec.source_heading,
                heading_mode=spec.heading_mode,
                pagination=spec.pagination,
            )
            for spec in grammar.active_sections(context or {})
            if str(sections.get(spec.source_heading, "")).strip()
        )
        # Preserve legacy optional blocks without letting them change order.
        declared = {block.source_heading for block in blocks}
        optional = tuple(
            DocumentBlock(
                section_id=heading.lower(), component=ComponentType.ANNEXURE_LIST,
                content=str(sections[heading]), source_heading=heading,
                heading_mode=HeadingMode.VISIBLE,
                pagination=PaginationPolicy(page_break_before=True),
            )
            for heading in ("Annexures", "Enclosures")
            if heading in sections and heading not in declared and str(sections[heading]).strip()
        )
        return DocumentAST(
            document_id=grammar.document_id,
            grammar_version=grammar.version,
            language=language,
            script=script,
            blocks=blocks + optional,
            metadata={"document_family": grammar.document_family},
        )


document_schema_registry = DocumentSchemaRegistry()
