import re
from typing import TYPE_CHECKING, Any, ClassVar

from app.schemas.analysis import EntityResponse

if TYPE_CHECKING:
    from app.llm.base import LLMProvider


class EntityExtractor:
    # Keyword portions are scoped case-insensitive with inline `(?i:...)` so a
    # global IGNORECASE flag can't also loosen the value-capturing groups below
    # ([0-9A-Z]...) — that previously let ordinary lowercase words right after
    # "sec"/"case"/"fir"/"complaint" (e.g. "security", "case very serious")
    # get misread as section/case numbers.
    patterns: ClassVar[dict[str, str]] = {
        "case_number": r"\b(?i:case|fir|complaint)\s*(?i:no\.?|number)?\s*[:\-]?\s*([A-Z0-9/\-]{3,})",
        "section_number": r"\b(?i:section|sec\.?|s\.)\s*([0-9A-Z][0-9A-Z\-()/.]*)",
        "amount": r"(?i:rs\.?|inr|₹)\s*([0-9,]+(?:\.\d{1,2})?)",
        "date": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})\b",
        "vehicle_number": r"\b(?i:[A-Z]{2}\s?\d{1,2}\s?[A-Z]{1,3}\s?\d{3,4})\b",
    }

    act_terms: ClassVar[list[str]] = [
        "Bharatiya Nyaya Sanhita",
        "Bharatiya Nagarik Suraksha Sanhita",
        "Bharatiya Sakshya Adhiniyam",
        "Consumer Protection Act",
        "Information Technology Act",
        "Indian Contract Act",
        "Companies Act",
        "Transfer of Property Act",
        "Negotiable Instruments Act",
        "Hindu Marriage Act",
        "Domestic Violence Act",
        "RTI Act",
    ]

    async def extract_structured(
        self,
        text: str,
        language: str | None = None,
        *,
        conversation_facts: dict[str, Any] | None = None,
        llm: "LLMProvider | None" = None,
        page_of: dict[str, int] | None = None,
    ) -> EntityResponse:
        """`extract()` plus the Phase 3 structured layer.

        The flat `entities` map is produced exactly as before, so every
        existing caller sees an unchanged response. The added fields carry
        what the flat map cannot: which text supported each value, which page
        it was on, whether a rule or a model produced it, and what still needs
        asking before any of it is relied on.

        Kept as a separate method rather than folded into `extract()` because
        `extract()` is on the hot path of every chat turn and this does more
        work -- including, optionally, a model call.
        """
        from app.entity_extraction import hybrid
        from app.entity_extraction.timeline import extract_timeline

        flat = await self.extract(text, language)
        structured = await hybrid.extract(
            text, conversation_facts=conversation_facts, llm=llm, page_of=page_of
        )
        timeline = extract_timeline(text, page_of=page_of)
        return flat.model_copy(
            update={
                "structured": structured.entities,
                "unresolved_roles": structured.unresolved_roles,
                "conflicts": structured.conflicts,
                "timeline": timeline.events,
                "undated_events": timeline.undated_events,
                "date_contradictions": timeline.contradictions,
                "questions": [*structured.questions, *timeline.questions],
            }
        )

    async def extract(self, text: str, language: str | None = None) -> EntityResponse:
        entities: dict[str, Any] = {}
        for name, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[name] = list(dict.fromkeys(match.strip() for match in matches))
        acts = [act for act in self.act_terms if act.lower() in text.lower()]
        if acts:
            entities["act_name"] = acts
        state_match = re.search(r"\b(Maharashtra|Delhi|Karnataka|Tamil Nadu|Telangana|Gujarat|Punjab|Kerala|West Bengal|Odisha|Assam)\b", text, re.IGNORECASE)
        if state_match:
            entities["state"] = state_match.group(1)
        confidence = 0.55 + min(0.4, len(entities) * 0.08)
        return EntityResponse(entities=entities, confidence=confidence)
