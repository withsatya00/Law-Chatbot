import re
from typing import Any, ClassVar


class LegalEntityLinker:
    act_aliases: ClassVar[dict[str, str]] = {
        "it act": "Information Technology Act",
        "information technology act": "Information Technology Act",
        "ni act": "Negotiable Instruments Act",
        "negotiable instruments act": "Negotiable Instruments Act",
        "consumer act": "Consumer Protection Act",
        "bns": "Bharatiya Nyaya Sanhita",
        "bnss": "Bharatiya Nagarik Suraksha Sanhita",
        "bsa": "Bharatiya Sakshya Adhiniyam",
        "hindu marriage act": "Hindu Marriage Act",
        "transfer of property act": "Transfer of Property Act",
    }

    lawyer_aliases: ClassVar[dict[str, str]] = {
        "cyber": "Cyber Lawyer",
        "criminal": "Criminal Lawyer",
        "family": "Family Lawyer",
        "property": "Property Lawyer",
        "consumer": "Consumer Lawyer",
        "labour": "Labour Lawyer",
        "tax": "Tax Lawyer",
        "ip": "IP Lawyer",
    }

    async def link(self, text: str, entities: dict[str, Any] | None = None) -> dict[str, Any]:
        linked = dict(entities or {})
        lowered = text.lower()
        acts = [canonical for alias, canonical in self.act_aliases.items() if alias in lowered]
        if acts:
            linked["normalized_acts"] = sorted(set(acts))
        sections = re.findall(r"\b(?:section|sec\.?|s\.)\s*([0-9A-Z][0-9A-Z()./-]*)", text, re.IGNORECASE)
        if sections:
            linked["normalized_sections"] = [section.upper() for section in sections]
        articles = re.findall(r"\barticle\s+([0-9A-Z][0-9A-Z()./-]*)", text, re.IGNORECASE)
        if articles:
            linked["normalized_articles"] = [article.upper() for article in articles]
        lawyers = [canonical for alias, canonical in self.lawyer_aliases.items() if alias in lowered]
        if lawyers:
            linked["lawyer_categories"] = sorted(set(lawyers))
        return linked
