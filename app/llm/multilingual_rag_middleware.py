from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

MODEL = "claude-opus-5"

NO_VERIFIED_CONTEXT_EN = "No verified document related to this question is currently available in the Knowledge Base."

_LOCALIZED_FALLBACK_CACHE: dict[str, str] = {
    "english": NO_VERIFIED_CONTEXT_EN,
    "hindi": "इस प्रश्न से संबंधित कोई सत्यापित दस्तावेज़ वर्तमान Knowledge Base में उपलब्ध नहीं है।",
}

SYSTEM_PROMPT = (
    "You are a strict Legal RAG Assistant. You operate ONLY on the <context> provided in the user message "
    "-- never on outside or pre-trained knowledge of Indian law.\n"
    "Rule 1: If the <context> genuinely answers the question, synthesize it in {language} and cite the source "
    "(Act and section) naturally.\n"
    "Rule 2: If the <context> is empty, or on inspection does not actually answer this question, output exactly "
    "and only this line, with nothing before or after it: {fallback_line}\n"
    "Rule 3: NEVER use general knowledge to fill a gap in the <context>. NEVER add a hedging disclaimer in place "
    "of Rule 2's line. NEVER answer in any language other than {language}, even if the context is in English."
)


@dataclass
class TranslatedQuery:
    english_query: str
    detected_language: str


class MultilingualRAGMiddleware:
    def __init__(self, client: anthropic.AsyncAnthropic | None = None, model: str = MODEL) -> None:
        self.client = client or anthropic.AsyncAnthropic()
        self.model = model

    async def translate_query(self, user_query: str) -> TranslatedQuery:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=(
                "Detect the language of the user's legal query and translate it to English, preserving exact "
                "legal nuance (section numbers, Act names, legal terms, named entities). Respond with ONLY "
                'compact JSON, no prose: {"detected_language": "<lowercase language name>", "english_query": '
                '"<translation>"}. If the query is already English, detected_language is "english" and '
                "english_query is the query unchanged."
            ),
            messages=[{"role": "user", "content": user_query}],
            output_config={"effort": "low"},
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        try:
            data = json.loads(text)
            return TranslatedQuery(english_query=data["english_query"], detected_language=data["detected_language"].strip().lower())
        except (json.JSONDecodeError, KeyError, AttributeError):
            return TranslatedQuery(english_query=user_query, detected_language="english")

    async def _localized_fallback(self, language: str) -> str:
        if language in _LOCALIZED_FALLBACK_CACHE:
            return _LOCALIZED_FALLBACK_CACHE[language]
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=f"Translate the following sentence to {language} exactly. Output ONLY the translation -- no quotes, no explanation, no additions.",
            messages=[{"role": "user", "content": NO_VERIFIED_CONTEXT_EN}],
            output_config={"effort": "low"},
        )
        translated = "".join(block.text for block in response.content if block.type == "text").strip()
        _LOCALIZED_FALLBACK_CACHE[language] = translated
        return translated

    async def generate_multilingual_response(self, user_query: str, detected_language: str, english_context: str) -> str:
        fallback_line = await self._localized_fallback(detected_language)
        if not english_context.strip():
            return fallback_line
        system_prompt = SYSTEM_PROMPT.format(language=detected_language, fallback_line=fallback_line)
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"<context>\n{english_context}\n</context>\n\nUser question (in {detected_language}): {user_query}",
            }],
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        )
        if response.stop_reason == "refusal":
            return fallback_line
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return text or fallback_line
