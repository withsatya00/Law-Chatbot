import re
from typing import ClassVar


class DraftSafetyGuard:
    """Best-effort keyword guard against requests to fabricate legal content.

    This is a lightweight regex scan, not a certified compliance control — it
    catches explicit requests to forge, backdate, or fabricate content before
    any LLM call, mirroring `app.utils.prompt_security.PromptInjectionScanner`.
    The primary safety mechanism is the drafting prompt instructing the model
    to formalize only facts explicitly given (see `legal_drafting_prompt.md`
    and each template's `drafting_notes`).
    """

    risky_patterns: ClassVar[list[str]] = [
        r"\bforge(d|ry)?\b",
        r"\bbackdate\b",
        r"\bfake (signature|id|identity|evidence|document|affidavit)\b",
        r"\bfabricat(e|ed|ion)\b",
        r"\bfalse affidavit\b",
        r"\bimpersonat(e|ion)\b",
        r"\bmake up (a |the )?(evidence|witness|document)\b",
        r"\bcreate (a |an )?fake\b",
    ]

    def scan(self, text: str) -> tuple[bool, list[str]]:
        findings = [pattern for pattern in self.risky_patterns if re.search(pattern, text, flags=re.IGNORECASE)]
        return bool(findings), findings
