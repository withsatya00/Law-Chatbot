import re
from typing import ClassVar


class PromptInjectionScanner:
    risky_patterns: ClassVar[list[str]] = [
        r"ignore (all )?(previous|system|developer) instructions",
        r"reveal (the )?(system prompt|developer message|hidden prompt)",
        r"print (the )?(prompt|instructions)",
        r"act as unrestricted",
        r"jailbreak",
    ]

    def scan(self, text: str) -> tuple[bool, list[str]]:
        findings = [
            pattern
            for pattern in self.risky_patterns
            if re.search(pattern, text, flags=re.IGNORECASE)
        ]
        return bool(findings), findings
