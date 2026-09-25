import re
from typing import ClassVar


class QueryNormalizer:
    replacements: ClassVar[dict[str, str]] = {
        r"\bfirr?\b": "FIR",
        r"\bfr\b": "FIR",
        r"\bsalery\b": "salary",
        r"\bcyber\s+frud\b": "cyber fraud",
        r"\bonline\s+scam\b": "cyber fraud online fraud",
        r"\bfirst information report\b": "FIR",
        r"\bpolice report\b": "police complaint FIR",
        r"\bcheque bounce\b": "cheque bounce Negotiable Instruments Act section 138",
        r"\brent\b": "rental tenancy landlord tenant",
    }

    def normalize(self, text: str) -> str:
        normalized = text.strip()
        for pattern, replacement in self.replacements.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", normalized)
