import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings


@dataclass(frozen=True)
class DocumentQualityReport:
    document_hash: str
    is_duplicate_candidate: bool
    is_corrupt: bool
    ocr_quality_score: float
    metadata_quality_score: float
    passed: bool
    issues: list[str]


class DocumentQualityChecker:
    def hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    async def assess(
        self,
        path: Path,
        text: str,
        metadata: dict[str, Any],
        known_hashes: set[str] | None = None,
    ) -> DocumentQualityReport:
        document_hash = self.hash_file(path)
        known_hashes = known_hashes or set()
        issues: list[str] = []
        is_corrupt = not text.strip()
        if is_corrupt:
            issues.append("No extractable text found.")
        duplicate = document_hash in known_hashes
        if duplicate:
            issues.append("Duplicate document hash detected.")
        ocr_quality = self._ocr_quality(text)
        if ocr_quality < settings.min_ocr_quality_score:
            issues.append("OCR/text quality score is below configured threshold.")
        metadata_quality = self._metadata_quality(metadata)
        if metadata_quality < settings.min_metadata_quality_score:
            issues.append("Metadata quality score is below configured threshold.")
        passed = not is_corrupt and not duplicate and ocr_quality >= settings.min_ocr_quality_score
        return DocumentQualityReport(
            document_hash=document_hash,
            is_duplicate_candidate=duplicate,
            is_corrupt=is_corrupt,
            ocr_quality_score=ocr_quality,
            metadata_quality_score=metadata_quality,
            passed=passed,
            issues=issues,
        )

    def _ocr_quality(self, text: str) -> float:
        if not text.strip():
            return 0.0
        chars = len(text)
        alnum = sum(1 for char in text if char.isalnum() or char.isspace())
        broken = len(re.findall(r"[�□]{1,}", text))
        return max(0.0, min(1.0, (alnum / max(chars, 1)) - (broken / max(chars, 1))))

    def _metadata_quality(self, metadata: dict[str, Any]) -> float:
        required = ["source_document", "document_type", "language", "source"]
        present = sum(1 for key in required if metadata.get(key))
        legal_fields = sum(1 for key in ["act_name", "section_number", "chapter", "url", "government_source"] if metadata.get(key))
        return min(1.0, (present / len(required)) * 0.7 + min(0.3, legal_fields * 0.075))
