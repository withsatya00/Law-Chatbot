import re
from dataclasses import dataclass


@dataclass
class LegalChunk:
    text: str
    act: str | None
    chapter: str | None
    section: str | None


class LegalDocumentProcessor:
    ACT_RE = re.compile(
        r"^\s*Act\s*:\s*(.+?)\s*$|^\s*(?:THE\s+)?([A-Z][A-Za-z0-9 ,'\-]{3,80}\bACT)\b[, ]*(?:\d{4})?\s*$",
        re.MULTILINE,
    )
    CHAPTER_RE = re.compile(
        r"^\s*Chapter\s*[:\-]?\s*([IVXLCDM]+|\d+)\s*[:\-]?\s*(.*)$"
        r"|^\s*CHAPTER\s+([IVXLCDM]+|\d+)\s*[:\-]?\s*(.*)$",
        re.MULTILINE,
    )
    SECTION_RE = re.compile(
        r"^\s*(?:Section|Sec\.?|Article|Rule)\s*[:\-]?\s*(\d+[A-Z]{0,2})\s*[:\-]?\s*(.*)$"
        r"|^\s*(\d{1,4}[A-Z]{0,2})\.\s+([A-Z][^\n]{0,140}?)[—–]\s*$",
        re.MULTILINE,
    )
    SENTENCE_RE = re.compile(r"(?<=[.?!])\s+(?=[A-Z0-9(])")

    def __init__(self, min_tokens: int = 512, max_tokens: int = 1024, overlap_ratio: float = 0.12) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio

    def _tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _sentences(self, text: str) -> list[str]:
        return [s.strip() for s in self.SENTENCE_RE.split(text) if s.strip()]

    def _header(self, act: str | None, chapter: str | None, section: str | None) -> str:
        parts = []
        if act:
            parts.append(f"Act: {act}")
        if chapter:
            parts.append(f"Chapter: {chapter}")
        if section:
            parts.append(f"Section: {section}")
        return (", ".join(parts) + " | ") if parts else ""

    def _overlap_sentences(self, sentences: list[str]) -> list[str]:
        target = int(self._tokens(" ".join(sentences)) * self.overlap_ratio)
        kept, tok = [], 0
        for s in reversed(sentences):
            if tok >= target:
                break
            kept.append(s)
            tok += self._tokens(s)
        return list(reversed(kept))

    def process(self, text: str) -> list[LegalChunk]:
        act = chapter = section = None
        buffer: list[str] = []
        chunks: list[LegalChunk] = []

        def flush() -> None:
            nonlocal buffer
            if not buffer:
                return
            body = " ".join(buffer).strip()
            if body:
                chunks.append(LegalChunk(text=f"{self._header(act, chapter, section)}{body}", act=act, chapter=chapter, section=section))
            buffer = self._overlap_sentences(buffer)

        for line in text.splitlines():
            if not line.strip():
                continue
            m = self.ACT_RE.match(line)
            if m:
                flush()
                act = (m.group(1) or m.group(2)).strip()
                continue
            m = self.CHAPTER_RE.match(line)
            if m:
                flush()
                chapter = (m.group(1) or m.group(3)).strip()
                continue
            m = self.SECTION_RE.match(line)
            if m:
                flush()
                section = (m.group(1) or m.group(3)).strip()
                continue
            for sentence in self._sentences(line):
                buffer.append(sentence)
                if self._tokens(" ".join(buffer)) >= self.max_tokens:
                    flush()
        flush()
        return chunks


if __name__ == "__main__":
    dummy_text = """Act: Indian Penal Code (IPC)
Chapter 17: Of Offences Against Property
Section 420: Cheating and dishonestly inducing delivery of property
Whoever cheats and thereby dishonestly induces the person deceived to deliver any property to any person, or to make, alter or destroy the whole or any part of a valuable security, or anything which is signed or sealed, and which is capable of being converted into a valuable security, shall be punished with imprisonment of either description for a term which may extend to seven years, and shall also be liable to fine. """ + (
        "This applies irrespective of the value of the property involved in the transaction. " * 40
    ) + """
Section 421: Dishonest or fraudulent removal or concealment of property
Whoever dishonestly or fraudulently removes, conceals, transfers or delivers to any person any property, shall be punished with imprisonment."""

    processor = LegalDocumentProcessor(min_tokens=200, max_tokens=300, overlap_ratio=0.12)
    for i, chunk in enumerate(processor.process(dummy_text)):
        print(f"--- chunk {i} (act={chunk.act}, chapter={chunk.chapter}, section={chunk.section}) ---")
        print(chunk.text[:200], "...\n")
