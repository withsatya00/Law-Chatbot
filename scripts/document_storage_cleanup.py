"""Cleans up the raw upload storage directory (`storage/uploads`).

This is a file-management utility only: it hashes, fingerprints, dedupes,
and renames the *physical* PDF files sitting in the upload directory. It
never touches the RAG pipeline, embeddings, MongoDB, API routes, or UI --
those all key off `document_id`/chunk text already persisted in Mongo, not
off the on-disk filename, so cleaning up this directory is safe to run
independently (confirmed by grepping the app for any other reader of
`settings.upload_storage_dir` -- there is none; the file is written once at
upload time and never read back by path afterwards).

Two-step duplicate detection:
  1. Exact duplicates: SHA-256 of the full file. Requires reading the whole
     file, which is the one case the task spec allows full reads for.
  2. Near duplicates: for files that survive step 1 (unique hash), extract
     only the PDF title/metadata + first page text (capped at 500 words),
     build a normalized fingerprint (lowercase, no spaces/punctuation), and
     compare pairwise with difflib.SequenceMatcher (deterministic, no LLM
     calls). >=95% similarity groups files as duplicates.

In both steps the oldest file (by mtime -- these are write-once uploads, so
mtime == upload time) in a duplicate group is kept; the rest are candidates
for removal.

Renaming: any surviving file whose name isn't already a generated
"<Subject>_<Type>[_<date>].pdf" style name (i.e. anything UUID-like or
purely numeric) gets a new name derived from deterministic keyword/regex
extraction over the same title + first-page text used for fingerprinting.
No LLM calls, no embeddings, no random IDs/timestamps/UUIDs in the
generated name -- collisions are resolved with a small numeric suffix.

Safe by default: this script only *reports* what it would do unless run
with --apply. There is no version control for storage/uploads (it's
gitignored), so deletions here are unrecoverable -- review the dry-run
report before applying.

Usage:
    python scripts/document_storage_cleanup.py               # dry run
    python scripts/document_storage_cleanup.py --apply        # execute
    python scripts/document_storage_cleanup.py --root PATH    # override dir
"""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from pypdf import PdfReader

from app.core.config import settings

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^\d+$")
GENERATED_NAME_RE = re.compile(r"^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*(?:_\d{4}-\d{2}-\d{2})?(?:_\d+)?$")

MAX_FIRST_PAGE_WORDS = 500
FINGERPRINT_SIMILARITY_THRESHOLD = 0.95

# (pattern, subject-or-None, document_type). First match wins; order matters
# (more specific phrases before generic ones). `subject=None` means derive
# the subject from title/first-line text instead of a fixed keyword.
DOC_TYPE_RULES: list[tuple[re.Pattern, str | None, str]] = [
    (re.compile(r"cyber\s*crime|cyber\s*fraud|online\s*fraud|hacking", re.IGNORECASE), "Cyber_Crime", "Complaint"),
    (re.compile(r"vehicle\s+theft|car\s+theft|bike\s+theft|motorcycle\s+theft|stolen\s+vehicle", re.IGNORECASE), "Vehicle_Theft", "Complaint"),
    (re.compile(r"right\s+to\s+information|\brti\b", re.IGNORECASE), "RTI", "Application"),
    (re.compile(r"first\s+information\s+report|\bfir\b", re.IGNORECASE), "FIR", "Complaint"),
    (re.compile(r"affidavit", re.IGNORECASE), None, "Affidavit"),
    (re.compile(r"legal\s+notice|notice\s+of", re.IGNORECASE), None, "Notice"),
    (re.compile(r"power\s+of\s+attorney", re.IGNORECASE), "Power_Of_Attorney", "Deed"),
    (re.compile(r"\blast\s+will\b|\btestament\b", re.IGNORECASE), None, "Will"),
    (re.compile(r"petition", re.IGNORECASE), None, "Petition"),
    (re.compile(r"agreement|contract", re.IGNORECASE), None, "Agreement"),
    (re.compile(r"complaint", re.IGNORECASE), None, "Complaint"),
]

MONTHS = {
    "jan": "01", "january": "01", "feb": "02", "february": "02", "mar": "03", "march": "03",
    "apr": "04", "april": "04", "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "sept": "09", "september": "09", "oct": "10",
    "october": "10", "nov": "11", "november": "11", "dec": "12", "december": "12",
}
DATE_ISO_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
DATE_SLASH_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2}|\d{2})\b")
DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(MONTHS) + r")\.?,?\s+(20\d{2})\b", re.IGNORECASE,
)


@dataclass
class ExtractedContent:
    title: str
    first_page_text: str
    fingerprint: str


@dataclass
class PlanItem:
    path: Path
    action: str  # "remove_exact_duplicate" | "remove_near_duplicate" | "rename" | "keep" | "skip"
    detail: str = ""
    new_path: Path | None = None


@dataclass
class Report:
    total_scanned: int = 0
    duplicates_found: int = 0
    files_removed: list[str] = field(default_factory=list)
    files_renamed: list[tuple[str, str]] = field(default_factory=list)
    files_skipped: list[tuple[str, str]] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            "=== Document Storage Cleanup Report ===",
            f"Total files scanned: {self.total_scanned}",
            f"Duplicate files found: {self.duplicates_found}",
            f"Files removed: {len(self.files_removed)}",
        ]
        for name in self.files_removed:
            lines.append(f"  - removed: {name}")
        lines.append(f"Files renamed: {len(self.files_renamed)}")
        for old, new in self.files_renamed:
            lines.append(f"  - renamed: {old} -> {new}")
        lines.append(f"Files skipped: {len(self.files_skipped)}")
        for name, reason in self.files_skipped:
            lines.append(f"  - skipped: {name} ({reason})")
        return "\n".join(lines)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_fingerprint(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def extract_content(path: Path) -> ExtractedContent:
    reader = PdfReader(str(path))
    title = ""
    if reader.metadata and reader.metadata.title:
        title = reader.metadata.title.strip()
    first_page_text = ""
    if reader.pages:
        raw = reader.pages[0].extract_text() or ""
        words = raw.split()[:MAX_FIRST_PAGE_WORDS]
        first_page_text = " ".join(words)
    fingerprint = _normalize_fingerprint(f"{title} {first_page_text}")
    return ExtractedContent(title=title, first_page_text=first_page_text, fingerprint=fingerprint)


def _extract_date(text: str) -> str | None:
    match = DATE_ISO_RE.search(text)
    if match:
        return match.group(0)
    match = DATE_SLASH_RE.search(text)
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = f"20{year}"
        try:
            return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
        except ValueError:
            return None
    match = DATE_TEXT_RE.search(text)
    if match:
        day, month_name, year = match.groups()
        month = MONTHS[month_name.lower()]
        return f"{int(year):04d}-{month}-{int(day):02d}"
    return None


def _clean_words(text: str, limit: int = 4) -> str:
    words = re.findall(r"[A-Za-z0-9]+", text)[:limit]
    return "_".join(w.capitalize() for w in words if w)


def _derive_subject(content: ExtractedContent) -> str:
    if content.title and len(content.title) > 3 and content.title.lower() not in {"untitled", "document"}:
        subject = _clean_words(content.title)
        if subject:
            return subject
    for line in content.first_page_text.splitlines() or [content.first_page_text]:
        line = line.strip()
        if len(line) > 8:
            subject = _clean_words(line)
            if subject:
                return subject
    return "Document"


def generate_filename(content: ExtractedContent, used_names: set[str]) -> str:
    combined = f"{content.title} {content.first_page_text}"
    subject, doc_type = None, "Document"
    for pattern, fixed_subject, matched_type in DOC_TYPE_RULES:
        if pattern.search(combined):
            doc_type = matched_type
            subject = fixed_subject or _derive_subject(content)
            break
    if subject is None:
        subject = _derive_subject(content)
    date = _extract_date(combined)
    base = f"{subject}_{doc_type}_{date}" if date else f"{subject}_{doc_type}"
    candidate = f"{base}.pdf"
    suffix = 1
    lower_used = {n.lower() for n in used_names}
    while candidate.lower() in lower_used:
        suffix += 1
        candidate = f"{base}_{suffix}.pdf"
    used_names.add(candidate)
    return candidate


def _looks_generated(stem: str) -> bool:
    return bool(UUID_RE.match(stem) or NUMERIC_RE.match(stem)) or not GENERATED_NAME_RE.match(stem)


class UnionFind:
    def __init__(self, items: list[Path]) -> None:
        self.parent = {p: p for p in items}

    def find(self, item: Path) -> Path:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: Path, b: Path) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[Path, list[Path]]:
        out: dict[Path, list[Path]] = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return out


def build_plan(root: Path) -> tuple[list[PlanItem], Report]:
    report = Report()
    pdf_files = sorted(p for p in root.glob("*.pdf") if p.is_file())
    report.total_scanned = len(pdf_files)

    plan: list[PlanItem] = []
    survivors: list[Path] = []

    # Step 1: exact duplicates by SHA-256.
    hashes: dict[str, list[Path]] = {}
    for path in pdf_files:
        hashes.setdefault(sha256_of(path), []).append(path)

    for group in hashes.values():
        group.sort(key=lambda p: p.stat().st_mtime)
        keeper, dupes = group[0], group[1:]
        survivors.append(keeper)
        if dupes:
            report.duplicates_found += len(dupes)
        for dupe in dupes:
            plan.append(PlanItem(dupe, "remove_exact_duplicate", f"identical to {keeper.name}"))
            report.files_removed.append(dupe.name)

    # Step 2: near duplicates by content fingerprint, among step-1 survivors.
    content_by_path: dict[Path, ExtractedContent] = {}
    unreadable: list[Path] = []
    for path in survivors:
        try:
            content_by_path[path] = extract_content(path)
        except Exception:  # noqa: BLE001 - any parse failure just excludes it from fingerprinting
            unreadable.append(path)

    fingerprinted = [p for p in survivors if p in content_by_path]
    uf = UnionFind(fingerprinted)
    for i, a in enumerate(fingerprinted):
        fp_a = content_by_path[a].fingerprint
        if not fp_a:
            continue
        for b in fingerprinted[i + 1 :]:
            fp_b = content_by_path[b].fingerprint
            if not fp_b:
                continue
            if SequenceMatcher(None, fp_a, fp_b).ratio() >= FINGERPRINT_SIMILARITY_THRESHOLD:
                uf.union(a, b)

    kept: list[Path] = []
    for members in uf.groups().values():
        members.sort(key=lambda p: p.stat().st_mtime)
        keeper, dupes = members[0], members[1:]
        kept.append(keeper)
        if dupes:
            report.duplicates_found += len(dupes)
        for dupe in dupes:
            plan.append(PlanItem(dupe, "remove_near_duplicate", f"content fingerprint matches {keeper.name}"))
            report.files_removed.append(dupe.name)

    # Files that failed content extraction never entered the fingerprint
    # grouping, so they're always kept (never auto-removed on a heuristic
    # we couldn't actually evaluate) but also never renamed.
    kept.extend(unreadable)

    # Step 3: rename survivors that don't already have a meaningful name.
    used_names = {p.name for p in root.glob("*") if p not in {item.path for item in plan}}
    for path in sorted(kept):
        if path in unreadable:
            plan.append(PlanItem(path, "skip", "unreadable PDF content, left as-is"))
            report.files_skipped.append((path.name, "unreadable PDF content"))
            continue
        if not _looks_generated(path.stem):
            plan.append(PlanItem(path, "keep", "already has a meaningful name"))
            continue
        content = content_by_path[path]
        if not content.title and not content.first_page_text:
            plan.append(PlanItem(path, "skip", "no extractable title or text"))
            report.files_skipped.append((path.name, "no extractable title or text"))
            continue
        used_names.discard(path.name)
        new_name = generate_filename(content, used_names)
        plan.append(PlanItem(path, "rename", "", new_path=path.with_name(new_name)))
        report.files_renamed.append((path.name, new_name))

    return plan, report


def apply_plan(plan: list[PlanItem]) -> None:
    for item in plan:
        if item.action in ("remove_exact_duplicate", "remove_near_duplicate"):
            item.path.unlink()
        elif item.action == "rename" and item.new_path is not None:
            item.path.rename(item.new_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect duplicate PDFs and rename non-meaningful filenames in upload storage.")
    parser.add_argument("--root", type=Path, default=settings.upload_storage_dir, help="Upload storage directory.")
    parser.add_argument("--apply", action="store_true", help="Actually delete duplicates and rename files (default is dry-run).")
    args = parser.parse_args()

    plan, report = build_plan(args.root)
    print(report.render())

    if args.apply:
        apply_plan(plan)
        print("\nApplied.")
    else:
        print("\nDry run only -- no files were changed. Re-run with --apply to execute this plan.")


if __name__ == "__main__":
    main()
