"""Phase 2 item 3: case timeline and contradiction detection.

Two related jobs, kept in one module because they read the same inputs:

* the TIMELINE puts everything the case knows -- chat messages, uploaded
  evidence, collected draft fields -- on one dated sequence, which is what a
  complaint's "chronological sequence of events" section and a lawyer-ready
  summary both need;
* CONTRADICTION DETECTION compares the same facts across those sources and
  reports where they disagree.

The second exists because of a specific failure mode. A user types "₹35,000"
in chat, the bank receipt they upload says ₹45,000, and the draft has to
contain one number. Silently picking either is the worst possible behaviour:
it produces a document the user signs, addressed to a police station, stating
an amount they never verified -- and the discrepancy surfaces later, in front
of the authority, as an apparent falsehood by the complainant.

So the rule this module enforces is: never resolve a conflict automatically.
Detect it, attribute both sides to their source, and make the user choose.
`blocking_conflicts()` is what the export path consults; a conflict stays
blocking until the user explicitly resolves it.
"""

from dataclasses import dataclass, field
from datetime import date

from app.casefile.facts import (
    CONFLICTING_SLOTS,
    SLOT_LABELS,
    ExtractedFact,
    extract_facts,
    facts_from_fields,
    parse_date,
)
from app.core import clock


@dataclass(frozen=True)
class TimelineEvent:
    """One dated thing that happened, or one undated thing that was said."""

    occurred_on: str  # ISO date, or "" when the source carried no date
    description: str
    source: str
    source_kind: str  # "chat" | "evidence" | "draft"
    reference: str = ""  # annexure number / message id, when there is one

    def as_dict(self) -> dict[str, str]:
        return {
            "occurred_on": self.occurred_on,
            "description": self.description,
            "source": self.source,
            "source_kind": self.source_kind,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class FactConflict:
    """Two or more incompatible values for the same fact slot."""

    slot: str
    label: str
    values: tuple[str, ...]
    sources: tuple[str, ...]
    excerpts: tuple[str, ...]

    @property
    def question(self) -> str:
        rendered = " / ".join(f'"{value}"' for value in self.values)
        return (
            f"{self.label}: your case currently states {rendered}. "
            "Which one is correct? I will not choose for you -- a legal document has to carry the value you "
            "can stand behind."
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "slot": self.slot,
            "label": self.label,
            "values": list(self.values),
            "sources": list(self.sources),
            "excerpts": list(self.excerpts),
            "question": self.question,
        }


@dataclass
class CaseFacts:
    """Everything the case knows, with each fact still attached to its source."""

    facts: list[ExtractedFact] = field(default_factory=list)
    events: list[TimelineEvent] = field(default_factory=list)

    def by_slot(self, slot: str) -> list[ExtractedFact]:
        return [fact for fact in self.facts if fact.slot == slot]


def _sort_key(event: TimelineEvent) -> tuple[int, str]:
    # Undated entries sort last, not first: a timeline that opens with
    # "(no date) the applicant states..." reads as though the case has no
    # beginning. `0`/`1` is the dated/undated bucket.
    return (1, event.description) if not event.occurred_on else (0, event.occurred_on)


def build_timeline(
    *,
    messages: list[dict[str, str]] | None = None,
    evidence: list[dict[str, object]] | None = None,
    draft_fields: dict[str, str] | None = None,
) -> CaseFacts:
    """Collects facts and dated events from every source the case has.

    `messages` are `{"role", "content"}` dicts as stored in conversation
    memory; only the user's own turns are read, since an assistant reply
    restates the user's facts and would double-count them (and, worse, make a
    correctly-restated fact look like independent corroboration).
    """
    facts: list[ExtractedFact] = []
    events: list[TimelineEvent] = []

    for index, message in enumerate(messages or []):
        if message.get("role") != "user":
            continue
        content = str(message.get("content") or "")
        if not content.strip():
            continue
        source = f"chat message {index + 1}"
        message_facts = extract_facts(content, source)
        facts.extend(message_facts)
        for fact in message_facts:
            if fact.slot == "incident_date":
                events.append(TimelineEvent(
                    occurred_on=fact.normalized,
                    description=_describe(content),
                    source=source,
                    source_kind="chat",
                ))

    for item in evidence or []:
        name = str(item.get("document_name") or item.get("filename") or "evidence")
        annexure = str(item.get("annexure") or "")
        source = f"{annexure} ({name})" if annexure else name
        text = str(item.get("text") or "")
        item_facts = extract_facts(text, source)
        facts.extend(item_facts)
        dated = [fact for fact in item_facts if fact.slot == "incident_date"]
        if dated:
            for fact in dated:
                events.append(TimelineEvent(
                    occurred_on=fact.normalized,
                    description=f"{name} records {fact.value}",
                    source=source,
                    source_kind="evidence",
                    reference=annexure,
                ))
        else:
            events.append(TimelineEvent(
                occurred_on="",
                description=f"{name} (no date found in the document)",
                source=source,
                source_kind="evidence",
                reference=annexure,
            ))

    if draft_fields:
        field_facts = facts_from_fields(draft_fields)
        facts.extend(field_facts)
        for fact in field_facts:
            if fact.slot == "incident_date":
                events.append(TimelineEvent(
                    occurred_on=fact.normalized,
                    description="Incident date as entered in the draft",
                    source="draft details",
                    source_kind="draft",
                ))

    return CaseFacts(facts=facts, events=sorted(events, key=_sort_key))


def _describe(content: str, limit: int = 160) -> str:
    collapsed = " ".join(content.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[:limit].rstrip()}..."


def detect_conflicts(case_facts: CaseFacts, resolved: dict[str, str] | None = None) -> list[FactConflict]:
    """Slots where the case holds two or more incompatible values.

    `resolved` maps a slot to the value the user chose. A resolved slot drops
    out entirely -- that is the ONLY way a conflict stops being reported, and
    it requires an explicit human decision (see `resolve_conflict`).

    Only slots in `CONFLICTING_SLOTS` are compared. Two phone numbers or two
    names are ordinary (a complainant and a respondent both have one); two
    different amounts lost in the same incident are not.
    """
    resolved = resolved or {}
    conflicts: list[FactConflict] = []
    for slot in sorted(CONFLICTING_SLOTS):
        if slot in resolved:
            continue
        slot_facts = case_facts.by_slot(slot)
        distinct: dict[str, ExtractedFact] = {}
        for fact in slot_facts:
            distinct.setdefault(fact.normalized, fact)
        if len(distinct) < 2:
            continue
        chosen = list(distinct.values())
        conflicts.append(FactConflict(
            slot=slot,
            label=SLOT_LABELS.get(slot, slot),
            values=tuple(fact.value for fact in chosen),
            sources=tuple(fact.source for fact in chosen),
            excerpts=tuple(fact.excerpt for fact in chosen),
        ))
    return conflicts


def blocking_conflicts(case_facts: CaseFacts, resolved: dict[str, str] | None = None) -> list[FactConflict]:
    """The conflicts that must be resolved before a final export.

    Currently every detected conflict blocks. Kept as a separate function from
    `detect_conflicts` so the two questions -- "what disagrees?" and "what
    stops the user filing?" -- stay independently answerable, and so a future
    severity distinction has an obvious home that does not change either
    caller.
    """
    return detect_conflicts(case_facts, resolved)


def resolve_conflict(resolved: dict[str, str], slot: str, chosen_value: str) -> dict[str, str]:
    """Records the user's decision for one slot. Returns a NEW mapping rather
    than mutating, so a resolution is an explicit write by the caller and
    cannot happen as a side effect of merely inspecting conflicts."""
    updated = dict(resolved or {})
    updated[slot] = chosen_value
    return updated


def render_timeline(events: list[TimelineEvent], *, unknown_date_label: str = "Date not recorded") -> str:
    """The timeline as plain text, for a draft's chronology section or a
    lawyer-ready summary."""
    lines: list[str] = []
    for event in events:
        when = _pretty_date(event.occurred_on) if event.occurred_on else unknown_date_label
        reference = f" [{event.reference}]" if event.reference else ""
        lines.append(f"{when} — {event.description}{reference} (source: {event.source})")
    return "\n".join(lines)


def _pretty_date(iso: str) -> str:
    parsed = parse_date(iso)
    if parsed is None:
        return iso
    return parsed.strftime("%d %B %Y")


def timeline_span(events: list[TimelineEvent]) -> tuple[str, str]:
    """First and last dated event, as ISO strings. `("", "")` when nothing in
    the case carries a date -- which is itself worth surfacing, since a
    complaint with no date at all is missing a fact every authority asks for."""
    dated = sorted(event.occurred_on for event in events if event.occurred_on)
    return (dated[0], dated[-1]) if dated else ("", "")


def days_since(iso_date: str, *, today: date | None = None) -> int | None:
    """Days between `iso_date` and today. Used by the cyber-fraud workflow,
    where the reporting window materially changes what the user should do
    first -- `None` when the date is unknown or unparseable."""
    parsed = parse_date(iso_date)
    if parsed is None:
        return None
    reference = today or clock.today()
    return (reference - parsed).days
