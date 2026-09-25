"""Picking one object out of several, without the user ever seeing an id.

The product rule this exists to satisfy: a user must never have to paste an
internal identifier. They say "the rent agreement one" or "the second one" or
just "2", and the workflow resolves that to a real record id internally.

Three properties matter and are why this is deterministic string matching
rather than a model call:

* a wrong resolution here means acting on SOMEONE ELSE'S object, or deleting
  the wrong draft -- so the mapping from what was typed to which record must
  be auditable, not inferred;
* the candidate list is always built by an owner-scoped service call, so a
  record the user may not touch is never in the list to be chosen at all;
* when nothing matches confidently, the answer is to ask again, never to
  guess -- `resolve` returns `None` and the caller re-lists.

Used by the draft, case, document and admin workflows, which is why it lives
here rather than in any one of them.
"""

import re
from dataclasses import dataclass

# Word forms of the first ten positions, in the languages this product is
# used in. "dusra"/"doosri"/"दूसरा" are what people actually type when they
# mean the second item in a list.
_ORDINALS: dict[str, int] = {
    "first": 1, "1st": 1, "pehla": 1, "pehli": 1, "पहला": 1, "पहली": 1,
    "second": 2, "2nd": 2, "dusra": 2, "dusri": 2, "doosra": 2, "doosri": 2, "दूसरा": 2, "दूसरी": 2,
    "third": 3, "3rd": 3, "teesra": 3, "teesri": 3, "तीसरा": 3, "तीसरी": 3,
    "fourth": 4, "4th": 4, "chautha": 4, "चौथा": 4,
    "fifth": 5, "5th": 5, "panchwa": 5, "पांचवां": 5,
    "last": -1, "aakhri": -1, "आखिरी": -1, "last wala": -1,
}

_LEADING_NUMBER = re.compile(r"^\s*(?:#|no\.?|number\s*)?([1-9][0-9]?)\b")
_NAMED_NUMBER = re.compile(r"\b(?:file|document|record)\s*(?:no\.?\s*)?([1-9][0-9]?)\b")
_ANY_NUMBER = re.compile(r"\b([1-9][0-9]?)\b")


@dataclass(frozen=True)
class Choice:
    """One selectable record. `key` is internal and never rendered."""

    key: str
    label: str
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "label": self.label, "detail": self.detail}


def from_dicts(items: list[dict[str, str]]) -> list[Choice]:
    """Rebuilds choices parked in workflow facts across turns."""
    return [
        Choice(key=str(item.get("key", "")), label=str(item.get("label", "")), detail=str(item.get("detail", "")))
        for item in items
        if item.get("key")
    ]


def render(choices: list[Choice], *, limit: int = 10) -> str:
    """A numbered list, ids omitted.

    Capped, because an admin listing hundreds of records in a chat bubble is
    unreadable -- the tail is summarised rather than dumped.
    """
    shown = choices[:limit]
    lines = [
        f"{index}. **{choice.label}**" + (f" — {choice.detail}" if choice.detail else "")
        for index, choice in enumerate(shown, start=1)
    ]
    if len(choices) > limit:
        lines.append(f"…and {len(choices) - limit} more. Narrow it down by name if you need one of those.")
    return "\n".join(lines)


def resolve(message: str, choices: list[Choice]) -> Choice | None:
    """The single choice `message` names, or None if it names none or many.

    Order: an explicit position (a number or an ordinal word), then a unique
    label match. A message matching two labels resolves to neither -- an
    ambiguous reference to a destructive action is exactly the case that must
    ask again.
    """
    if not choices:
        return None
    text = (message or "").strip().lower()
    if not text:
        return None

    position = _position(text, len(choices))
    if position is not None:
        return choices[position]

    matches = [choice for choice in choices if choice.label and choice.label.lower() in text]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        # Fall back to a distinctive word from each label ("rent" for
        # "Rent agreement notice"), which is how people actually refer to
        # their own documents. Still requires uniqueness.
        worded = [choice for choice in choices if _label_words(choice.label) & _label_words(text)]
        if len(worded) == 1:
            return worded[0]
    return None


def overlapping(message: str, choices: list[Choice]) -> list[Choice]:
    """Every choice `message` textually overlaps with, whether or not that
    is unique enough for `resolve` above to actually pick one.

    `resolve` collapses two very different situations into the same `None`:
    a message that names NO candidate at all (safe to assume the current
    context/default applies), and one that tries to name a specific
    candidate but matches more than one ambiguously (must ask, never
    silently guess -- confirmed live: "the loan agreement" against
    ["rent-agreement.pdf", "loan-agreement.pdf"] shares the word
    "agreement" with BOTH, so `resolve` correctly refuses to pick one, but
    a caller that could not tell the two situations apart silently fell
    back to whichever record was last used instead of asking, answering
    from the WRONG one). This lets a caller ask in the second case and
    default in the first.
    """
    text = (message or "").strip().lower()
    if not text or not choices:
        return []
    exact = [choice for choice in choices if choice.label and choice.label.lower() in text]
    if exact:
        return exact
    text_words = _label_words(text)
    if not text_words:
        return []
    return [choice for choice in choices if _label_words(choice.label) & text_words]


def _position(text: str, count: int) -> int | None:
    for word, ordinal in _ORDINALS.items():
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", text):
            index = count - 1 if ordinal == -1 else ordinal - 1
            return index if 0 <= index < count else None
    match = _LEADING_NUMBER.match(text) or _NAMED_NUMBER.search(text) or (
        _ANY_NUMBER.search(text) if len(text) <= 24 else None
    )
    if match:
        index = int(match.group(1)) - 1
        return index if 0 <= index < count else None
    return None


# Words too generic to identify a record on their own.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "my", "me", "one", "that", "this", "and", "for", "of", "to",
        "draft", "case", "document", "file", "open", "show", "delete", "please",
        "wala", "wali", "mera", "meri", "ko", "ka", "ki", "hai", "karo", "do",
    }
)


def _label_words(text: str) -> frozenset[str]:
    return frozenset(
        word for word in re.findall(r"[a-z0-9ऀ-ॿ]{3,}", (text or "").lower())
        if word not in _STOPWORDS
    )
