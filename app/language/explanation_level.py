import re

# `None` means "no cue found" -- kept distinct from `"citizen"` (the caller's
# default) so a message that explicitly asked for citizen-level phrasing
# could later be told apart from one that said nothing at all, even though
# there's no explicit citizen-trigger today.
_LIKE_IM_AGE_PATTERN = re.compile(r"like i(?:'m| am) (?:a )?(\d{1,2})\b", re.IGNORECASE)
_CHILD_PATTERN = re.compile(
    r"for a (child|kid)|explain (to|for) a (child|kid)|like a (child|kid)|child.?friendly", re.IGNORECASE
)
_PROFESSIONAL_PATTERN = re.compile(
    r"professionally|legal jargon|technical(ly)? explain|like a lawyer|for a lawyer|professional(?:\s+(?:terms|language|explanation))?",
    re.IGNORECASE,
)
_STUDENT_PATTERN = re.compile(
    r"like a student|student.?friendly|simple(r)? terms|explain simply|easy language|in simple words",
    re.IGNORECASE,
)

EXPLANATION_LEVEL_INSTRUCTIONS = {
    "professional": "Use precise legal terminology and cite provisions formally, as you would for a legal professional.",
    "citizen": "Use plain, everyday language a non-lawyer can follow, explaining any legal terms you use.",
    "student": "Use simple, clear language with relatable examples, as you would explaining to a student.",
    "child": "Use very simple words and short sentences, with a friendly everyday example, as you would explaining to a child.",
    "simple": "Use concise everyday language, explain every legal term, and preserve all citations, qualifications, and safety warnings.",
    "detailed": "Explain the rule, procedure, practical steps, exceptions, and uncertainties in detail while preserving the same citations and safety warnings.",
    "advocate": "Use precise advocate-style legal terminology and structured reasoning while preserving the same grounded citations, uncertainties, and safety warnings.",
}


def detect_explanation_level(text: str) -> str | None:
    """Detects an explicit request for a simplified/technical explanation style.

    Returns `None` when the message gives no such cue -- callers should default
    to `"citizen"` themselves rather than this function baking that in, since
    "no signal" and "explicitly asked for citizen-level" are different things.
    """
    age_match = _LIKE_IM_AGE_PATTERN.search(text)
    if age_match:
        return "child" if int(age_match.group(1)) <= 10 else "student"
    if _CHILD_PATTERN.search(text):
        return "child"
    if _PROFESSIONAL_PATTERN.search(text):
        return "professional"
    if _STUDENT_PATTERN.search(text):
        return "student"
    return None
