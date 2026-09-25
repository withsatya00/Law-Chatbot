"""Which languages this drafting engine can produce a FULLY localized
document in, and which it can only partly localize.

Why this exists: the drafting stack localizes a document through several
independent layers -- the LLM writes the body in the requested language, but
section headings (`heading_translations.py`), document titles
(`title_translations.py`), field labels (`field_label_translations.py`) and
the non-LLM fallback's boilerplate (`fallback_phrases.py`) all come from
closed lookup tables. Those tables cover thirteen languages. A request for
one of the other eleven Eighth Schedule languages this app nominally accepts
produced a document whose BODY was in the requested language but whose
HEADINGS were in English -- a mixed-language legal document, which is exactly
the defect the drafting-quality pass was opened to fix.

Nothing here blocks a draft. Producing a document with an English heading
over correct Assamese prose is still far more useful than refusing. What was
wrong was doing it SILENTLY, so the user discovered the mixture only after
downloading the PDF. `supported_level()` lets the conversation layer say so
up front, in one sentence, before the document is generated.

Deliberately NOT done here: substituting a related language's headings (Hindi
for Nepali/Maithili/Dogri, Bengali for Assamese, Urdu for Sindhi/Kashmiri).
Those scripts and much of the legal register do overlap, which is precisely
what makes the substitution dangerous -- it would look correct to a reviewer
skimming the document while presenting one language's legal terminology as
another's, in a document intended for filing. An honest English fallback is
visibly a fallback; a plausible wrong translation is not.
"""

from typing import Literal

SupportLevel = Literal["full", "body_only"]

# Languages with complete coverage across every localization table: headings,
# titles, field labels, and deterministic-fallback boilerplate.
FULLY_SUPPORTED_LANGUAGES: frozenset[str] = frozenset({
    "english",
    "hinglish",
    "hindi",
    "marathi",
    "gujarati",
    "bengali",
    "punjabi",
    "tamil",
    "telugu",
    "kannada",
    "malayalam",
    "odia",
    "urdu",
})

# Accepted, and the LLM will write the body in them, but the structural
# vocabulary above falls back to English. Listed explicitly rather than
# inferred so that adding a language to the translation tables is a
# deliberate, reviewable change in one place.
BODY_ONLY_LANGUAGES: frozenset[str] = frozenset({
    "assamese",
    "nepali",
    "konkani",
    "maithili",
    "sanskrit",
    "dogri",
    "bodo",
    "sindhi",
    "kashmiri",
    "santali",
    "manipuri",
})


def normalize(language: str | None) -> str:
    return (language or "english").strip().lower()


def supported_level(language: str | None) -> SupportLevel:
    """How completely a draft in `language` can be localized.

    An unrecognized language is reported as "full" rather than "body_only":
    the tables would fall back to English for it anyway, and warning about a
    language this app does not claim to support would be noise. The warning
    exists for the eleven languages we DO accept but only partly localize.
    """
    return "body_only" if normalize(language) in BODY_ONLY_LANGUAGES else "full"


def is_fully_supported(language: str | None) -> bool:
    return supported_level(language) == "full"
