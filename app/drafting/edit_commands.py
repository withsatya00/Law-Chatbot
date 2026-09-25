import re
from dataclasses import dataclass
from typing import Literal

from app.drafting.field_label_translations import localized_field_label
from app.drafting.intent import _typo_tolerant_match
from app.drafting.templates.base import DraftTemplateDefinition
from app.language.detector import extract_requested_language


def _find_fuzzy_synonym_span(text: str, synonym: str) -> str | None:
    """The actual substring of `text` matching `synonym` (a field label/
    synonym, possibly several words -- "mobile number", "applicant
    address"), tolerating one ordinary typo per word -- "moblie
    number"/"aplicant address" -- or `None` if nothing matches. Anchored to
    a contiguous, correctly-ordered run of words (same approach as
    `discovery._fuzzy_find`), and every synonym word must be 4+ characters
    for the same reason `_typo_tolerant_match` itself requires it: a short
    word is too easy to fuzzy-match by coincidence.

    Returns the matched TEXT (not just whether it matched) so callers can
    build a value-extraction pattern around what the user actually typed
    ("moblie number") instead of the correctly-spelled synonym, which would
    never literally appear when there was a typo in the field name too.

    Found via live testing: "change the moblie number to 9988776655" (field
    genuinely present on the template, no authored synonym needed -- just
    its own plain label) came back `action=unknown` purely because of the
    one-letter typo in "mobile".
    """
    lowered_text = text.lower()
    exact_at = lowered_text.find(synonym)
    if exact_at != -1:
        return text[exact_at : exact_at + len(synonym)]
    words = synonym.split(" ")
    if any(len(w) < 4 for w in words):
        return None
    text_word_matches = list(re.finditer(r"[^\s,.!?;:]+", text))
    span = len(words)
    for start in range(len(text_word_matches) - span + 1):
        window = text_word_matches[start : start + span]
        if all(_typo_tolerant_match(words[i], window[i].group().lower()) for i in range(span)):
            return text[window[0].start() : window[-1].end()]
    return None

EditAction = Literal[
    "replace_field", "append_field", "remove_paragraph", "translate", "regenerate", "restyle", "format",
    "suggest", "export", "approve", "lock", "unlock", "rollback", "unknown",
]

# Part 54 "Complete Multilingual Drafting Pass": each verb/phrase tuple below
# now also carries Tamil/Telugu/Kannada/Bengali forms alongside the existing
# English/Hindi ones -- previously an edit command in one of those four
# languages ("மாற்று" = "change") matched none of these, fell through to
# `EditAction.unknown`, and (per `DraftConversationEngine._continue_preview`)
# dropped out of the draft flow entirely instead of being recognized as an
# edit. Plain substring `in` containment (not `\b`-anchored regex) is used
# throughout this file already, so no Unicode word-boundary concerns apply
# here the way they did for `DraftIntentDetector` (see `intent.py`'s
# `_contains_as_word` docstring).
_REPLACE_VERBS = (
    "replace", "change", "correct", "update", "fix", "बदलो", "बदलिए", "सही करो",
    # Romanised Hinglish, which is how the observed session actually phrased
    # an edit: "Recipient ka address Noida se Ghaziabad kar do". None of
    # these existed, so the command parsed as `unknown` and the whole message
    # fell through to RAG while the draft sat untouched. Safe to include the
    # very generic "kar do"/"kardo": a replace verb only produces an edit
    # when `_match_field_and_value` also resolves a real field of THIS
    # template, otherwise interpretation continues past this branch.
    "badal do", "badal dijiye", "badlo", "badal", "kar do", "kardo", "kr do", "kar dijiye",
    # Devanagari counterparts of the generic "kar do" trigger just above --
    # same safety rationale (an edit only fires once `_match_field_and_value`
    # ALSO resolves a real field of THIS template). Missing entirely
    # previously, so a genuinely Devanagari-script edit command ("मोबाइल
    # नंबर बदलो 9988776655") never even reached the replace-verb branch at
    # all when it used a bare "करो"/"कर दो" rather than one of the three
    # specific phrases already listed above ("बदलो", "बदलिए", "सही करो").
    "करो", "कर दो", "कर दीजिये", "बदल दो", "बदलकर",

    "மாற்று", "திருத்து", "புதுப்பி",
    "మార్చు", "సరిచేయి", "నవీకరించు",
    "ಬದಲಾಯಿಸಿ", "ಬದಲಿಸಿ", "ಸರಿಪಡಿಸಿ", "ನವೀಕರಿಸಿ",
    "পরিবর্তন", "ঠিক করো", "হালনাগাদ করো",
    # Part 57: malayalam/marathi/gujarati/punjabi/odia/urdu forms.
    "മാറ്റുക", "മാറ്റൂ", "ശരിയാക്കുക", "പുതുക്കുക",
    "बदला", "दुरुस्त करा", "अद्ययावत करा",
    "બદલો", "સુધારો", "અપડેટ કરો",
    "ਬਦਲੋ", "ਠੀਕ ਕਰੋ", "ਅੱਪਡੇਟ ਕਰੋ",
    "ବଦଳାନ୍ତୁ", "ସଠିକ୍ କରନ୍ତୁ", "ଅଦ୍ୟତନ କରନ୍ତୁ",
    "بدلیں", "درست کریں", "اپ ڈیٹ کریں",
)
# "<old value> se <new value> kar do" -- the Hinglish way to state a
# replacement. Only the NEW value is captured; the old one is the thing being
# replaced and must never be written back into the draft.
#
# "ki jagah"/"ke jagah" ("instead of"/"in place of") is just as ordinary a
# way to say this as "se" -- "15 days ki jagah 7 days kar do" -- but was
# missing entirely (only "se"/"से" were recognised), so a from-to edit
# phrased this way came back `action=unknown` purely because of the
# connector word, even when a real, unambiguous field/value pair followed.
# Found via live QA testing: this exact phrasing, aimed at changing a
# draft's response deadline, fell all the way out of the draft flow and was
# answered as an ordinary (and here, actually wrong -- see the matching QA
# writeup) legal question instead of applying the edit.
_FROM_TO_PATTERN = re.compile(
    r"\b(?P<old>[\wऀ-ॿ][\w\sऀ-ॿ.,-]{0,60}?)\s+(?:se|से|ki\s*jagah|ke\s*jagah|की\s*जगह|के\s*जगह)\s+"
    r"(?P<new>[\wऀ-ॿ][\w\sऀ-ॿ.,-]{0,60}?)"
    r"\s*(?:kar\s*do|kardo|kar\s*dijiye|kr\s*do|badal\s*do|badlo|कर\s*दो|करो|बदल\s*दो)\b",
    re.IGNORECASE,
)
# "Sirf amount badlo, baaki same rakho." ("just change the amount, keep the
# rest the same") -- a trailing scope clause, not a replacement value. Every
# value-extraction pattern in `_match_field_and_value` below is anchored at
# the end of the string, so without stripping this first, the greedy/
# end-anchored patterns capture the literal trailer text (", baaki same
# rakho") as `new_value` and write it straight into the field. Confirmed live
# against the current interpreter before this fix:
# `EditCommandInterpreter().interpret("Sirf amount badlo, baaki same rakho.",
# get_template("legal_notice"), "english")` returned
# `new_value=", baaki same rakho"`. Stripped up front instead (scoped to
# `_match_field_and_value`/replace_field only -- `append_field`'s trailing
# text is the content being appended, not noise, and must never be stripped
# the same way), leaving `new_value=None` so the conversation layer asks
# "what should the new value be?" instead of silently corrupting the field.
_UNCHANGED_REST_TRAILER_PATTERN = re.compile(
    r",?\s*(?:and\s+)?(?:keep\s+)?(?:baaki|baki|rest|everything\s+else|sab(?:\s*kuch)?)\s+"
    r"(?:the\s+)?(?:same|wahi|waisa)\s*(?:hi\s*)?"
    r"(?:rakho|rakhiye|rehne\s*do|rahne\s*do|rehne\s*dena|rakhna|kar\s*do)?\s*[.!।]?\s*$",
    re.IGNORECASE,
)
# BUG-010 (QA pass, 2026-09-11): "add a fact to the draft" had NO
# representation at all in `EditAction` -- only `replace_field` (overwrite)
# existed, and mapping "add" onto it would silently DELETE whatever was
# already in the field the moment a user tried to add one more fact. This is
# a genuine third verb family, not a variant of replace.
#
# `\b`-anchored (unlike `_REPLACE_VERBS`/`_REMOVE_VERBS` above, which use
# plain `in` containment per this file's existing convention): "add" is
# short enough to collide with an ordinary substring match inside "add**
# ress**", which every legal-notice edit command mentioning an address
# would otherwise trip over.
_APPEND_VERB_PATTERN = re.compile(
    r"\b(?:add|include|mention|insert|append|जोड़ो|जोड़ दो|जोड़ें|शामिल करो|jodo|jod\s*do|likho|लिखो)\b",
    re.IGNORECASE,
)
# Captures the actual content to append -- everything after the append verb
# (and its optional "karo"/"kar do" helper and an optional "that"/"ye"/"ki"/
# ":" connector). Mirrors the shape of the existing replace-field extraction
# patterns below, but for the append verb family instead of the replace one.
#  - "Ye fact add karo: maine 5 aur 12 August ko reminders bheje." -> the verb
#    is "add", followed by "karo", then ":", then the content.
#  - "Add that I sent reminders on 5 and 12 August." -> the verb is "add",
#    followed directly by the English connector "that", then the content.
_APPEND_VALUE_PATTERN = re.compile(
    r"\b(?:add|include|mention|insert|append|जोड़ो|जोड़ दो|जोड़ें|शामिल करो|jodo|jod\s*do|likho|लिखो)\b"
    r"\s*(?:karo|kar\s*do|kardo|kar\s*dijiye|करो|कर\s*दो)?"
    r"\s*(?:that|this|ye|yeh|ki|isme|is\s*mein)?"
    r"\s*[:\-]?\s*(?P<value>.+)$",
    re.IGNORECASE,
)
# What's left over when the regex above backtracks out of its own optional
# helper-verb group (e.g. "Add karo" alone: nothing to say, so ".+" is left
# to capture "karo" itself) -- not real content to append.
_APPEND_BARE_HELPER_WORDS = frozenset({
    "karo", "kar do", "kardo", "kar dijiye", "करो", "कर दो", "that", "this", "ye", "yeh", "ki",
})


# What the rendered draft calls its blocks, mapped to the field a user means
# when they name one. Applied only when the template actually has that field.
_RENDERED_BLOCK_ALIASES: tuple[tuple[str, str], ...] = (
    ("recipient address", "respondent_address"),
    ("recipient name", "respondent_name"),
    ("recipient", "respondent_name"),
    ("sender address", "applicant_address"),
    ("sender name", "applicant_name"),
    ("sender", "applicant_name"),
    ("deponent name", "applicant_name"),
    ("complainant name", "applicant_name"),
    ("applicant name", "applicant_name"),
    ("mera naam", "applicant_name"),
    ("my name", "applicant_name"),
    ("naam", "applicant_name"),
)
_REMOVE_VERBS = (
    "remove", "delete", "हटाओ", "हटा दो",
    "நீக்கு", "తీసివేయి", "ತೆಗೆದುಹಾಕಿ", "সরাও", "মুছে দাও",
    "നീക്കം ചെയ്യുക", "काढून टाका", "દૂર કરો", "ਹਟਾਓ", "ହଟାନ୍ତୁ", "ہٹا دیں",
)
_REGENERATE_VERBS = (
    "regenerate", "rewrite", "फिर से बनाओ", "दोबारा बनाओ",
    "மீண்டும் உருவாக்கு", "మళ్లీ తయారు చేయి", "ಮತ್ತೆ ರಚಿಸಿ", "আবার তৈরি করো",
    "വീണ്ടും തയ്യാറാക്കുക", "पुन्हा तयार करा", "ફરીથી બનાવો", "ਦੁਬਾਰਾ ਬਣਾਓ",
    "ପୁନଃ ପ୍ରସ୍ତୁତ କରନ୍ତୁ", "دوبارہ تیار کریں",
)
_EXPORT_VERBS = (
    "export", "download", "save as", "generate pdf", "generate a pdf", "make pdf", "create pdf",
    "பதிவிறக்கு", "డౌన్‌లోడ్", "ಡೌನ್‌ಲೋಡ್", "ডাউনলোড",
    "ഡൗൺലോഡ്", "डाउनलोड", "ડાઉનલોડ", "ਡਾਊਨਲੋਡ", "ଡାଉନଲୋଡ୍", "ڈاؤن لوڈ",
)
_APPROVE_PHRASES = (
    "looks good", "approve", "approved", "finalize", "that's fine", "that looks fine", "ठीक है", "सही है",
    "நன்றாக இருக்கிறது", "బాగుంది", "ಚೆನ್ನಾಗಿದೆ", "ভালো লাগছে",
    "മികച്ചത്", "छान आहे", "साचु छे", "ਠੀਕ ਹੈ", "ଠିକ୍ ଅଛି", "ٹھیک ہے", "منظور",
)
# Part 57 "Drafting Lifecycle Redesign": confirms locking an already-approved
# draft ("Approve" -> "Lock this draft?" -> this). Deliberately narrower/more
# explicit than `_APPROVE_PHRASES` (a bare "yes"/"ok" is handled separately
# by `DraftConversationEngine`'s own confirmation-turn logic, which already
# knows which question it's confirming) -- these are the phrases someone
# would use to name the "lock" action specifically, not just agree.
_LOCK_PHRASES = (
    "lock", "lock it", "lock the draft", "lock this draft", "confirm lock", "yes lock",
    "लॉक करें", "लॉक करो", "इसे लॉक करें",
    "பூட்டு", "லாக் செய்", "లాక్ చేయి", "ಲಾಕ್ ಮಾಡಿ", "লক করো",
    "ലോക്ക് ചെയ്യുക", "લોક કરો", "लॉक करा", "ਲਾਕ ਕਰੋ", "ଲକ୍ କରନ୍ତୁ", "لاک کریں",
)
_UNLOCK_PHRASES = (
    "unlock", "unlock it", "unlock the draft", "unlock this draft", "edit again", "make changes",
    "अनलॉक करें", "अनलॉक करो", "इसे अनलॉक करें",
    "பூட்டு திற", "அன்லாக் செய்", "అన్‌లాక్ చేయి", "ಅನ್‌ಲಾಕ್ ಮಾಡಿ", "আনলক করো",
    "അൺലോക്ക് ചെയ്യുക", "અનલોક કરો", "अनलॉक करा", "ਅਨਲਾਕ ਕਰੋ", "ଅନଲକ୍ କରନ୍ତୁ", "ان لاک کریں",
)
# "rollback to version 2" / "restore version 2" / "revert to version 2" --
# also matched against the native "version"/rollback words for the 6 newly
# localized languages, always paired with a digit (the version number is
# mandatory; a bare "rollback" with no number is `unknown`, not guessed at).
_ROLLBACK_PATTERN = re.compile(
    r"(rollback|roll\s*back|restore|revert|पुनर्स्थापित|वापस\s*लाओ|மீட்டமை|పునరుద్ధరించు|ಮರುಸ್ಥಾಪಿಸಿ|পুনরুদ্ধার"
    r"|പുനഃസ്ഥാപിക്ക|पुनर्संचयित|પુનઃસ્થાપિત|ਮੁੜ-ਬਹਾਲ|ପୁନଃସ୍ଥାପନ|بحال)"
    r".*?(\d+)",
    re.IGNORECASE,
)
# Part 52 "One-Page Format": requests to tighten/re-format an already-generated
# draft without changing its language or field content.
_FORMAT_PHRASES = (
    "one page", "one-page", "single page", "single-page",
    "format professionally", "professionally format", "professional format",
    "make it professional", "compact it", "make it compact",
    "ஒரு பக்கம்", "ఒక పేజీ", "ಒಂದು ಪುಟ", "এক পাতা",
    "ഒരു പേജ്", "एक पान", "એક પાનું", "ਇੱਕ ਪੰਨਾ", "ଏକ ପୃଷ୍ଠା", "ایک صفحہ",
)

# "restyle" edit action -- QA pass 2026-09-11, step 8 of the Priority-2
# journey. "Tone polite but firm karo." had no recognized action at all
# (bare "karo" isn't a `_REPLACE_VERBS` trigger, and there's no field named
# "tone" for `_match_field_and_value` to resolve anyway -- this is a
# request to RE-RENDER the same facts in a different voice, not to change
# one field's value). Left the draft flow entirely and was answered as a
# fresh, unrelated legal question. `_TONE_VALUE_PATTERN` captures the
# requested tone/register itself ("polite but firm") for
# `LegalDraftEngine.regenerate`'s `style_instruction` to turn into a
# same-facts-different-voice prompt directive.
_TONE_VALUE_PATTERN = re.compile(
    r"\b(?:tone|register|lehja|लहजा|स्वर)\b\s*(?:ko|को)?\s*"
    r"(?P<value>.+?)\s*"
    r"(?:karo|kar\s*do|kardo|kar\s*dijiye|banao|banaiye|करो|कर\s*दो|बनाओ|बना\s*दो)?\s*[.!।]?$",
    re.IGNORECASE,
)

# Finding-010 (QA pass, 2026-09-11): "review and suggest improvements, but
# don't touch the draft" had no recognised action at all -- it fell through
# unrecognised (safely: the fallback never mutates the draft, but it also
# never actually reviews it) to a generic "I don't have a template for
# that" reply. This is a genuinely distinct request from every OTHER action
# here: every one of them (replace/append/restyle/regenerate/...) changes
# the persisted draft; this one must never call `LegalDraftEngine.
# regenerate` at all, only read the current draft and comment on it.
# Substring `in` containment, like `_APPROVE_PHRASES`/`_FORMAT_PHRASES`
# above -- "suggest" alone covers "suggest"/"suggests"/"suggestion(s)" as a
# plain substring, so those don't need separate entries.
_SUGGEST_PHRASES = (
    "suggest", "sujhav", "सुझाव", "review this draft", "review the draft", "review my draft",
)


@dataclass(frozen=True)
class EditCommand:
    action: EditAction
    target_field: str | None = None
    new_value: str | None = None
    target_language: str | None = None
    export_format: str | None = None
    version_number: int | None = None


class EditCommandInterpreter:
    """Rule-based interpreter for preview-stage chat messages.

    Best-effort, not a general NLU parser: recognizes a fixed set of verbs
    (replace/change/remove/translate/regenerate/export/approve) plus each
    template's own `field_synonyms` (defined in that template's YAML file) to
    figure out which field a message like "change my address" targets.
    Anything it can't confidently parse comes back as `unknown` so the
    conversation engine asks a clarifying question instead of guessing wrong.
    """

    def interpret(self, text: str, template: DraftTemplateDefinition, language: str = "english") -> EditCommand:
        lowered = text.strip().lower()

        # An explicit language name ("Hindi me kar do", "Tamil la draft
        # pannunga") is just as much a language-change request as one that
        # literally says "translate" -- Part 52 requires both phrasings to
        # work, since real revision requests rarely use the word "translate".
        target_language = extract_requested_language(text)
        if "translate" in lowered or target_language:
            return EditCommand(action="translate", target_language=target_language)

        if any(phrase in lowered for phrase in _FORMAT_PHRASES):
            return EditCommand(action="format")

        tone_match = _TONE_VALUE_PATTERN.search(text)
        if tone_match:
            tone_value = tone_match.group("value").strip().rstrip(".")
            if tone_value and tone_value.lower() not in _APPEND_BARE_HELPER_WORDS:
                return EditCommand(action="restyle", new_value=tone_value)

        # Checked before every field-changing verb below: a suggestion-only
        # request must never be mistaken for one of them (Finding-010 -- see
        # `_SUGGEST_PHRASES`'s own comment). None of these phrases overlap
        # with `_REPLACE_VERBS`/`_APPEND_VERB_PATTERN`/`_REMOVE_VERBS`.
        if any(phrase in lowered for phrase in _SUGGEST_PHRASES):
            return EditCommand(action="suggest")

        if any(verb in lowered for verb in _EXPORT_VERBS):
            export_format = next((fmt for fmt in ("pdf", "docx", "txt", "rtf") if fmt in lowered), None)
            return EditCommand(action="export", export_format=export_format)

        rollback_match = _ROLLBACK_PATTERN.search(lowered)
        if rollback_match:
            return EditCommand(action="rollback", version_number=int(rollback_match.group(2)))

        # Checked before the broader `_APPROVE_PHRASES` so "lock"/"unlock"
        # (which don't overlap with any approve phrase) always resolve to
        # their own distinct action rather than falling through.
        if any(phrase in lowered for phrase in _UNLOCK_PHRASES):
            return EditCommand(action="unlock")
        if any(phrase in lowered for phrase in _LOCK_PHRASES):
            return EditCommand(action="lock")

        if any(phrase in lowered for phrase in _APPROVE_PHRASES):
            return EditCommand(action="approve")

        # Checked before `_REPLACE_VERBS`: append is a distinct action
        # (BUG-010), and must never be reached via the replace path, which
        # would overwrite rather than merge.
        if _APPEND_VERB_PATTERN.search(lowered):
            target_field, new_value = self._match_append_field_and_value(text, template, language)
            if target_field:
                return EditCommand(action="append_field", target_field=target_field, new_value=new_value)

        if any(verb in lowered for verb in _REPLACE_VERBS):
            target_field, new_value = self._match_field_and_value(lowered, text, template, language)
            if target_field:
                return EditCommand(action="replace_field", target_field=target_field, new_value=new_value)

        if any(verb in lowered for verb in _REMOVE_VERBS):
            target_field, _ = self._match_field_and_value(lowered, text, template, language)
            return EditCommand(action="remove_paragraph", target_field=target_field)

        if any(verb in lowered for verb in _REGENERATE_VERBS):
            return EditCommand(action="regenerate")

        return EditCommand(action="unknown")

    @staticmethod
    def _strip_possessives(text: str) -> str:
        """Drops the Hinglish/Hindi possessive particles between two nouns.

        "recipient ka address" and "recipient address" are the same phrase;
        without this only the second one could ever match a synonym, and the
        first fell back to the bare "address" synonym -- a different field.

        Never strips "ki"/"ke" when it is actually the first half of the
        "ki jagah"/"ke jagah" ("instead of") from-to connector `_FROM_TO_
        PATTERN` looks for -- found live: "15 days ki jagah 7 days kar do"
        had its "ki" stripped as if it were the unrelated possessive
        particle, leaving orphaned "jagah" for `_FROM_TO_PATTERN` to no
        longer recognise as a connector at all, which fell through to a
        cruder fallback pattern that captured "jagah 7 days" (with the
        stray word still attached) as the new value instead of just "7 days".
        """
        return re.sub(r"\s+(?:ka|ki|ke|का|की|के)\s+(?!jagah\b)(?!जगह\b)", " ", text)

    @staticmethod
    def _field_candidates(template: DraftTemplateDefinition, language: str = "english") -> dict[str, str]:
        """The full synonym -> field_key vocabulary for `template`, shared
        by both replace-field and append-field target resolution (extracted
        from what used to be inlined at the top of `_match_field_and_value`
        only -- append needs the exact same "which field does the user
        mean" resolution, just without that method's own value-extraction
        half).

        Part 54 "Complete Multilingual Drafting Pass": the YAML-authored
        `field_synonyms` are English-only (creative shorthand like "their
        address"/"phone"), so a Tamil/Telugu/Kannada/Bengali user naming a
        field only by the label they were actually shown during collection
        ("இடம்" for "Place") previously matched nothing at all. Each
        field's own localized label is added as an extra synonym -- lower
        priority than the hand-authored `field_synonyms` (tried first, and
        sorted longest-first same as before) since a synonym is often more
        specific than the bare label.
        """
        candidates: dict[str, str] = dict(template.field_synonyms)
        # The words this product's OWN rendered draft uses for its blocks.
        # A user who reads a section headed "Recipient" says "recipient",
        # which no template lists as a synonym -- and the bare synonym
        # "address" then matched first and pointed the edit at the wrong
        # field. Longest-match ordering below makes "recipient address" win.
        field_keys = template.field_keys()
        for phrase, key in _RENDERED_BLOCK_ALIASES:
            if key in field_keys:
                candidates.setdefault(phrase, key)
        # Originally scoped to the fixed single-language-label set (below),
        # deliberately excluding "english"/"hinglish" on the reasoning that
        # `field_synonyms` already covers English shorthand. It didn't: most
        # templates' `field_synonyms` are a short, hand-picked list (a
        # handful of the most-likely-to-be-edited fields), not one entry per
        # field -- so a field with no authored synonym (e.g. "Mobile
        # Number", "Applicant Address" on `tenancy_termination_notice`) had
        # no way to be targeted by name at all in English/Hinglish, even
        # though that is EXACTLY the label the user was just shown during
        # collection. Confirmed live: "change the mobile number to
        # 9988776655" against a template whose only synonyms are
        # tenant/premises/place came back `action=unknown`. Every field's
        # own plain-English `label` is now a matchable synonym in every
        # language, same as the localized label already was for the ten
        # languages below -- lower priority than `field_synonyms`
        # (`setdefault`, tried first, longest-match wins regardless).
        for draft_field in template.all_fields():
            candidates.setdefault(draft_field.label.lower(), draft_field.key)
        if language in {
            "hindi", "tamil", "telugu", "kannada", "bengali",
            "malayalam", "marathi", "gujarati", "punjabi", "odia", "urdu",
        }:
            for draft_field in template.all_fields():
                localized = localized_field_label(template.draft_id, draft_field, language).lower()
                candidates.setdefault(localized, draft_field.key)
        return candidates

    def _match_field_and_value(
        self, lowered: str, original: str, template: DraftTemplateDefinition, language: str = "english"
    ) -> tuple[str | None, str | None]:
        # See `_UNCHANGED_REST_TRAILER_PATTERN`'s own comment: a trailing
        # "keep everything else the same" clause is scope, not a value, and
        # must be gone before any of the end-anchored value-extraction
        # patterns below ever see it.
        lowered = _UNCHANGED_REST_TRAILER_PATTERN.sub("", lowered).rstrip()
        original = _UNCHANGED_REST_TRAILER_PATTERN.sub("", original).rstrip()
        candidates = self._field_candidates(template, language)
        normalised = self._strip_possessives(lowered)
        # Case-preserved, so a captured value keeps the user's own
        # capitalisation -- a place or a party name written back into a legal
        # document lower-cased is a defect of its own.
        normalised_original = self._strip_possessives(original)
        for synonym, field_key in sorted(candidates.items(), key=lambda kv: -len(kv[0])):
            # The ACTUAL text matched -- possibly a typo'd spelling of
            # `synonym` -- is what the value-extraction patterns below
            # search for, not `synonym` itself: a pattern built from the
            # correctly-spelled synonym would never find anything in a
            # message that misspelled the field name too ("change the
            # moblie number to ..."), leaving `new_value` stuck at `None`
            # even after `target_field` was correctly identified.
            matched_text = _find_fuzzy_synonym_span(lowered, synonym) or _find_fuzzy_synonym_span(normalised, synonym)
            if matched_text is None:
                continue
            matched_original = (
                _find_fuzzy_synonym_span(normalised_original, matched_text.lower())
                or _find_fuzzy_synonym_span(normalised_original, synonym)
                or matched_text
            )
            # Everything after "to"/"with"/"as"/":" following the synonym is
            # the new value, e.g. "change police station to Hazratganj" ->
            # new_value = "Hazratganj".
            pattern = re.compile(re.escape(matched_original) + r"\s*(?:to|with|as|:)\s*(.+)$", re.IGNORECASE)
            match = pattern.search(original) or pattern.search(normalised_original)
            new_value = match.group(1).strip().rstrip(".") if match else None
            if new_value is None:
                # Hinglish states the change as "<old> se <new> kar do" --
                # from-to rather than to. Read only the NEW value; the old
                # one is what is being replaced.
                from_to = _FROM_TO_PATTERN.search(normalised_original)
                if from_to:
                    new_value = from_to.group("new").strip().rstrip(".")
            if new_value is None:
                # Natural Hinglish often omits "to": "pichli detail mein
                # naam Rahul Kumar kar do".  This intentionally runs AFTER
                # the more specific "old se new" grammar above. Devanagari
                # verb forms ("बदलो", "कर दो", "सही करो") are included
                # alongside the romanised ones -- previously only the
                # romanised forms were recognised, so a genuinely
                # Devanagari-script Hindi edit command ("मोबाइल नंबर बदलो
                # 9988776655") resolved WHICH field to change correctly (via
                # the field's own `hindi_label`) but always came back with
                # `new_value=None`, since no value-extraction pattern here
                # understood the Devanagari verb ending the sentence.
                direct = re.search(
                    re.escape(matched_original)
                    + r"\s+(?P<value>.+?)\s+(?:kar\s*do|kardo|kr\s*do|badal\s*do|badlo"
                    r"|करो|कर\s*दो|बदलो|बदल\s*दो|सही\s*करो)\s*[.!।]?$",
                    normalised_original,
                    re.IGNORECASE,
                )
                if direct:
                    new_value = direct.group("value").strip().rstrip(".")
            if new_value is None:
                # The opposite word order from the pattern above: the
                # change-verb sits right after the field name, with the new
                # value trailing at the end -- "mobile number change karo
                # 9988776655", "naam badal do Suresh Kumar" -- rather than
                # "<field> <value> kar do". Confirmed live: this exact
                # phrasing came back with `target_field` resolved but
                # `new_value=None`, silently asking the user to re-state a
                # value they had just given. Devanagari verbs ("बदलो", "सही
                # करो") covered the same way as the `direct` pattern above.
                verb_first = re.search(
                    re.escape(matched_original)
                    # Longer alternatives listed BEFORE the shorter ones they
                    # literally start with ("बदलो" before "बदल") -- regex
                    # alternation tries each option in order and stops at the
                    # first match, so "बदल" alone would otherwise "win" at
                    # position 0 of "बदलो" and leave its own trailing "ो"
                    # (a combining vowel sign) captured as part of the value.
                    + r"\s+(?:change|update|badlo|badal|बदलो|बदल|सही)\s*"
                    r"(?:kar\s*do|karo|kardo|kr\s*do|do|करो|कर\s*दो)?\s*(?::|to)?\s*"
                    r"(?P<value>.+?)\s*[.!।]?$",
                    normalised_original,
                    re.IGNORECASE,
                )
                if verb_first:
                    new_value = verb_first.group("value").strip().rstrip(".") or None
            return field_key, new_value
        return None, None

    def _match_append_field_and_value(
        self, original: str, template: DraftTemplateDefinition, language: str = "english"
    ) -> tuple[str | None, str | None]:
        """Resolves (target_field, content_to_append) for an `append_field`
        command -- BUG-010's fix.

        Deliberately separate from `_match_field_and_value`: replace has to
        find a field name to have ANY meaning ("change X" with no X is not
        an edit), but append needs somewhere to land even when the user
        names no field at all ("add that I sent reminders on 5 and 12
        August" -- no field mentioned, and shouldn't need to be: it
        overwhelmingly means the template's own freeform narrative field).
        Falls back to the template's `facts` field (rule 4: "select a safe
        semantic location and record that choice" -- the caller names the
        chosen section in its reply) when no field is named explicitly, and
        refuses to target a non-`textarea` field even if one is explicitly
        named (rule: appending to a single-value field like a name or a
        phone number doesn't make semantic sense) -- falling back to
        `facts` there too rather than corrupting a structured value.

        A bare append verb with NOTHING to add ("Ye naya fact add karo.",
        nothing after "karo") still resolves a `target_field` here (`value`
        comes back `None`) rather than `(None, None)` -- the caller
        (`EditCommandInterpreter.interpret`) only returns an `append_field`
        command when `target_field` is truthy, and `DraftConversationEngine.
        _continue_preview`'s `append_field` branch already asks "what should
        I add?" whenever `new_value` is falsy, exactly like `replace_field`
        does for a field named with no new value. Before this, the verb was
        recognised but silently discarded -- the message fell all the way
        out of the draft flow and was answered as an unrelated question,
        rather than asking the one clarifying question that would actually
        resolve it.
        """
        value: str | None = None
        value_match = _APPEND_VALUE_PATTERN.search(original)
        if value_match:
            candidate_value = value_match.group("value").strip().rstrip(".")
            # A bare helper word left over from an optional group the regex
            # backtracked out of ("Add karo" with nothing else to say) is not
            # real content -- e.g. without this, "Add karo" alone captured
            # "karo" itself as the "value" to append.
            if candidate_value and candidate_value.lower() not in _APPEND_BARE_HELPER_WORDS:
                value = candidate_value

        lowered = original.lower()
        normalised = self._strip_possessives(lowered)
        candidates = self._field_candidates(template, language)
        explicit_field: str | None = None
        for synonym, field_key in sorted(candidates.items(), key=lambda kv: -len(kv[0])):
            if _find_fuzzy_synonym_span(lowered, synonym) or _find_fuzzy_synonym_span(normalised, synonym):
                explicit_field = field_key
                break

        target_field = explicit_field
        if target_field is not None:
            resolved = template.get_field(target_field)
            if resolved is not None and resolved.field_type != "textarea":
                target_field = None  # not a narrative field -- fall through to the default below

        if target_field is None:
            default_field = template.get_field("facts")
            target_field = "facts" if default_field is not None else None

        return target_field, value

    @staticmethod
    def append_value(existing: str, new_content: str) -> str:
        """Merges `new_content` into `existing`, never discarding what was
        already there (rule 2: "existing content must remain") and never
        duplicating a fact that's already present (rule 7) -- a
        case/whitespace-insensitive substring check, since the point is to
        avoid re-adding the same sentence, not to police exact phrasing.
        """
        existing = (existing or "").strip()
        new_content = new_content.strip()
        if not existing:
            return new_content
        if new_content.lower() in existing.lower():
            return existing
        separator = " " if existing.endswith((".", "!", "?", "।")) else ". "
        return f"{existing}{separator}{new_content}"
