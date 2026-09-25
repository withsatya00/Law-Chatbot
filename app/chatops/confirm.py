"""Natural-language confirmation, refusal and control-verb detection.

Confirmation is a SECURITY control here, not a UX nicety: it is what stands
between a user and an irreversible or outward-facing action (e-signing,
submitting to a notary, deleting a draft, revoking an account). So it is
deterministic pattern matching, never a model judgement -- "is this a yes?"
must not be answerable by something that can hallucinate.

Conservative by design. An unrecognised reply is NOT a yes: `is_affirmative`
returns False and the caller re-asks. The cost of re-asking is a wasted turn;
the cost of reading an ambiguous reply as consent is an action the user never
authorised.
"""

import re

# WHY NOT `\b` HERE
# -----------------
# Python's `\b` is defined in terms of `\w`, and `\w` does NOT include Indic
# combining marks: in "हाँ" the vowel sign "ा" (Mc) and candrabindu "ँ" (Mn)
# are both non-word characters, so `हाँ\b` never matches "हाँ". The same is
# true of every Indic script this app supports -- Tamil "ஆம்" ends in a
# pulli, Malayalam and Telugu affirmatives end in vowel signs, and so on.
#
# Left as `\b`, a Hindi user answering "हाँ" to an e-sign or notarization
# confirmation would have been told the reply was unclear, every time, in
# every Indic script -- while English "yes" worked. Replaced with an explicit
# token-end assertion that accepts end-of-string or a separator.
#
# This is also stricter than `\b` for Latin, which is a bonus: a bare "y"
# alternative can no longer fire inside "your document", because "o" is not a
# separator.
_TOKEN_END = r"(?=$|[\s.!,?;:।॥؟…—–-])"

# Affirmatives across the languages this product replies in. Anchored to the
# start of the message so "haan" only counts when it IS the answer, not when
# it appears mid-sentence in an unrelated clause.
_AFFIRMATIVE = re.compile(
    r"^\s*(?:"
    r"y|ye|yes|yeah|yep|yup|ok|okay|k|sure|fine|correct|right|confirm(?:ed|s)?|agree(?:d)?|proceed|go\s*ahead"
    r"|do\s*it|please\s*do|accept(?:ed)?|approve(?:d)?|submit|send\s*it"
    r"|haan|han|haa|ha|hn|ji|ji\s*haan|theek|thik|thik\s*hai|theek\s*hai|sahi|sahi\s*hai|bilkul|zaroor|jaroor"
    r"|kar\s*do|kar\s*dijiye|karo|kro|bana\s*do|banao|generate\s*karo|submit\s*karo|aage\s*badho"
    r"|हाँ|हां|जी|जी\s*हाँ|ठीक|ठीक\s*है|सही|बिल्कुल|ज़रूर|जरूर|कर\s*दो|करो|बनाओ|आगे\s*बढ़ो|पुष्टि|सहमत"
    r"|होय|बरोबर"                       # Marathi
    r"|હા|બરાબર"                        # Gujarati
    r"|হ্যাঁ|হ্যা|ঠিক"                   # Bengali/Assamese
    r"|ਹਾਂ|ਠੀਕ"                          # Punjabi
    r"|ହଁ|ଠିକ୍"                          # Odia
    r"|ஆம்|சரி"                          # Tamil
    r"|అవును|సరే"                        # Telugu
    r"|ಹೌದು|ಸರಿ"                         # Kannada
    r"|അതെ|ശരി"                          # Malayalam
    r"|ہاں|جی|ٹھیک"                      # Urdu
    r")" + _TOKEN_END,
    re.IGNORECASE,
)

_NEGATIVE = re.compile(
    r"^\s*(?:"
    r"n|no|nope|nah|not|don'?t|do\s*not|cancel|stop|abort|reject|decline|refuse|wrong|incorrect"
    r"|nahi|nahin|nai|na|mat|mat\s*karo|nahi\s*chahiye|rehne\s*do|chhodo|chod\s*do|galat"
    r"|नहीं|नही|ना|मत|मत\s*करो|नहीं\s*चाहिए|रहने\s*दो|छोड़\s*दो|ग़लत|गलत|रद्द"
    r"|नको|नाही"                        # Marathi
    r"|ના|નહીં"                          # Gujarati
    r"|না|নয়"                            # Bengali/Assamese
    r"|ਨਹੀਂ|ਨਾ"                          # Punjabi
    r"|ନା|ନାହିଁ"                          # Odia
    r"|இல்லை|வேண்டாம்"                    # Tamil
    r"|కాదు|వద్దు"                        # Telugu
    r"|ಇಲ್ಲ|ಬೇಡ"                          # Kannada
    r"|അല്ല|വേണ്ട"                        # Malayalam
    r"|نہیں|نہ"                          # Urdu
    r")" + _TOKEN_END,
    re.IGNORECASE,
)


def is_affirmative(message: str) -> bool:
    """True only for a recognised yes. An unclear reply is never consent."""
    text = (message or "").strip()
    if not text:
        return False
    return bool(_AFFIRMATIVE.match(text)) and not bool(_NEGATIVE.match(text))


def is_negative(message: str) -> bool:
    return bool(_NEGATIVE.match((message or "").strip()))


def classify(message: str) -> str:
    """`"yes"`, `"no"`, or `"unclear"` -- the three outcomes a caller handles."""
    if is_affirmative(message):
        return "yes"
    if is_negative(message):
        return "no"
    return "unclear"
