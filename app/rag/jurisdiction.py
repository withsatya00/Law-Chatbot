"""Guards against presenting a MODEL law as binding law.

The reported answer (live, Hinglish, to "Mera landlord bina notice ghar khali
karwa raha hai"):

    "Model Tenancy Act, 2021 ke tehat landlord bina sahi legal process ke
     aapko ghar se nahi nikaal sakta ... Section 21 ... Section 17 ...
     Section 14 ... Rent Authority ke paas deposit kar sakte hain."

Every section number there is real. The answer is still wrong, in the way that
matters most to someone about to act on it: **the Model Tenancy Act, 2021 is
not in force anywhere by its own operation.**

* It is a *model* law circulated by the Union Government (approved by the
  Union Cabinet on 2 June 2021) for States and Union Territories to adopt.
* Land, and the rights of landlord and tenant, are a State subject --
  Entry 18 of List II (State List), Seventh Schedule to the Constitution.
  Parliament cannot enact a binding tenancy code for the States, which is
  precisely why this one is a model.
* It binds a tenant only where their State has enacted it (with whatever
  modifications that State made). Where a State has not, that State's own
  rent-control or tenancy statute continues to govern -- and those differ
  substantially from the Model Act on exactly the questions users ask about:
  notice periods, grounds of eviction, deposit caps, and which forum hears
  the dispute.

So telling a tenant in a non-adopting State that "the landlord cannot evict
you without following Section 21" is not a harmless simplification. It names
a forum ("Rent Authority") that may not exist for them, and a procedure their
landlord is under no obligation to follow.

This module mirrors `statute_currency.py`'s two-stage shape exactly:

* `jurisdiction_directive()` runs BEFORE generation and puts the constraint
  in the prompt, so a well-behaved answer is correct in the first place.
* `annotate_answer()` runs AFTER generation as a safety net, appending the
  caveat when the model asserted a model law anyway.

What this module deliberately does NOT do is claim which States have adopted
the Model Act. That list changes, this repository has no verified source for
it, and a wrong entry either way would be worse than the caveat itself. The
caveat is written to be correct regardless of which State the user is in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelLaw:
    """A central *model* statute that binds nobody until a State enacts it."""

    name: str
    # Matches the Act by name in a question, a retrieved chunk, or an answer.
    pattern: re.Pattern[str]
    # The constitutional reason it cannot be central binding law.
    subject_entry: str
    # What governs instead where the model has not been adopted.
    fallback_description: str


MODEL_LAWS: tuple[ModelLaw, ...] = (
    ModelLaw(
        name="Model Tenancy Act, 2021",
        pattern=re.compile(
            r"model\s+tenancy\s+act|मॉडल\s+टेनेंसी|आदर्श\s+किरायेदारी|मॉडल\s+किरायेदारी",
            re.IGNORECASE,
        ),
        subject_entry="Entry 18 of the State List (Seventh Schedule)",
        fallback_description=(
            "that State's own rent control or tenancy legislation, which may set different notice "
            "periods, grounds of eviction, deposit limits and adjudicating forum"
        ),
    ),
)

# Union Territories and States, plus a small set of major cities, so a caveat
# can name the user's State when they have already mentioned it rather than
# asking a question they have effectively answered. Deliberately small and
# unambiguous -- this is a convenience, and `detect_state` returning None
# simply means the caveat asks instead of naming.
_STATES: tuple[str, ...] = (
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir", "Ladakh", "Puducherry",
    "Chandigarh", "Andaman and Nicobar Islands", "Lakshadweep",
)
CITY_TO_STATE: dict[str, str] = {
    "mumbai": "Maharashtra", "pune": "Maharashtra", "nagpur": "Maharashtra", "nashik": "Maharashtra",
    "bengaluru": "Karnataka", "bangalore": "Karnataka", "mysuru": "Karnataka",
    "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu", "madurai": "Tamil Nadu",
    "hyderabad": "Telangana", "kolkata": "West Bengal", "howrah": "West Bengal",
    "ahmedabad": "Gujarat", "surat": "Gujarat", "vadodara": "Gujarat", "rajkot": "Gujarat",
    "jaipur": "Rajasthan", "jodhpur": "Rajasthan", "udaipur": "Rajasthan",
    "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh", "noida": "Uttar Pradesh",
    "ghaziabad": "Uttar Pradesh", "varanasi": "Uttar Pradesh", "agra": "Uttar Pradesh",
    "meerut": "Uttar Pradesh", "prayagraj": "Uttar Pradesh", "allahabad": "Uttar Pradesh",
    "gurgaon": "Haryana", "gurugram": "Haryana", "faridabad": "Haryana",
    "indore": "Madhya Pradesh", "bhopal": "Madhya Pradesh", "gwalior": "Madhya Pradesh",
    "patna": "Bihar", "ranchi": "Jharkhand", "raipur": "Chhattisgarh",
    "bhubaneswar": "Odisha", "guwahati": "Assam", "dehradun": "Uttarakhand",
    "thiruvananthapuram": "Kerala", "kochi": "Kerala", "kozhikode": "Kerala",
    "amritsar": "Punjab", "ludhiana": "Punjab", "shimla": "Himachal Pradesh",
    "visakhapatnam": "Andhra Pradesh", "vijayawada": "Andhra Pradesh",
    "new delhi": "Delhi", "noida extension": "Uttar Pradesh",
}


def detect_state(text: str) -> str | None:
    """The Indian State/UT named (or clearly implied by a city) in `text`.

    Checked longest-name-first so "Andhra Pradesh" is never shadowed by a
    substring match, and city lookup only runs when no State was named
    outright.
    """
    if not text:
        return None
    lowered = text.lower()
    for state in sorted(_STATES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(state.lower())}\b", lowered):
            return state
    for city, state in sorted(CITY_TO_STATE.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(city)}\b", lowered):
            return state
    return None


def relevant_model_laws(text: str) -> list[ModelLaw]:
    return [law for law in MODEL_LAWS if law.pattern.search(text or "")]


def jurisdiction_directive(question: str, context: str) -> str:
    """A prompt block constraining how a model law may be described.

    Empty for the overwhelming majority of turns, so the prompt is unchanged
    unless a model law is actually in play.
    """
    laws = relevant_model_laws(f"{question}\n{context}")
    if not laws:
        return ""
    state = detect_state(question)
    lines = [
        (
            "Jurisdiction note -- the question and/or the retrieved material involve a MODEL law. A model law "
            "is a template the Union Government recommends to States; it is NOT binding law by its own force "
            "anywhere in India. You must therefore:"
        ),
    ]
    for law in laws:
        lines.append(
            f"- Describe the {law.name} as a model/template law, never as law automatically applicable "
            f"across India. This subject falls under {law.subject_entry}, so it binds a person only where "
            f"their State has actually enacted it. Where a State has not, {law.fallback_description} "
            f"applies instead."
        )
    lines.append(
        "- Do NOT state a definitive nationwide rule about notice periods, grounds of eviction, deposit "
        "limits or which authority hears the dispute on the strength of the model law alone."
    )
    if state:
        lines.append(
            f"- The user appears to be in {state}. Say plainly that whether this model law governs their "
            f"tenancy depends on whether {state} has enacted it, and that their State's own tenancy/rent "
            f"legislation may govern instead. Do not assert either way unless the retrieved material says so."
        )
    else:
        lines.append(
            "- The user has not named their State. Ask which State the property is in, since the answer "
            "genuinely depends on it, and give the general position in the meantime."
        )
    lines.append(
        "- Never claim to know which States have adopted the model law unless the retrieved material "
        "explicitly says so."
    )
    return "\n".join(lines)


# The appended caveat, per language. English is the authored original; the
# others follow the codebase's existing convention of covering the languages
# this product actually replies in and falling back to English otherwise.
#
# TRANSLATION PROVENANCE: hindi/hinglish authored here; not reviewed by a
# practising advocate.
_CAVEAT_TEMPLATES: dict[str, str] = {
    "english": (
        "**Important -- which law actually applies to you:** the {law} is a *model* law. The Union "
        "Government recommended it to the States; it does not apply anywhere on its own. Tenancy is a "
        "State subject ({entry}), so it governs your tenancy only if {state_clause} enacted it. If not, "
        "your State's own rent control or tenancy law applies, and its notice periods, grounds of "
        "eviction and adjudicating forum may be quite different. {ask}"
    ),
    "hinglish": (
        "**Zaroori -- aap par kaunsa kanoon lagta hai:** {law} ek *model* law hai. Central Government ne "
        "ise States ko recommend kiya tha; yeh apne aap kahin lagoo nahi hota. Tenancy State subject hai "
        "({entry}), isliye yeh aap par tabhi lagega jab {state_clause} ise enact kiya ho. Agar nahi, to "
        "aapke State ka apna rent control/tenancy kanoon lagega, aur uske notice period, eviction ke "
        "grounds aur forum kaafi alag ho sakte hain. {ask}"
    ),
    "hindi": (
        "**महत्वपूर्ण -- आप पर कौन-सा कानून लागू होता है:** {law} एक *आदर्श (model)* कानून है। केंद्र सरकार ने "
        "इसे राज्यों को अपनाने हेतु अनुशंसित किया था; यह स्वतः कहीं लागू नहीं होता। किरायेदारी राज्य सूची का "
        "विषय है ({entry}), इसलिए यह आप पर तभी लागू होगा जब {state_clause} इसे अधिनियमित किया हो। यदि नहीं, "
        "तो आपके राज्य का अपना किराया नियंत्रण/किरायेदारी कानून लागू होगा, जिसमें नोटिस अवधि, बेदखली के "
        "आधार तथा सक्षम मंच काफी भिन्न हो सकते हैं। {ask}"
    ),
}
_STATE_CLAUSE: dict[str, dict[bool, str]] = {
    "english": {True: "{state} has", False: "your State has"},
    "hinglish": {True: "{state} ne", False: "aapke State ne"},
    "hindi": {True: "{state} ने", False: "आपके राज्य ने"},
}
_ASK: dict[str, str] = {
    "english": "Tell me which State the property is in and I can be more specific.",
    "hinglish": "Property kis State mein hai yeh bata dijiye, to main aur specific bata sakta hoon.",
    "hindi": "बताइए कि संपत्ति किस राज्य में है, तो मैं अधिक स्पष्ट रूप से बता सकूँगा।",
}
_ENTRY_LABEL: dict[str, str] = {
    "english": "Entry 18, State List, Seventh Schedule",
    "hinglish": "Entry 18, State List, Seventh Schedule",
    "hindi": "प्रविष्टि 18, राज्य सूची, सातवीं अनुसूची",
}
# If the answer already carries this idea, the caveat is redundant. Matching on
# the distinguishing CONCEPT (model/adoption/State subject) rather than exact
# wording, so a correctly-hedged answer is not annotated twice.
_ALREADY_HEDGED = re.compile(
    r"model\s+law|template\s+law|adopt(?:ed|ion)?\s+by\s+(?:the\s+)?state|state\s+subject"
    r"|not\s+automatically\s+(?:applicable|in\s+force)|apne\s+aap\s+kahin\s+lagoo\s+nahi"
    r"|आदर्श\s*\(model\)|स्वतः\s+कहीं\s+लागू\s+नहीं|राज्य\s+सूची",
    re.IGNORECASE,
)


def annotate_answer(answer: str, language: str | None, question: str = "") -> str:
    """Append the model-law caveat when `answer` relies on a model law without
    already saying that it depends on State adoption.

    Purely additive -- the grounded answer text is never edited, so this
    cannot corrupt a correct answer. Silent when the answer already hedges
    correctly, which is the normal case once `jurisdiction_directive` has
    done its job.
    """
    if not answer:
        return answer
    laws = relevant_model_laws(answer)
    if not laws or _ALREADY_HEDGED.search(answer):
        return answer
    key = (language or "english").strip().lower()
    if key not in _CAVEAT_TEMPLATES:
        key = "english"
    state = detect_state(f"{question}\n{answer}")
    state_clause = _STATE_CLAUSE[key][state is not None].format(state=state or "")
    caveat = _CAVEAT_TEMPLATES[key].format(
        law=laws[0].name,
        entry=_ENTRY_LABEL[key],
        state_clause=state_clause,
        ask="" if state else _ASK[key],
    ).strip()
    return f"{answer}\n\n{caveat}"


def jurisdiction_warnings(answer: str, question: str = "") -> list[str]:
    """Machine-readable companions to the caveat, for `ChatResponse.warnings`.

    Lets a caller (or a UI) surface the limitation without parsing prose.
    """
    laws = relevant_model_laws(answer)
    if not laws:
        return []
    state = detect_state(f"{question}\n{answer}")
    warnings = [
        f"{law.name} is a model law and applies only in States that have enacted it."
        for law in laws
    ]
    if state is None:
        warnings.append("The user's State is unknown; tenancy law is State-specific.")
    return warnings
