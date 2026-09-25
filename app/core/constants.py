LEGAL_DISCLAIMER = (
    "This information is provided for educational purposes only and should not be considered "
    "legal advice. Please consult a qualified advocate for legal advice specific to your situation."
)

DRAFT_DISCLAIMER = (
    "This document is an AI-generated draft based on the information provided by the user. It is "
    "intended only as a drafting aid. Before submitting it before any Court, Government Authority, "
    "Police Department, Tribunal, or any legal forum, it must be reviewed and approved by a "
    "qualified advocate."
)

INSUFFICIENT_CONTEXT_MESSAGE = "I couldn't find sufficient verified legal information to answer this accurately."

# Strict-RAG guardrail: the exact, hardcoded string returned when retrieval finds
# zero chunks or nothing clears the relevance threshold -- returned verbatim,
# never passed through the LLM, and never appended to (see `chat_service.py`'s
# `no_verified_context` short-circuit in `answer()`/`answer_stream()`).
# The strict-RAG refusal, per language. Previously a single hardcoded Hindi
# string returned to EVERY user regardless of the language they asked in, so a
# Tamil or English speaker who tripped the guardrail got an unexplained wall of
# Devanagari. `rag_prompt.md` still instructs the LLM to emit the Hindi wording
# verbatim when it judges the retrieved chunks insufficient -- that's why
# `is_no_verified_context()` below matches ANY variant, not just the caller's:
# the model's Hindi refusal has to be recognized before it can be swapped for
# the user's own language.
#
# TRANSLATION PROVENANCE: hindi is the original, authored string. The rest were
# produced by this assistant and have NOT been checked by native speakers --
# they are structurally safe (a fixed sentence with no interpolation) but the
# wording of the less widely-resourced languages in particular deserves review
# before this is shown to real users.
NO_VERIFIED_CONTEXT_MESSAGES: dict[str, str] = {
    "hindi": "इस प्रश्न से संबंधित कोई सत्यापित दस्तावेज़ वर्तमान Knowledge Base में उपलब्ध नहीं है।",
    "english": "No verified document related to this question is currently available in the Knowledge Base.",
    "hinglish": "Is sawaal se related koi verified document abhi Knowledge Base mein uplabdh nahi hai.",
    "assamese": "এই প্ৰশ্নৰ সৈতে জড়িত কোনো সত্যাপিত নথি বৰ্তমানে Knowledge Base ত উপলব্ধ নাই।",
    "bengali": "এই প্রশ্ন সম্পর্কিত কোনো যাচাইকৃত নথি বর্তমানে Knowledge Base-এ উপলব্ধ নেই।",
    "bodo": "बे सोंथिजों सोमोन्दो थानाय जेबो थि खालामनाय फोरमान बिलाइ दानि Knowledge Base आव मोननो हायाखै।",
    "dogri": "इस सवाल कन्ने सरबंधत कोई तस्दीक कीता दस्तावेज़ हून Knowledge Base च उपलब्ध नेईं ऐ।",
    "gujarati": "આ પ્રશ્ન સંબંધિત કોઈ ચકાસાયેલ દસ્તાવેજ હાલમાં Knowledge Base માં ઉપલબ્ધ નથી.",
    "kannada": "ಈ ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಯಾವುದೇ ಪರಿಶೀಲಿತ ದಾಖಲೆ ಪ್ರಸ್ತುತ Knowledge Base ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ.",
    "kashmiri": "یِمہ سوالَس مُتعلق کانہہ تصدیق شُدہ دستاویز وُنی Knowledge Base منز دستیاب چھُ نہٕ۔",
    "konkani": "ह्या प्रस्नाकडेन संबंदीत खंयचोच तपासिल्लो दस्तावेज सद्या Knowledge Base त उपलब्ध ना.",
    "maithili": "एहि प्रश्न सँ सम्बन्धित कोनो सत्यापित दस्तावेज वर्तमान मे Knowledge Base मे उपलब्ध नहि अछि।",
    "malayalam": "ഈ ചോദ്യവുമായി ബന്ധപ്പെട്ട സ്ഥിരീകരിച്ച രേഖകളൊന്നും നിലവിൽ Knowledge Base-ൽ ലഭ്യമല്ല.",
    "manipuri": "ꯃꯁꯤꯒꯤ ꯋꯥꯍꯪꯗꯨꯗꯥ ꯃꯔꯤ ꯂꯩꯅꯕꯥ ꯑꯆꯨꯝꯕꯥ ꯄꯅꯈꯛꯄꯥ ꯗꯣꯀꯨꯃꯦꯟꯠ ꯑꯃꯠꯇꯥ ꯍꯧꯖꯤꯛ Knowledge Base ꯗꯥ ꯂꯩꯇꯦ꯫",
    "marathi": "या प्रश्नाशी संबंधित कोणतेही सत्यापित दस्तऐवज सध्या Knowledge Base मध्ये उपलब्ध नाही.",
    "nepali": "यस प्रश्नसँग सम्बन्धित कुनै पनि प्रमाणित कागजात हाल Knowledge Base मा उपलब्ध छैन।",
    "odia": "ଏହି ପ୍ରଶ୍ନ ସହ ସମ୍ପର୍କିତ କୌଣସି ଯାଞ୍ଚ ହୋଇଥିବା ଦଲିଲ ବର୍ତ୍ତମାନ Knowledge Base ରେ ଉପଲବ୍ଧ ନାହିଁ।",
    "punjabi": "ਇਸ ਸਵਾਲ ਨਾਲ ਸਬੰਧਤ ਕੋਈ ਪ੍ਰਮਾਣਿਤ ਦਸਤਾਵੇਜ਼ ਇਸ ਵੇਲੇ Knowledge Base ਵਿੱਚ ਉਪਲਬਧ ਨਹੀਂ ਹੈ।",
    "sanskrit": "अस्य प्रश्नस्य सम्बद्धं किमपि प्रमाणितं पत्रं सम्प्रति Knowledge Base इत्यत्र न उपलभ्यते।",
    "santali": "ᱱᱚᱶᱟ ᱠᱩᱠᱞᱤ ᱥᱟᱶ ᱡᱩᱲᱟᱣ ᱡᱟᱦᱟᱸ ᱥᱟᱹᱨᱤ ᱠᱟᱜᱚᱡᱽ ᱱᱤᱛᱚᱜ Knowledge Base ᱨᱮ ᱵᱟᱹᱱᱩᱜᱼᱟ᱾",
    "sindhi": "هن سوال سان لاڳاپيل ڪوبه تصديق ٿيل دستاويز هن وقت Knowledge Base ۾ موجود ناهي.",
    "tamil": "இந்தக் கேள்வி தொடர்பான சரிபார்க்கப்பட்ட ஆவணம் எதுவும் தற்போது Knowledge Base-இல் கிடைக்கவில்லை.",
    "telugu": "ఈ ప్రశ్నకు సంబంధించిన ధృవీకరించబడిన పత్రం ఏదీ ప్రస్తుతం Knowledge Base లో అందుబాటులో లేదు.",
    "urdu": "اس سوال سے متعلق کوئی تصدیق شدہ دستاویز فی الحال Knowledge Base میں دستیاب نہیں ہے۔",
}

# Kept as a module-level name: `rag_prompt.md` quotes this exact wording, and
# existing callers/tests import it directly.
NO_VERIFIED_CONTEXT_MESSAGE = NO_VERIFIED_CONTEXT_MESSAGES["hindi"]


def no_verified_context_message(language: str | None) -> str:
    """The strict-RAG refusal in `language`, falling back to English for any
    language with no translation (never to Hindi -- showing an unfamiliar
    script is exactly the bug this replaces)."""
    if not language:
        return NO_VERIFIED_CONTEXT_MESSAGES["english"]
    return NO_VERIFIED_CONTEXT_MESSAGES.get(language.strip().lower(), NO_VERIFIED_CONTEXT_MESSAGES["english"])


def is_no_verified_context(text: str) -> bool:
    """True when `text` opens with the strict-RAG refusal in ANY language.

    Both the short-circuit path and the LLM itself can produce this line, and
    the LLM produces the Hindi wording (that is what `rag_prompt.md` tells it
    to emit) regardless of the user's language -- so recognizing it can't be
    scoped to the current language.
    """
    stripped = (text or "").lstrip()
    if any(stripped.startswith(message) for message in NO_VERIFIED_CONTEXT_MESSAGES.values()):
        return True
    # The model sometimes prefaces the mandatory line with its own empathy
    # sentence ("Yeh sunkar bahut bura laga ... <refusal>"), despite Rule 3.
    # A prefix-only match then missed it, so the refusal was shown with
    # unrelated citations still attached (confirmed live: a mobile-theft
    # refusal carrying a BNS Section 303 source). The messages are long,
    # distinctive sentences, so finding one early in a short answer is a
    # reliable refusal signal; a long substantive answer is left alone.
    head = stripped[:600]
    return any(message in head for message in NO_VERIFIED_CONTEXT_MESSAGES.values())


# Appended AFTER the strict-RAG refusal above (never shown alone, never shown
# alongside a confident/grounded answer) when a `needs_review` chunk -- not
# yet `review_status=approved` -- was found that might be relevant. See
# `unverified_source_disclosure` below for why the excerpt itself is never
# translated or paraphrased. Only hindi/english/hinglish are hand-authored;
# every other language falls back to english, same rule as
# `no_verified_context_message`.
UNVERIFIED_SOURCE_HEADER_MESSAGES: dict[str, str] = {
    "hindi": "हालाँकि, एक संभावित रूप से संबंधित स्रोत मिला है, जिसे अभी तक किसी व्यक्ति ने आधिकारिक पाठ से मिलाकर सत्यापित नहीं किया है:",
    "english": "However, a possibly related source was found that has NOT yet been checked by a human reviewer against the official text:",
    "hinglish": "Lekin, ek possibly related source mila hai jise abhi tak kisi insaan ne official text se milaakar verify nahi kiya hai:",
}
UNVERIFIED_SOURCE_FOOTER_MESSAGES: dict[str, str] = {
    "hindi": "यह पुष्टि किया गया कानून नहीं है। इस पर भरोसा करने से पहले कृपया किसी योग्य अधिवक्ता से पुष्टि करें।",
    "english": "This is NOT confirmed law. Please verify it independently or consult a qualified advocate before relying on it.",
    "hinglish": "Yeh confirmed law nahi hai. Isper bharosa karne se pehle kisi qualified advocate se zaroor confirm karein.",
}


def unverified_source_disclosure(language: str | None, source_label: str, excerpt: str) -> str:
    """Composes the disclosure block appended after the strict-RAG refusal.

    `excerpt` is quoted VERBATIM from the retrieved `needs_review` chunk --
    this function never summarizes, translates, or otherwise rewrites it, and
    no LLM call sits between the chunk and this string. That is deliberate:
    the entire point of `review_status=needs_review` is that nobody has yet
    confirmed this text matches the official source, so composing a fresh
    sentence ABOUT it (which an LLM would inevitably do with some paraphrase
    or added confidence) risks stating something the source doesn't actually
    say. Quoting it exactly, with the source named and a warning on both
    sides, is the only presentation that can't misrepresent unverified text.

    `source_label` should name the document/Act, not a bare chunk id, so a
    user (or the human reviewer this excerpt should send them looking for)
    knows what to check.
    """
    header = UNVERIFIED_SOURCE_HEADER_MESSAGES.get(
        (language or "").strip().lower(), UNVERIFIED_SOURCE_HEADER_MESSAGES["english"]
    )
    footer = UNVERIFIED_SOURCE_FOOTER_MESSAGES.get(
        (language or "").strip().lower(), UNVERIFIED_SOURCE_FOOTER_MESSAGES["english"]
    )
    return f"{header}\n\n\U0001f4c4 {source_label.strip()}\n\n“{excerpt.strip()}”\n\n⚠️ {footer}"


# What this assistant tells a user who asks what it can do. Fixed, reviewed
# text -- never generated -- for the same reason the strict-RAG refusal is:
# a model asked to describe its own capabilities invents them, and an invented
# capability in a legal tool is a promise the product then breaks. Every line
# below corresponds to a workflow registered in `app/chatops/registry.py` or
# to a branch of `ChatService._dispatch_conversation_intent`.
# `{count}` is filled from the live template registry, so the number can never
# drift from the templates actually installed.
#
# Post-Phase-3 hardening (Phase 2, milestone E) rewrote this text. Two defects
# in the previous version, both confirmed against a real session:
#
# 1. It promised "I cite the Act and section behind every answer." That is not
#    something this system can guarantee and not what it does. A citation
#    carries an Act/section only when the retrieved chunk's governance metadata
#    records them (`LegalCitationEngine.citations_from_chunks`), and a corpus
#    document that is a circular, a handbook or a judgment often records
#    neither. Promising it on every answer made the honest cases -- an answer
#    grounded in a source with no section identity -- look like a malfunction.
# 2. It described five capabilities. The chat surface actually registers
#    twenty-nine workflows: saved drafts, draft editing/export, document
#    review/comparison/timeline/risky-clause extraction, cases, evidence
#    annexures, lawyer-ready summaries, notarization preparation/status/
#    verification, downloads, background jobs and preferences. A user who was
#    never told those exist cannot ask for them.
#
# Deliberately NOT listed: the admin and notary-queue workflows
# (`admin_knowledge_base`, `admin_analytics`, `admin_sources`, `notary_queue`,
# `notary_admin`). Each declares a `required_role` and is refused for an
# ordinary account, so naming them here would advertise a capability the
# reader cannot use and would disclose the operator surface to every user.
CAPABILITY_OVERVIEW_MESSAGES: dict[str, str] = {
    "english": (
        "Here's what I can help you with:\n\n"
        "- **Answer legal questions** about Indian law, grounded only in verified documents in my "
        "knowledge base. Where the source records an Act and section, I name them and show the page "
        "evidence; where it doesn't, I say what the source is and that the provision isn't identified. "
        "If I have no verified source for your question, I say so instead of answering from memory.\n"
        "- **Draft legal documents** — {count} templates including police complaints, legal notices, "
        "affidavits, RTI applications, rent agreements and consumer complaints. I ask for the details "
        "one at a time, then produce the document.\n"
        "- **Manage your drafts** — list your saved drafts, pick one back up, change any detail by "
        "telling me what to change, and download it as PDF, DOCX, TXT or RTF.\n"
        "- **Work on documents you upload** — summarise them, flag risky or missing clauses, review a "
        "contract, compare two versions, or build a dated timeline of events from them.\n"
        "- **Organise a matter** — keep your documents and drafts under a case, arrange your evidence "
        "into a numbered annexure list, and produce a lawyer-ready summary of the whole file.\n"
        "- **Prepare a document for notarization** — I assemble the checklist and the paperwork, and "
        "can show the status of a submission or verify a notarized document's code. I do not notarize "
        "anything myself, and I can't stamp, register or attest a document; a notary or the issuing "
        "authority does that.\n"
        "- **Your account** — your downloads, your background jobs, your language and answer-style "
        "preferences, and clearing or deleting your own conversation data.\n"
        "- **Suggest the kind of advocate** to approach — the specialisation your matter calls for.\n\n"
        "Some operations are limited to the account they belong to, and a few are restricted to "
        "verified professional accounts. Everything I produce is a drafting aid and must be reviewed "
        "by a qualified advocate before you file or send it.\n\n"
        "Just describe your situation in your own words, in whichever language you're comfortable with."
    ),
    "hinglish": (
        "Main aapki in cheezon mein madad kar sakta hoon:\n\n"
        "- **Legal sawaalon ke jawab** — Indian law ke baare mein, sirf verified documents ke aadhar "
        "par. Jahan source mein Act aur Section darj hai, wahan main woh naam bhi batata hoon aur page "
        "evidence bhi dikhata hoon; jahan darj nahi hai, wahan saaf keh deta hoon ki source kaun sa hai "
        "aur provision identify nahi hui. Agar verified source hi na ho, to main memory se jawab dene "
        "ke bajaye saaf mana kar deta hoon.\n"
        "- **Legal documents draft karna** — {count} templates hain, jaise police complaint, legal "
        "notice, affidavit, RTI application, rent agreement, consumer complaint. Main ek-ek karke "
        "details poochta hoon, phir document bana deta hoon.\n"
        "- **Aapke drafts sambhalna** — saved drafts ki list, kisi bhi draft ko dobara continue karna, "
        "koi bhi detail badalna (bas bata dijiye kya badalna hai), aur PDF, DOCX, TXT ya RTF mein "
        "download karna.\n"
        "- **Aapke upload kiye documents par kaam** — summary, risky ya missing clauses, contract "
        "review, do versions ka comparison, ya unse events ki dated timeline.\n"
        "- **Poore maamle ko organise karna** — documents aur drafts ko ek case ke andar rakhna, "
        "evidence ko numbered annexure list mein lagana, aur poori file ka lawyer-ready summary banana.\n"
        "- **Notarization ki taiyari** — checklist aur paperwork main taiyar kar deta hoon, submission "
        "ka status dikha sakta hoon, aur notarized document ka code verify kar sakta hoon. Main khud "
        "notarize nahi karta — stamp, registration ya attestation notary ya jaari karne wale "
        "authority ka kaam hai.\n"
        "- **Aapka account** — downloads, background jobs, bhasha aur jawab ke style ki preferences, "
        "aur apna conversation data clear ya delete karna.\n"
        "- **Kis tarah ke advocate se milna chahiye** — aapke maamle ke liye kaun si specialisation "
        "chahiye.\n\n"
        "Kuch operations sirf unhi ke account tak seemit hain jinke woh hain, aur kuch sirf verified "
        "professional accounts ke liye hain. Jo bhi main banata hoon woh drafting aid hai — file ya "
        "send karne se pehle kisi qualified advocate se review zaroor karwaiye.\n\n"
        "Aap apni problem apne shabdon mein, apni bhasha mein likh dijiye."
    ),
    "hindi": (
        "मैं आपकी इन चीज़ों में मदद कर सकता हूँ:\n\n"
        "- **कानूनी सवालों के जवाब** — भारतीय कानून के बारे में, केवल सत्यापित दस्तावेज़ों के आधार पर। जहाँ स्रोत में "
        "अधिनियम और धारा दर्ज है, वहाँ मैं उनका नाम बताता हूँ और पृष्ठ-साक्ष्य भी दिखाता हूँ; जहाँ दर्ज नहीं है, वहाँ "
        "स्पष्ट कह देता हूँ कि स्रोत कौन-सा है और प्रावधान चिह्नित नहीं है। यदि सत्यापित स्रोत ही उपलब्ध न हो, तो मैं "
        "स्मृति से उत्तर देने के बजाय स्पष्ट मना कर देता हूँ।\n"
        "- **कानूनी दस्तावेज़ तैयार करना** — {count} टेम्पलेट उपलब्ध हैं, जैसे पुलिस शिकायत, लीगल नोटिस, शपथ पत्र, "
        "RTI आवेदन, किरायानामा और उपभोक्ता शिकायत। मैं एक-एक करके विवरण पूछता हूँ, फिर दस्तावेज़ बना देता हूँ।\n"
        "- **आपके ड्राफ्ट संभालना** — सहेजे गए ड्राफ्ट की सूची, किसी भी ड्राफ्ट को दोबारा आगे बढ़ाना, कोई भी विवरण "
        "बदलना, और PDF, DOCX, TXT या RTF में डाउनलोड करना।\n"
        "- **आपके अपलोड किए दस्तावेज़ों पर काम** — सारांश, जोखिम भरे या छूटे हुए क्लॉज़, अनुबंध की समीक्षा, दो "
        "संस्करणों की तुलना, या उनसे घटनाओं की तिथिवार समयरेखा।\n"
        "- **पूरे मामले को व्यवस्थित करना** — दस्तावेज़ और ड्राफ्ट को एक केस के अंतर्गत रखना, साक्ष्यों को क्रमांकित "
        "अनुलग्नक सूची में लगाना, और पूरी फ़ाइल का अधिवक्ता-तैयार सारांश बनाना।\n"
        "- **नोटरीकरण की तैयारी** — चेकलिस्ट और कागज़ात मैं तैयार कर देता हूँ, प्रस्तुत आवेदन की स्थिति दिखा सकता "
        "हूँ, और नोटरीकृत दस्तावेज़ का कोड सत्यापित कर सकता हूँ। मैं स्वयं नोटरीकरण नहीं करता — मुहर, पंजीकरण या "
        "अनुप्रमाणन नोटरी अथवा जारीकर्ता प्राधिकारी का कार्य है।\n"
        "- **आपका खाता** — डाउनलोड, पृष्ठभूमि कार्य, भाषा एवं उत्तर-शैली की वरीयताएँ, तथा अपना वार्तालाप डेटा "
        "हटाना या मिटाना।\n"
        "- **किस प्रकार के अधिवक्ता से संपर्क करें** — आपके मामले के लिए कौन-सी विशेषज्ञता चाहिए।\n\n"
        "कुछ कार्य केवल उसी खाते तक सीमित हैं जिसके वे हैं, और कुछ केवल सत्यापित व्यावसायिक खातों के लिए हैं। मैं जो "
        "भी तैयार करता हूँ वह प्रारूपण-सहायक है — दाखिल या प्रेषित करने से पूर्व किसी योग्य अधिवक्ता से समीक्षा अवश्य "
        "करा लें।\n\n"
        "आप अपनी समस्या अपने शब्दों में, अपनी भाषा में लिख दीजिए।"
    ),
}

def capability_overview(language: str | None, template_count: int) -> str:
    """The capability description in `language`, falling back to English for
    any language without one (never to Hindi -- same reasoning as
    `no_verified_context_message`)."""
    key = (language or "english").strip().lower()
    template = CAPABILITY_OVERVIEW_MESSAGES.get(key, CAPABILITY_OVERVIEW_MESSAGES["english"])
    return template.format(count=template_count)


# What the assistant says when a turn is nothing but a request to change the
# language it replies in ("Mujhe simple Hindi mein jawab diya karo"). Fixed,
# reviewed text in each language rather than a generated reply, for the same
# reason the capability overview is: the acknowledgement itself has to be IN
# the language just asked for, and a model asked to acknowledge a preference
# will happily also start answering the previous legal question from memory.
LANGUAGE_PREFERENCE_ACKNOWLEDGEMENTS: dict[str, str] = {
    "hindi": "ठीक है—अब मैं हिंदी में जारी रखूँगा।",
    "hinglish": "Theek hai—ab main Hinglish mein continue karunga.",
    "english": "Okay—I will continue in English.",
}

# Appended only when a workflow is mid-collection, so the user knows the
# switch did not throw away what they had already supplied.
LANGUAGE_PREFERENCE_WORKFLOW_SUFFIXES: dict[str, str] = {
    "hindi": " कृपया पिछले सवाल का जवाब दें; आपकी दी हुई जानकारी सुरक्षित है।",
    "hinglish": " Pichhle sawal ka jawab dein; aapki di hui information safe hai.",
    "english": " Please answer the previous question; the information you supplied is preserved.",
}


def language_preference_acknowledgement(language: str, has_active_workflow: bool = False) -> str:
    """The fixed acknowledgement for a language switch, in `language`.

    A language with no reviewed wording of its own gets an English sentence
    naming it, never a Hindi one -- same reasoning as
    `no_verified_context_message`.
    """
    key = (language or "english").strip().lower()
    acknowledgement = LANGUAGE_PREFERENCE_ACKNOWLEDGEMENTS.get(key, f"Okay—I will continue in {key.title()}.")
    if not has_active_workflow:
        return acknowledgement
    return acknowledgement + LANGUAGE_PREFERENCE_WORKFLOW_SUFFIXES.get(key, " Please continue.")


# The 22 languages of the Eighth Schedule to the Constitution of India, plus
# English and Hinglish (romanized Hindi -- how a large share of Indian users
# actually type, and detected separately from Hindi so replies can match).
SUPPORTED_LANGUAGES = {
    "english",
    "hinglish",
    "assamese",
    "bengali",
    "bodo",
    "dogri",
    "gujarati",
    "hindi",
    "kannada",
    "kashmiri",
    "konkani",
    "maithili",
    "malayalam",
    "manipuri",
    "marathi",
    "nepali",
    "odia",
    "punjabi",
    "sanskrit",
    "santali",
    "sindhi",
    "tamil",
    "telugu",
    "urdu",
}

# Fixed, reviewed strings for the controlled General Knowledge (GK) fallback
# (`app.core.gk_fallback`) -- shown ONLY when the strict-RAG guardrail found
# no verified Knowledge Base document and the LLM answered from general
# legal knowledge instead. Never shown alone: `label` always precedes and
# `disclaimer` always follows the generated body, so the answer can never be
# mistaken for a Knowledge-Base-grounded one. Only hindi/english/hinglish are
# hand-authored; every other language falls back to english, same rule as
# `no_verified_context_message`.
GENERAL_KNOWLEDGE_LABEL_MESSAGES: dict[str, str] = {
    "english": "**General Legal Knowledge** (not from the verified Knowledge Base)",
    "hindi": "**सामान्य कानूनी जानकारी** (सत्यापित Knowledge Base से नहीं)",
    "hinglish": "**General Legal Knowledge** (verified Knowledge Base se nahi)",
}

GENERAL_KNOWLEDGE_DISCLAIMER_MESSAGES: dict[str, str] = {
    "english": (
        "This answer is based on general legal knowledge and has NOT been verified against this "
        "system's Legal Knowledge Base or the current text of the law. It may be incomplete, "
        "outdated, or not applicable to your specific facts. Please verify independently or consult "
        "a qualified advocate before relying on it."
    ),
    "hindi": (
        "यह उत्तर सामान्य कानूनी जानकारी पर आधारित है और इसे इस सिस्टम के Legal Knowledge Base या कानून के "
        "वर्तमान पाठ से सत्यापित नहीं किया गया है। यह अधूरा, पुराना, या आपकी विशिष्ट स्थिति पर लागू न होने वाला "
        "हो सकता है। इस पर भरोसा करने से पहले कृपया स्वतंत्र रूप से पुष्टि करें या किसी योग्य अधिवक्ता से सलाह लें।"
    ),
    "hinglish": (
        "Yeh jawab general legal knowledge par based hai aur isse system ke Legal Knowledge Base ya "
        "kanoon ke current text se verify nahi kiya gaya hai. Yeh incomplete, purana, ya aapki "
        "specific situation par apply na hone wala ho sakta hai. Isper bharosa karne se pehle khud "
        "verify karein ya kisi qualified advocate se salah lein."
    ),
}


def general_knowledge_label(language: str | None) -> str:
    """The fixed "General Legal Knowledge" header, in `language`."""
    key = (language or "english").strip().lower()
    return GENERAL_KNOWLEDGE_LABEL_MESSAGES.get(key, GENERAL_KNOWLEDGE_LABEL_MESSAGES["english"])


def general_knowledge_disclaimer(language: str | None) -> str:
    """The fixed unverified-answer disclaimer, in `language`."""
    key = (language or "english").strip().lower()
    return GENERAL_KNOWLEDGE_DISCLAIMER_MESSAGES.get(key, GENERAL_KNOWLEDGE_DISCLAIMER_MESSAGES["english"])


ALLOWED_UPLOAD_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
    ".html",
    ".htm",
    ".json",
    ".csv",
    ".rtf",
    ".odt",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
}
