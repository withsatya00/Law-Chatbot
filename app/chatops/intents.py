"""Multilingual intent vocabulary for the conversational orchestrator.

Pattern-first by design. A regex that matched is evidence; a model that
guessed is not, and routing a user into a legal workflow on a guess is how
someone ends up filling in a police complaint when they asked about an RTI.
The `ConversationIntentClassifier` still runs as a fallback for anything
these patterns do not recognise.

COVERAGE RULE: every phrase set below carries, at minimum, English, Hindi
(Devanagari), and romanised Hinglish, because that is what Indian users
actually type. Native-script vocabulary for the other Eighth Schedule
languages is added where the verb is short and unambiguous; where it is not,
the language-name request (`extract_requested_language`) plus an English or
Hinglish verb still routes correctly, and the REPLY comes back in the user's
own language regardless.

Kept as data, not code: adding a phrase is a one-line change here, never a
change to the orchestrator.
"""

import re

# --- shared building blocks -------------------------------------------------

# "make / create / prepare / draft / generate", across scripts.
_MAKE = (
    r"banao|banana|bana\s*do|banwana|banwao|bnao|likho|likhna|likh\s*do|likhwana"
    r"|draft|drafting|generate|create|make|prepare|write|compose|file|register"
    r"|चाहिए|बनाओ|बनाना|बना\s*दो|लिखो|लिखना|लिख\s*दो|तैयार|तैयारी|दर्ज"
    r"|तयार|बनवा|लिहा|लिहून"                 # Marathi
    r"|બનાવ|લખો|તૈયાર"                        # Gujarati
    r"|তৈরি|লিখুন|লেখো|খসড়া"                  # Bengali/Assamese
    r"|ਬਣਾ|ਲਿਖ|ਤਿਆਰ"                          # Punjabi
    r"|ତିଆରି|ଲେଖ|ପ୍ରସ୍ତୁତ"                    # Odia
    r"|உருவாக்க|எழுது|தயார்"                   # Tamil
    r"|తయారు|రాయ|సిద్ధ"                        # Telugu
    r"|ಮಾಡಿ|ಬರೆ|ಸಿದ್ಧ"                         # Kannada
    r"|ഉണ്ടാക്ക|എഴുത|തയ്യാറാക്ക"                # Malayalam
    r"|بنائیں|لکھ|تیار"                        # Urdu
)

# "show / open / see / list / find"
_SHOW = (
    r"dikhao|dikha\s*do|dikhaiye|batao|bata\s*do|kholo|khol\s*do|dekhna|dekho"
    r"|show|open|list|see|view|find|fetch|display|get|check|status"
    r"|दिखाओ|दिखाएं|बताओ|खोलो|खोल\s*दो|देखना|देखो|सूची|स्थिति"
    r"|दाखवा|उघडा"                             # Marathi
    r"|બતાવ|ખોલ"                               # Gujarati
    r"|দেখাও|খোল|তালিকা"                        # Bengali/Assamese
    r"|ਦਿਖਾ|ਖੋਲ੍ਹ"                              # Punjabi
    r"|ଦେଖାନ୍ତୁ|ଖୋଲ"                            # Odia
    r"|காட்ட|திற"                              # Tamil
    r"|చూపు|తెరు"                               # Telugu
    r"|ತೋರಿಸು|ತೆರೆ"                             # Kannada
    r"|കാണിക്ക|തുറക്ക"                          # Malayalam
    r"|دکھائ|کھول"                              # Urdu
)


# Latin alternatives inside `_MAKE`/`_SHOW` must be word-bounded. Without
# this, "view" matched the middle of "Review", and every message containing
# "Review this PDF..." was routed into the saved-drafts workflow.
_MAKE = rf"(?:\b(?:{_MAKE})\b)"
_SHOW = rf"(?:\b(?:{_SHOW})\b)"

# An explicit request to DO something. Topical vocabulary alone is never
# enough to start a workflow: "cyber fraud kya hota hai" is a question and
# "I lost money in an online cyber fraud" is a situation report -- both want
# a grounded legal answer, not to be pulled into an emergency form. Only a
# message that also asks for an action gets routed to a workflow.
_ACTION_CUE = re.compile(
    r"\b(?:"
    r"report|file|register|lodge|submit|complain|raise|start|begin|initiate|open|help\s*me"
    r"|i\s*(?:want|need|would\s*like)|can\s*you|please|prepare|draft|generate|create|make|do\s*it"
    r"|karna|karni|karwana|karana|karo|kro|kar\s*do|karn[ae]\s*hai|chahiye|banao|bana\s*do|banwana"
    r"|likhna|likho|dikhao|batao|shuru|madad"
    r")\b"
    r"|करना|करनी|कराना|करो|कर\s*दो|चाहिए|बनाओ|बनाना|लिखो|दिखाओ|बताओ|शुरू|मदद|दर्ज"
    r"|पाहिजे|करायच"                       # Marathi
    r"|જોઈએ|કરવું"                          # Gujarati
    r"|করতে|চাই|দরকার"                       # Bengali/Assamese
    r"|ਕਰਨਾ|ਚਾਹੀਦਾ"                          # Punjabi
    r"|କରିବାକୁ|ଦରକାର"                        # Odia
    r"|செய்ய|வேண்டும்"                        # Tamil
    r"|చేయాలి|కావాలి"                          # Telugu
    r"|ಮಾಡಬೇಕು|ಬೇಕು"                          # Kannada
    r"|ചെയ്യണം|വേണം"                          # Malayalam
    r"|کرنا|چاہیے",                          # Urdu
    re.IGNORECASE,
)


def wants_action(message: str) -> bool:
    """Whether `message` asks for something to be DONE, not merely discussed."""
    return bool(_ACTION_CUE.search(message or ""))


def _pattern(*alternatives: str) -> re.Pattern[str]:
    return re.compile("|".join(alternatives), re.IGNORECASE)


# --- capability vocabularies ------------------------------------------------

# Notarization. Deliberately matches the user's own framing ("notary karani
# hai") even though the product can only PREPARE and submit for review -- the
# workflow is what corrects the expectation, in words, on the first turn.
NOTARIZATION_PREPARE = _pattern(
    rf"\bnotar(?:y|ize|ise|ised|ized|ization|isation)\b.*(?:{_MAKE})",
    rf"(?:{_MAKE}).*\bnotar(?:y|ize|ise|ization|isation)\b",
    r"\bnotar(?:y|ize|ise|ization|isation)\b.*\b(?:ready|prepare|preparation|karani|karna|karwana|chahiye)\b",
    r"notary\s*(?:ke\s*liye|के\s*लिए)",
    r"नोटरी.*(?:करानी|करना|कराना|तैयार|चाहिए)",
    r"नोटरी\s*के\s*लिए",
    r"\battest(?:ation)?\b.*\b(?:need|want|chahiye|karani)\b",
    r"\bstamp\s*paper\b.*\bnotar",
)
NOTARIZATION_VERIFY = _pattern(
    r"\bverify\b.*\b(?:document|notari[sz]ation|certificate|token|qr)\b",
    r"\b(?:document|notari[sz]ation)\b.*\bverify\b",
    r"verification\s*(?:code|token)",
    r"(?:document|दस्तावेज़|दस्तावेज).*(?:verify|सत्यापित|सत्यापन)",
    r"सत्यापित\s*कर|सत्यापन\s*कर",
    r"\bqr\b.*\b(?:scan|check|verify)\b",
)
NOTARIZATION_STATUS = _pattern(
    r"\bnotar\w*\b.*\bstatus\b", r"\bstatus\b.*\bnotar\w*\b",
    r"नोटरी.*(?:स्थिति|status)", r"notary.*(?:kya hua|kahan tak|kitna hua)",
    r"\bmy\b.*\bnotari[sz]ation\b.*\brequest",
)
NOTARY_QUEUE = _pattern(
    r"\bpending\b.*\b(?:review|request)s?\b",
    r"\breview\s*queue\b", r"\bmy\b.*\brequests?\b.*\breview\b",
    r"(?:meri|mere|मेरी|मेरे).*(?:pending|लंबित).*(?:request|अनुरोध)",
    r"\bstart\s*(?:the\s*)?review\b", r"review\s*(?:start|शुरू)",
    r"\b(?:approve|reject)\b.*\brequest\b",
    r"request.*(?:approve|reject|मंज़ूर|अस्वीकार)",
)
NOTARY_ADMIN = _pattern(
    r"\bpending\b.*\bnotary\s*accounts?\b",
    r"\bverify\b.*\bnotary\s*account\b", r"\bnotary\s*account\b.*\bverify\b",
    r"\brevoke\b.*\bnotary\b", r"\bnotar\w*\b.*\baudit\s*log\b",
    r"\baudit\s*log\b.*\bnotar\w*\b",
    r"\bfailed\b.*\b(?:verification|signature|attempt)s?\b",
    r"नोटरी\s*(?:खाता|अकाउंट).*(?:सत्यापित|verify)",
)

# Phase 2 legal workflows.
CYBER_FRAUD = _pattern(
    r"\bcyber\s*(?:crime|fraud)\b", r"\bonline\s*fraud\b",
    r"\b(?:upi|otp|phishing)\b.*\b(?:fraud|scam|report)\b",
    r"\bfraud\b.*\breport\b", r"\breport\b.*\bfraud\b",
    r"साइबर\s*(?:अपराध|फ्रॉड|धोखा)", r"ऑनलाइन\s*(?:ठगी|धोखा|फ्रॉड)",
    r"सायबर", r"સાયબર", r"সাইবার", r"சைபர்", r"సైబర్", r"ಸೈಬರ್",
    r"\b1930\b", r"cybercrime\.gov\.in",
)
JURISDICTION = _pattern(
    r"\bjurisdiction\b", r"\bwhich\s*(?:court|police\s*station|forum)\b",
    r"\bkahan\b.*\b(?:file|complaint|case)\b",
    r"\bkaunsa\b.*\b(?:court|thana|police)\b",
    r"क्षेत्राधिकार", r"कौन\s*सा\s*(?:न्यायालय|कोर्ट|थाना)",
    r"कहाँ\s*(?:शिकायत|केस|मुकदमा)", r"कहां\s*(?:शिकायत|केस)",
)
LAWYER_SUMMARY = _pattern(
    r"\blawyer[- ]?ready\b", r"\bsummary\b.*\b(?:lawyer|advocate|vakil)\b",
    r"\b(?:lawyer|advocate|vakil)\b.*\bsummary\b",
    r"वकील.*(?:सारांश|समरी)", r"(?:सारांश|समरी).*वकील",
    r"vakil.*(?:summary|saransh)",
)
EVIDENCE_ORGANIZE = _pattern(
    r"\bevidence\b.*\b(?:organi[sz]e|arrange|list|prepare)\b",
    # Verb-first is at least as common as noun-first ("organise my evidence"),
    # and the noun-first-only pattern silently missed it.
    r"\b(?:organi[sz]e|arrange|prepare|index)\b[^.?!]{0,16}\bevidence\b",
    r"\bannexure\b", r"\bexhibit\s*list\b",
    r"\bsabut\b[^.?!]{0,16}\b(?:organize|arrange|list|taiyar)\b",
    r"सबूत.*(?:व्यवस्थित|तैयार|सूची)", r"अनुलग्नक",
)
# Case files. "case" alone is far too common in ordinary legal English
# ("in that case", "a bail case") to route on, so every pattern pairs it
# with an explicit management verb or an unmistakable case-file noun.
_CASE_NOUN = r"(?:case|matter|mukadma|mukaddama|केस|मुकदम[ाे]|मामल[ेा])"
CASE_CREATE = _pattern(
    rf"\b(?:create|add|start|open|register|file|new)\b\s+(?:a\s+|new\s+|nay[ai]\s+)?\b{_CASE_NOUN}\b",
    rf"\b{_CASE_NOUN}\b\s*(?:file\s*)?(?:banao|bana\s*do|create\s*karo|add\s*karo|shuru\s*karo)",
    rf"(?:नया|नई)\s*{_CASE_NOUN}\s*(?:बनाओ|दर्ज|जोड़)",
    rf"{_CASE_NOUN}\s*(?:दर्ज\s*कर|बनाओ|जोड़ो)",
)
CASE_LIST = _pattern(
    # PLURAL only for the bare possessive: "my case" is ordinary English in
    # "check the jurisdiction for my case", which is a jurisdiction question,
    # not a request for a case list.
    r"\b(?:my|meri|mere|all)\s+(?:cases|matters|mukadme)\b",
    r"(?:मेरे|मेरी|सभी)\s*(?:केस|मुकदमे|मामले)",
    rf"\b(?:list|show|dikhao|batao|दिखाओ)\b[^.?!]{{0,16}}\b{_CASE_NOUN}s?\b",
    rf"\b{_CASE_NOUN}s?\b[^.?!]{{0,16}}\b(?:list|dikhao|दिखाओ)\b",
    r"\bupcoming\s+hearings?\b", r"\bnext\s+hearing\b", r"\bhearing\s+reminders?\b",
    r"(?:आगामी|अगली)\s*(?:सुनवाई|पेशी)", r"\b(?:agli|agli)\s*(?:sunwai|peshi)\b",
)
CASE_MANAGE = _pattern(
    rf"\b(?:delete|close|update|edit|change)\b[^.?!]{{0,20}}\b{_CASE_NOUN}\b",
    rf"\b{_CASE_NOUN}\b[^.?!]{{0,20}}\b(?:delete|close|update|band\s*karo|hata\s*do|बंद\s*कर|हटा)\b",
    r"\badd\b[^.?!]{0,20}\b(?:hearing|note|task|reminder)\b",
    r"\b(?:hearing|note|task)\b[^.?!]{0,16}\b(?:add|jodo|जोड़)\b",
    r"\b(?:acknowledge|ack)\b[^.?!]{0,20}\breminder\b",
    r"\battach\b[^.?!]{0,24}\b(?:case|document|evidence)\b",
    r"(?:सुनवाई|नोट|कार्य)\s*(?:जोड़|दर्ज)",
)
CASE_TIMELINE = _pattern(
    r"\b(?:case\s*)?timeline\b", r"\bchronology\b",
    r"(?:केस|मामले).*(?:समयरेखा|टाइमलाइन)", r"समय\s*रेखा",
)

# Draft management (the drafting CONVERSATION itself is handled by the
# existing `DraftConversationEngine`; these are the management verbs around
# it that previously needed a sidebar panel).
SAVED_DRAFTS = _pattern(
    rf"(?:{_SHOW}).*\b(?:saved\s*)?drafts?\b",
    rf"\b(?:saved\s*)?drafts?\b.*(?:{_SHOW})",
    r"\bmy\s*drafts?\b", r"(?:meri|mere|मेरी|मेरे)\s*(?:draft|ड्राफ्ट|मसौद)",
    r"\bpending\s*drafts?\b", r"\bresume\b.*\bdraft\b",
    # "meri save ki hui draft dikhao" allows descriptive words between
    # the possessive and noun. Unicode escapes keep this source robust in
    # Windows shells whose display encoding is not UTF-8.
    r"(?:\u092e\u0947\u0930\u0940|\u092e\u0947\u0930\u0947).{0,40}"
    r"(?:\u0921\u094d\u0930\u093e\u092b\u094d\u091f|\u092e\u0938\u094c\u0926).{0,24}"
    r"(?:\u0926\u093f\u0916\u093e\u0913|\u0926\u093f\u0916\u093e\u090f\u0902)",
)
DRAFT_EXPORT = _pattern(
    # "PDF me", "DOCX mein" -- the Hinglish postposition, which must follow
    # the format word IMMEDIATELY. Written as a bare alternative it matched
    # the English pronoun in "tell me the risky clauses".
    r"\b(?:pdf|docx|word|txt|rtf)\s+(?:me|mein|म[ेैं])\b",
    r"\b(?:pdf|docx|word|txt|rtf)\b[^.?!]{0,40}?\b(?:download|export|save|chahiye|banao|de\s*do)\b",
    r"\b(?:download|export)\b.*\b(?:pdf|docx|word|txt|rtf|draft|document)\b",
    r"(?:pdf|docx|word).*(?:में|मे|चाहिए|डाउनलोड)",
    r"डाउनलोड\s*कर", r"\bdownload\s*kar",
)
DRAFT_VERSIONS = _pattern(
    r"\bversion\s*history\b", r"\bprevious\s*versions?\b",
    r"\bcompare\b.*\bversions?\b", r"\bversions?\b.*\bcompare\b",
    r"संस्करण", r"पिछला\s*संस्करण",
)
# Draft LIFECYCLE management. Deliberately requires a management verb next
# to the word "draft": "Review this PDF and draft a legal notice" names a
# draft and a review and is neither -- it is a request to write something,
# and pulling it into draft administration would be exactly wrong.
_DRAFT_NOUN = r"(?:drafts?|ड्राफ्ट|मसौद[ाेों]+)"
# Verbs that can only mean draft administration. Near the word "draft" in
# either order, these are decisive.
#
# QA session 2026-09-24 ("BUG-105"): "finalize"/"finalise" used to be listed
# here, on the same "can only mean draft administration" assumption as
# "approve"/"lock"/"delete" -- but unlike those, "finalize the draft" is at
# least as commonly said to an ACTIVELY-COLLECTING draft, meaning "please
# generate/complete it now with what I've already given you", as it is a
# request to lock/complete a previously-SAVED one. `interrupts_draft` (this
# module's caller, `app/chatops/registry.py`) has no notion of whether the
# session's current draft is still `collecting` or already `preview`/
# complete, so it can't disambiguate the two meanings -- it just interrupted
# the in-progress draft unconditionally. Live-reproduced: "Please finalize
# the draft with the information already provided" (asked while a
# cheque-bounce notice was still mid-collection) got routed to
# `DraftManagementWorkflow`, which correctly reported "I could not find a
# saved draft to work with" -- true for ITS purpose, but the in-progress
# draft's own conversation state was lost in the process. None of the other
# verbs here ("approve", "lock", "delete", ...) have this same "also means
# 'please finish writing it'" ambiguity, so only this one needed to move.
_DRAFT_VERB_STRONG = (
    r"approve|lock|unlock|rollback|roll\s*back|revert|restore"
    r"|duplicate|copy|clone|delete|remove|discard|compare|translate"
    r"|मंज़ूर|मंजूर|स्वीकृत|लॉक|अनलॉक|वापस|प्रतिलिपि|हटा|मिटा|तुलना|अनुवाद"
)
# Verbs that are ALSO ordinary English about documents in general. "Review
# this PDF and draft a legal notice" contains "review" and "draft" and is a
# request to WRITE something -- so a weak verb only counts when the draft is
# explicitly the user's own existing one ("review my draft").
_DRAFT_VERB_WEAK = r"review|open|resume|show|samiksha|समीक्षा|खोल"
_DRAFT_POSSESSIVE = r"(?:my|this|that|the|meri|mera|mere|is|us|मेरा|मेरी|मेरे|इस|उस)"
DRAFT_MANAGE = _pattern(
    rf"\b(?:{_DRAFT_VERB_STRONG})\b[^.?!]{{0,24}}?\b{_DRAFT_NOUN}\b",
    rf"\b{_DRAFT_NOUN}\b[^.?!]{{0,24}}?\b(?:{_DRAFT_VERB_STRONG})\b",
    rf"\b(?:{_DRAFT_VERB_WEAK})\b\s+{_DRAFT_POSSESSIVE}\s+{_DRAFT_NOUN}\b",
    rf"\b{_DRAFT_NOUN}\s*(?:ko|ki|को|की|का)?\s*(?:{_DRAFT_VERB_WEAK})\b",
    rf"\b{_DRAFT_NOUN}\s+(?:versions?|history|इतिहास|संस्करण)\b",
    # Devanagari verbs are written WITHOUT `\b`: Python's `\b` is defined in
    # terms of `\w`, which excludes Indic combining marks, so `हटा\b` never
    # matches "हटा " -- the same failure `app/chatops/confirm.py` documents
    # for every Indic affirmative. Left bounded, "ड्राफ्ट हटा दो" silently
    # did not route.
    r"(?:ड्राफ्ट|मसौद[ाेों]+)\s*(?:को|की|का)?\s*"
    r"(?:हटा|मिटा|डिलीट|मंज़ूर|मंजूर|स्वीकृत|लॉक|अनलॉक|अनुवाद|तुलना|वापस|प्रतिलिपि|खोल|दिखा)",
    r"(?:हटा|मिटा|डिलीट|मंज़ूर|मंजूर|लॉक|अनलॉक|अनुवाद|तुलना)\s*(?:दो|दीजिए|करो|कर)?\s*"
    r"(?:ड्राफ्ट|मसौद[ाेों]+)",
    r"\bversion\s*history\b",
    r"\bcompare\b[^.?!]{0,20}\bversions?\b",
    r"संस्करण\s*इतिहास",
)
DRAFT_DELETE = _pattern(
    r"\bdelete\b.*\bdraft\b", r"\bdraft\b.*\b(?:delete|remove)\b",
    r"ड्राफ्ट.*(?:हटा|मिटा|डिलीट)", r"draft.*(?:hata|mita|delete\s*kar)",
)

# Document handling.
DOCUMENT_VERIFY = NOTARIZATION_VERIFY
# Asking for a document to be summarised or its risks listed. Both patterns
# are checked against `_ALSO_WANTS_A_DRAFT` by the workflow: a message that
# ALSO asks for a draft is a multi-intent chain owned by
# `app/services/workflow_orchestrator.py`, and stealing it here would break
# the "review the PDF, then draft from it" flow.
DOCUMENT_SUMMARY = _pattern(
    r"\bsummar(?:y|ise|ize)\b[^.?!]{0,24}\b(?:document|pdf|file|agreement|contract|notice)\b",
    r"\b(?:document|pdf|file|agreement|contract)\b[^.?!]{0,24}\bsummar(?:y|ise|ize)\b",
    r"\b(?:document|pdf|file)\b[^.?!]{0,16}\b(?:short|brief|gist|key\s*points?)\b",
    r"(?:दस्तावेज़|दस्तावेज|पीडीएफ|फ़ाइल).*(?:सारांश|संक्षेप|मुख्य\s*बात)",
    r"\b(?:document|pdf|file)\s*(?:ka|ki)\s*(?:summary|saransh|sar)\b",
)
# A checklist review of one document ("is anything missing from my rent
# agreement?"). Distinct from RISKY_CLAUSES, which reports what IS there.
DOCUMENT_REVIEW = _pattern(
    r"\b(?:check|review|vet|verify)\b[^.?!]{0,24}\b(?:agreement|contract|deed|nda|lease|notice|affidavit)\b",
    r"\b(?:agreement|contract|deed|nda|lease)\b[^.?!]{0,24}\b(?:check|review|vet|complete|missing)\b",
    r"\bwhat(?:'s| is)\s+missing\b", r"\bany(?:thing)?\s+missing\b",
    r"\bis\s+(?:this|my)\s+(?:agreement|contract|deed|lease|nda)\s+(?:ok|okay|fine|complete|safe)\b",
    r"\bchecklist\b",
    r"(?:क्या\s*कुछ\s*(?:छूट|कमी))|(?:समझौत[ेा].*(?:जाँच|जांच|समीक्षा))",
    r"\b(?:agreement|contract)\s*(?:me|mein)\s*(?:kya|kuch)\s*(?:missing|kami|chhut)\b",
)
# Comparing two documents the conversation already holds.
DOCUMENT_COMPARE = _pattern(
    r"\bcompare\b[^.?!]{0,30}\b(?:documents?|agreements?|contracts?|versions?|drafts?|files?|two|dono)\b",
    r"\b(?:documents?|agreements?|contracts?|files?)\b[^.?!]{0,20}\bcompare\b",
    r"\bwhat(?:'s| is| has)\s+changed\b",
    # "difference between" MUST name documents. Unqualified, it matched
    # "What is the difference between RTI and PIL?" -- an ordinary legal
    # question -- and routed it into document comparison.
    r"\bdiff(?:erence)?s?\s+between\b[^.?!]{0,40}"
    r"\b(?:documents?|agreements?|contracts?|versions?|drafts?|files?|these|two)\b",
    r"\bnew\s+(?:agreement|contract|version)\b[^.?!]{0,20}\b(?:changed?|different)\b",
    # `differences?` with the optional plural: written `difference\b` it
    # silently failed on "risky differences", which is how people ask.
    r"\bdono\b[^.?!]{0,28}\b(?:compare|differences?|antar|farak|fark)\b",
    r"\bkya\s+(?:change|badla)\b",
    r"(?:दोनों).*(?:तुलना|अंतर|फर्क)", r"क्या\s*बदल[ाे]", r"तुलना\s*कर",
)
# Dates and deadlines inside an uploaded document, as opposed to
# `CASE_TIMELINE`, which is the chronology of a case FILE.
DOCUMENT_TIMELINE = _pattern(
    r"\b(?:prepare|make|build|create|show)?\s*(?:a\s+)?timeline\b[^.?!]{0,24}"
    r"\b(?:from|of|for)\s+(?:this\s+|the\s+|my\s+)?(?:document|pdf|file|agreement|contract|notice)\b",
    r"\b(?:dates?|deadlines?|due\s*dates?|time\s*limits?)\b[^.?!]{0,24}"
    r"\b(?:document|pdf|isme|इसमें)\b",
    r"\b(?:document|pdf)\b[^.?!]{0,24}"
    r"\b(?:dates?|deadlines?|due\s*dates?|time\s*limits?)\b",
    # "agreement"/"contract"/"notice" are ALSO everyday generic legal
    # concepts (the same reasoning `_DOC_GENERIC_CONCEPT` documents for
    # `DOCUMENT_QUESTION` below), so -- unlike "document"/"pdf" above -- they
    # only count as a reference to an UPLOADED file behind a "this"/"my"
    # determiner. Confirmed live: "Notice mein 15 din ki payment deadline
    # jodo" (add a 15-day payment deadline clause to the legal notice
    # currently being DRAFTED, nothing uploaded) matched the old bare-noun
    # version of this pair, hijacked the orchestrator into this workflow, and
    # got stuck asking for an attachment on every following turn -- including
    # unrelated ones -- because nothing else scored highly enough to displace
    # it (see `_select_workflow`'s fallback-to-current behaviour).
    r"\b(?:dates?|deadlines?|due\s*dates?|time\s*limits?)\b[^.?!]{0,24}"
    r"(?:this|my)\s+(?:\w+\s+){0,2}(?:agreement|contract|notice)\b",
    r"(?:this|my)\s+(?:\w+\s+){0,2}(?:agreement|contract|notice)\b[^.?!]{0,24}"
    r"\b(?:dates?|deadlines?|due\s*dates?|time\s*limits?)\b",
    # "file" is deliberately excluded from the bare noun list above (and
    # re-added below only behind a determiner). In Hinglish it is routinely
    # a VERB -- "complaint/case/FIR file karna" = "to file a
    # complaint/case/FIR" -- so matching it as a document noun mistook
    # ordinary limitation questions ("...complaint file karne ke liye kitna
    # time limit hai?") for a request about an uploaded document and got the
    # conversation stuck asking for an attachment nobody meant to send.
    r"\b(?:dates?|deadlines?|due\s*dates?|time\s*limits?)\b[^.?!]{0,24}"
    r"(?:this|the|my|uploaded|attached|isi?|ye)\s+file\b",
    r"(?:this|the|my|uploaded|attached|isi?|ye)\s+file\b[^.?!]{0,24}"
    r"\b(?:dates?|deadlines?|due\s*dates?|time\s*limits?)\b",
    # Keep generic deadline wording tied to an actual document. Bare phrases
    # such as "kitne din ke andar notice bhejna hota hai?" are ordinary
    # legal limitation questions; claiming they refer to an uploaded file
    # starts the document workflow and incorrectly asks for an attachment.
    r"\bwhat\s+(?:are\s+the\s+)?(?:dates|deadlines)\b[^.?!]{0,24}"
    r"\b(?:in|from|of)\s+(?:this\s+|the\s+|my\s+)?(?:document|pdf|file|agreement|contract|notice)\b",
    r"\b(?:kab\s*tak|kitne\s*din)\b[^.?!]{0,24}"
    r"\b(?:document|pdf|agreement|contract|isme)\b",
    r"\b(?:kab\s*tak|kitne\s*din)\b[^.?!]{0,24}"
    r"(?:this|the|my|uploaded|attached|isi?|ye)\s+file\b",
    r"(?:तारीख़?ें|तिथियाँ|समय\s*सीमा|अंतिम\s*तिथि)",
    r"\b(?:tareekh|tarikh|samay\s*seema)\b",
)
# Choosing between the documents already uploaded in this conversation.
DOCUMENT_CHOOSE = _pattern(
    r"\bwhich\s+documents?\s+(?:do\s+you\s+have|are\s+(?:uploaded|available))\b",
    r"\bmy\s+uploaded\s+documents?\b", r"\blist\s+(?:my\s+)?(?:uploaded\s+)?documents?\b",
    r"\bdocuments?\s+(?:i\s+)?(?:have\s+)?uploaded\b",
    r"(?:कौन\s*से|मेरे)\s*दस्तावेज़?", r"\bkaunse\s+documents?\b",
)
RISKY_CLAUSES = _pattern(
    r"\brisky?\s*clauses?\b", r"\brisks?\b.*\b(?:clause|contract|agreement)\b",
    r"\bclauses?\b.*\brisky?\b", r"\bred\s*flags?\b",
    r"(?:जोखिम|रिस्की).*(?:क्लॉज|खंड|शर्त)", r"खतरनाक\s*(?:शर्त|क्लॉज)",
    r"risky.*(?:clause|shart)",
)
# PDF Q&A acceptance pass (2026-09-12): a SPECIFIC question about an
# uploaded document's own content -- "What is the notice period in this
# agreement?", "Does this contract allow subletting?", "Explain this
# clause" -- as opposed to every other DOCUMENT_* pattern above, each of
# which reports a FIXED shape (summary/risks/dates/checklist) regardless of
# what was actually asked. Deliberately anchored to an explicit
# document/clause reference (a determiner + "document/pdf/file/agreement/
# contract/notice/clause"), the same convention `DOCUMENT_REVIEW`/
# `DOCUMENT_TIMELINE` already use to avoid firing on an ordinary legal
# question that never mentions an uploaded file at all.
_DOC_NOUN_INNER = "document|pdf|file|agreement|contract|notice"
# "document"/"pdf" essentially never name a generic legal CONCEPT ("the
# document" is not how anyone asks an abstract legal question) and are
# never ordinary verbs -- safe to match loosely, with "this/the/my" and up
# to 2 descriptive words before them ("the LOAN document").
_DOC_ARTIFACT = "document|pdf"
# "agreement"/"contract"/"notice" ARE everyday GENERIC legal concepts too
# ("does THE contract need to be registered?", "what is THE notice period
# for resignation?" -- both ordinary legal questions with no uploaded
# document in play at all -- confirmed live regressions, PDF Q&A
# acceptance pass 2026-09-12). Restricted to "this"/"my", never bare "the"
# (which is what makes the generic reading possible) -- but, unlike "file"
# below, they are never ordinary VERBS, so a descriptive word between the
# determiner and the noun is still safe and still needed ("this LOAN
# agreement", "my RENTAL contract" are completely ordinary ways to name a
# specific uploaded document by what kind of one it is).
_DOC_GENERIC_CONCEPT = "agreement|contract|notice"
# "file" gets the strictest treatment of all: on top of the same
# generic-reference risk as above, it is routinely a VERB ("to FILE a
# complaint"), so a descriptive-word gap after the determiner let "the
# process to FILE a complaint" match "the" ... "file" as if "file" were
# its object. "this"/"my" only, and DIRECTLY adjacent -- no gap.
_DOC_VERBLIKE = "file"
_DOC_PHRASE = (
    rf"(?:(?:this|the|my)\s+(?:\w+\s+){{0,2}}(?:{_DOC_ARTIFACT}))"
    rf"|(?:(?:this|my)\s+(?:\w+\s+){{0,2}}(?:{_DOC_GENERIC_CONCEPT}))"
    rf"|(?:(?:this|my)\s+(?:{_DOC_VERBLIKE}))"
)
DOCUMENT_QUESTION = _pattern(
    rf"\b(?:in|from|according\s+to)\s+(?:{_DOC_PHRASE})\b",
    rf"\b(?:{_DOC_PHRASE})\b[^.?!]{{0,20}}"
    r"\b(?:say|says|state|states|mention|mentions|allow|allows|permit|permits|require|requires)\b",
    rf"\bdoes\s+(?:{_DOC_PHRASE})\b",
    rf"\bwhat\s+(?:is|does|are)\b[^.?!]{{0,40}}\b(?:{_DOC_PHRASE})\b",
    r"\bexplain\s+this\s+clause\b", r"\bwhat\s+does\s+this\s+clause\s+mean\b",
    r"\b(?:is|iss|ye|yeh|yah)\s+(?:document|documnet|pdf|file|dastavez|agreement|contract|clause|para|paragraph)\b"
    r"[^.?!]{0,20}\b(?:samjhao|smjhao|samajhao|explain|kya\s+(?:hai|likha|matlab))\b",
    # "Ye agreement subletting allow karta hai?" -- a yes/no question about
    # the document, not phrased with "kya"/"samjhao" at all.
    r"\b(?:is|iss|ye|yeh|yah)\s+(?:document|documnet|pdf|file|dastavez|agreement|contract)\b"
    r"[^.?!]{0,30}\b(?:allow|allows|permit|permits|hai|hoga|milega|chahiye)\b",
    # "Is document mein deposit kitna hai?" -- the wh-word trails the noun
    # rather than following it directly, and asks "how much"/"who"/"when",
    # not just "what".
    r"\b(?:document|pdf|file|dastavez|agreement|contract|notice)\s*(?:mein|me|में|मे)\b"
    r"[^.?!]{0,30}\b(?:kya|kitna|kitne|kitni|kaun|kab)\b",
    r"(?:इस|ये)\s*(?:दस्तावेज़|पीडीएफ|खंड|क्लॉज)\s*(?:को|में|मे)?\s*(?:समझा|क्या)",
)

# Producing a clean, professionally typeset version of an uploaded document
# (often a handwritten/scanned application) -- as opposed to DOCUMENT_SUMMARY
# (a short executive summary) or DOCUMENT_QUESTION (one specific question
# about its content), this asks for the DOCUMENT ITSELF, retyped/formatted/
# extracted, with its own original facts preserved. Confirmed live
# regression: "isko likh ke de do proper" and "isme se text extract kr lo",
# both against an uploaded handwritten application, fell through to general
# retrieval and answered a completely unrelated legal topic, silently
# dropping the uploaded document from context. The first two alternatives
# require an explicit reference to "this/the" document right next to a
# make-verb, so a bare "police complaint likh do" (drafting a NEW document
# from scratch, nothing uploaded) never matches; `_ALSO_WANTS_A_DRAFT`
# (checked by the workflow, same as every sibling DOCUMENT_* workflow) is
# the second backstop for the remaining alternatives, which do not require
# that reference.
_DOC_REF_FOR_FORMAT = (
    r"(?:isko|iska|iski|isme|isamein|is\s+document|is\s+file|this\s+document|this\s+file"
    r"|इसको|इसे|इसका|इसमें|इस\s*दस्तावेज़?|इस\s*फ़ाइल)"
)
_FORMAT_VERB = r"type[ds]?|format(?:ted|ting)?|transcri(?:be|bed|ption)|extract(?:ed|ion)?"
DOCUMENT_FORMAT = _pattern(
    rf"{_DOC_REF_FOR_FORMAT}[^.?!]{{0,20}}\b(?:{_MAKE})\b",
    rf"\b(?:{_MAKE})\b[^.?!]{{0,20}}{_DOC_REF_FOR_FORMAT}",
    rf"{_DOC_REF_FOR_FORMAT}[^.?!]{{0,30}}\b(?:{_FORMAT_VERB})\b",
    rf"\b(?:{_FORMAT_VERB})\b[^.?!]{{0,30}}{_DOC_REF_FOR_FORMAT}",
    r"\b(?:proper(?:ly)?|clean(?:ly)?|type[ds]?|professional(?:ly)?)\b[^.?!]{0,20}"
    r"\b(?:likh|likho|likh\s*do|likh\s*ke|type|format|banao)\b",
    r"\b(?:likh|likho|likh\s*do|type|format)\b[^.?!]{0,20}\bproper(?:ly)?\b",
    r"\btext\s*(?:nikaal|nikal)\s*(?:do|kar\s*do)\b",
    r"(?:साफ़?|प्रॉपर|प्रोफेशनल|टाइप)[^।.?!]{0,20}(?:लिख|लिखो|लिख\s*दो)",
    r"(?:लिख|लिखो|लिख\s*दो)[^।.?!]{0,20}(?:साफ़?|प्रॉपर|प्रोफेशनल)",
    r"टेक्स्ट\s*(?:निकाल|एक्सट्रैक्ट)",
)

# Administration. Every pattern here names an administrative object
# explicitly -- an ordinary user asking about "the index" of a contract must
# never be routed at the search index, and a role check would refuse them
# anyway, which is a worse experience than simply not matching.
ADMIN_KNOWLEDGE_BASE = _pattern(
    r"\bknowledge\s*base\b", r"\bkb\s*(?:status|dashboard|staging)\b",
    r"\breindex\b|\bre-?index(?:ing)?\b",
    r"\bindexing\s*job\b", r"\bstaging\s*(?:queue|status|records)\b",
    r"\bunowned\s*documents?\b", r"\bassign\s*(?:an?\s*)?owner\b",
    # The KB review queue. Scoped to review-QUEUE vocabulary ("needs review",
    # "review queue", "files needing review") rather than the bare word
    # "review", which an ordinary user says about their own contract.
    r"\bneeds?[\s_-]*review\b", r"\breview\s*queue\b",
    r"\bfiles?\b[^.?!]{0,12}\breview\b", r"\breview\b[^.?!]{0,12}\bfiles?\b",
    r"\bfiles?\b[^.?!]{0,12}\b(?:approv\w*|reject\w*|archive)\b",
    r"\b(?:approv\w*|reject\w*|archive)\b[^.?!]{0,12}\bfiles?\b",
    r"\b(?:path[_ -]*missing|missing\s+files?)\b",
    r"समीक्षा\s*(?:कतार|सूची)",
    r"\bcache\s*(?:flush|purge|clear)\b", r"\bflush\b[^.?!]{0,12}\bcache\b",
    r"\bindex\s*(?:drift|reconcil\w*)\b", r"\breconcil\w*\b[^.?!]{0,16}\bindex\w*\b",
    r"ज्ञान\s*(?:आधार|कोष)", r"पुनः\s*अनुक्रमण",
)
ADMIN_ANALYTICS = _pattern(
    r"\badmin\s*dashboard\b", r"\banalytics\s*dashboard\b",
    r"\bunanswered\s*(?:questions?|queue)\b",
    r"\bknowledge\s*gaps?\b", r"\bcache\s*(?:stats|statistics|hit\s*rate)\b",
    r"\bmark\b[^.?!]{0,20}\breviewed\b",
    r"अनुत्तरित\s*(?:प्रश्न|सवाल)", r"एडमिन\s*डैशबोर्ड",
)
ADMIN_SOURCES = _pattern(
    r"\blegal\s*sources?\b", r"\bsource\s*registry\b",
    r"\b(?:verify|reject|review)\b[^.?!]{0,20}\b(?:legal\s*)?sources?\b",
    r"\bstale\s*sources?\b", r"\bunverified\s*sources?\b",
    r"\bevaluation\s*runs?\b", r"\brun\b[^.?!]{0,12}\bevaluation\b",
    r"\baudit\s*logs?\b",
    r"विधिक\s*स्रोत", r"स्रोत\s*सत्यापन",
)

# Account-level self-service.
PREFERENCES = _pattern(
    r"\b(?:my\s+)?(?:preferences?|settings?)\b",
    r"\bchange\b[^.?!]{0,20}\b(?:language|explanation|format)\b",
    r"\b(?:default|preferred)\s+(?:language|format)\b",
    r"(?:मेरी|मेरे)?\s*(?:सेटिंग|प्राथमिकता)", r"\bsetting\s*(?:badlo|dikhao|change)\b",
)
DOWNLOADS = _pattern(
    r"\bdownload\s*(?:centre|center|history)\b",
    r"\bmy\s+downloads?\b", r"\bmy\s+(?:files|artifacts)\b",
    r"\b(?:downloads?|files)\b[^.?!]{0,12}\b(?:list|dikhao|show)\b",
    r"मेरी\s*(?:डाउनलोड|फ़ाइल)", r"डाउनलोड\s*सूची",
)
JOBS = _pattern(
    r"\b(?:background\s+)?jobs?\b[^.?!]{0,16}\b(?:status|list|show|dikhao|failed|pending)\b",
    r"\b(?:status|show|list)\b[^.?!]{0,16}\b(?:background\s+)?jobs?\b",
    r"\bmy\s+(?:background\s+)?jobs?\b",
    r"\bretry\b[^.?!]{0,16}\b(?:job|export|analysis)\b",
    r"\b(?:job|kaam)\s*(?:ki\s*)?(?:status|sthiti)\b", r"जॉब\s*(?:की\s*)?स्थिति",
)
# Erasing an account's data. Deliberately narrow: this must never fire on a
# passing mention of deletion.
DELETE_MY_DATA = _pattern(
    r"\bdelete\b[^.?!]{0,24}\b(?:my\s+(?:data|account\s+data|personal\s+data|everything))\b",
    r"\berase\b[^.?!]{0,24}\bmy\s+(?:data|information|records?)\b",
    r"\bforget\s+(?:me|everything\s+about\s+me)\b",
    r"\bremove\s+all\s+my\s+data\b",
    r"मेरा\s*(?:सारा\s*)?(?:डेटा|डाटा)\s*(?:हटा|मिटा|डिलीट)",
    r"\bmera\s+(?:sara\s+)?data\s+(?:delete|hata|mita)\b",
)
CLEAR_CONVERSATION = _pattern(
    r"\b(?:clear|delete|erase)\b[^.?!]{0,20}\b(?:this\s+)?(?:chat|conversation|session)\b",
    r"\bstart\s+(?:a\s+)?(?:fresh|new)\s+conversation\b",
    r"(?:यह|इस)\s*(?:चैट|बातचीत)\s*(?:हटा|मिटा|साफ़)",
    r"\b(?:chat|baat)\s*(?:clear|delete|saaf)\s*(?:karo|kar\s*do)\b",
)

# Conversation control.
CANCEL = _pattern(
    r"^\s*(?:cancel|abort|stop|forget\s*it|never\s*mind|chhodo|chod\s*do|rehne\s*do|band\s*karo)\b",
    r"(?:workflow|process|ye|this).*(?:cancel|band\s*kar)",
    r"रद्द\s*कर", r"बंद\s*कर", r"छोड़\s*दो", r"रहने\s*दो",
)
PAUSE = _pattern(
    r"\b(?:pause|later|baad\s*me|baad\s*mein|hold\s*on|wait)\b",
    r"बाद\s*में", r"रोक\s*दो", r"रुको",
)
RESTART = _pattern(
    r"\b(?:restart|start\s*over|from\s*(?:the\s*)?beginning|shuru\s*se|dobara\s*shuru)\b",
    r"फिर\s*से\s*शुरू", r"शुरू\s*से", r"दोबारा\s*शुरू",
)
BACK = _pattern(
    r"\b(?:go\s*back|previous\s*(?:question|answer)|pichla|pichhla)\b",
    r"पिछला\s*(?:सवाल|जवाब|प्रश्न|उत्तर)", r"वापस\s*जाओ",
)
# "Pick the parked task back up." Anchored, because "continue" appearing
# inside a sentence ("the lease continues until March") is not a control
# verb -- reading it as one would abandon whatever the user was actually
# saying and silently reopen an unrelated workflow.
RESUME = _pattern(
    r"^\s*(?:continue|resume|carry\s*on|go\s*on|proceed\s*with\s*(?:it|that)|pick\s*up\s*where)\b",
    r"^\s*(?:continue|resume)\s*(?:karo|kro|kare|karte\s*hai)\b",
    r"^\s*(?:aage\s*badho|wapas\s*shuru|jari\s*rakho|jaari\s*rakho)\b",
    r"^\s*(?:आगे\s*बढ़ो|जारी\s*रखो|फिर\s*से\s*शुरू\s*करो|वापस\s*शुरू)",
)
# "That answer was wrong -- the name is actually X." Distinct from BACK,
# which clears the LAST answer; this names the field to change.
CORRECTION = _pattern(
    r"\b(?:change|correct|update|fix|edit)\b[^.?!]{0,30}?\b(?:to|as|into)\b",
    r"\b(?:actually|instead|rather)\b.{0,40}\b(?:is|should\s*be|hona\s*chahiye)\b",
    r"\b(?:galat|ghalat)\b.{0,30}\b(?:hai|tha|thi)\b",
    r"\b(?:badal\s*do|badlo|sudhar\s*do|theek\s*karo)\b",
    r"बदल\s*दो|बदलो|सुधार\s*दो|ठीक\s*करो|ग़लत\s*है|गलत\s*है",
)
# "What can you do?" -- answered from the real registry, never from a list
# written by hand in a prompt.
CAPABILITIES = _pattern(
    r"\bwhat\s+can\s+(?:you|u)\s+do\b", r"\byour\s+(?:capabilities|features)\b",
    r"\bhelp\s*menu\b", r"\bwhat\s+(?:all\s+)?(?:features|services)\b",
    r"\bkya\s+kya\s+kar\s+sakt[ae]\b", r"\btum\s+kya\s+kar\s+sakt[ae]\b",
    r"क्या\s*क्या\s*कर\s*सकत", r"आप\s*क्या\s*कर\s*सकत",
)
# The user asked an ordinary legal question in the middle of a workflow.
# Detected so the workflow can be parked instead of swallowing the question
# as a field answer.
QUESTION = _pattern(
    r"^\s*(?:what|why|how|when|where|which|who|is|are|can|does|do|should)\b.*\?",
    r"\bkya\s+(?:hota|hoti|hote|hain|hai|matlab)\b", r"\bkaise\b.*\?",
    r"\bmatlab\s+kya\b", r"\bkya\s+hai\b\s*\?",
    r"क्या\s*(?:होता|होती|होते|हैं|है|मतलब)", r"कैसे.*\?",
    # A trailing "?" is a near-universal question marker across scripts, and
    # the patterns above miss common Hinglish phrasings where "kya" sits
    # away from the verb ("kya X essential elements hote hain?", "X mein kya
    # likha hai?"). Requiring the "?" keeps this from also matching a plain
    # declarative field answer typed mid-workflow.
    r"\bkya\b.*\?", r"क्या.*\?",
)


def score(pattern: re.Pattern[str], message: str, weight: float = 0.9) -> float:
    """`weight` if `pattern` matches `message`, else 0.0.

    A single scoring helper so every workflow's `matches_intent` reads the
    same way and no workflow invents its own confidence scale.
    """
    return weight if pattern.search(message or "") else 0.0
