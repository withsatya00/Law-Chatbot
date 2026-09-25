import re
from typing import ClassVar

from app.language.normalizer import QueryNormalizer
from app.rag.multilingual import legal_english_variants

# Users still search by the old Indian Penal Code (IPC) section numbers even
# though IPC was repealed and replaced by the Bharatiya Nyaya Sanhita (BNS)
# 2023 on 1-7-2024 -- the KB is indexed under BNS numbering/terminology, not
# IPC's. Left un-rewritten, "Section 420 IPC" is searched as literal tokens
# ["420", "ipc"], and BNS's own numbering coincidentally collides with an
# unrelated IPC number (BNS Section 420 is "Appeal against conviction by High
# Court", nothing to do with IPC 420 "Cheating") -- confirmed live against the
# index: that wrong chunk, plus FAQ chunks that only mention "Sec. 34 IPC...
# has been removed", outrank every actually-relevant chunk (there wasn't one
# indexed at all -- see the Bharatiya Nyaya Sanhita ingestion gap this fix
# pairs with). Each mapping below is verified against the Act's own section
# headings, not guessed; extend only with equally-verified entries.
IPC_TO_BNS_CROSSWALK: dict[str, tuple[str, str]] = {
    "420": ("318", "cheating dishonestly inducing delivery of property"),
    "302": ("101", "murder punishment for murder"),
    "299": ("100", "culpable homicide"),
    "304b": ("80", "dowry death"),
    "375": ("63", "rape"),
    "376": ("64", "rape punishment"),
    "379": ("303", "theft punishment for theft"),
}
IPC_SECTION_RE = re.compile(
    r"\b(?:section\s+)?(\d{1,3}[A-Z]?)\s*(?:of\s+(?:the\s+)?)?ipc\b"
    r"|\bipc\s+(?:section\s+)?(\d{1,3}[A-Z]?)\b",
    re.IGNORECASE,
)

# The reverse direction of `IPC_TO_BNS_CROSSWALK`: a query naming a CURRENT
# provision by number alone ("BNS 318 kya hai?", "BNSS 173", "Section 35
# BNSS") carries no topic vocabulary at all, so BM25 has only the bare digits
# to work with and embedding similarity has nothing to anchor on. Supplying
# the provision's own statutory subject matter as an extra search variant is
# the same trick the IPC crosswalk already performs, applied to the numbering
# the corpus is actually indexed under. Keyed by "<ACT> <number>"; every entry
# is taken from the Act's own section heading, not paraphrased -- extend only
# with equally-verified entries.
CURRENT_SECTION_TOPICS: dict[str, str] = {
    "BNS 101": "murder punishment for murder",
    "BNS 100": "culpable homicide",
    "BNS 80": "dowry death",
    "BNS 63": "rape definition",
    "BNS 64": "punishment for rape",
    "BNS 316": "criminal breach of trust dishonest misappropriation",
    "BNS 318": "cheating and dishonestly inducing delivery of property punishment",
    "BNS 336": "forgery making a false document",
    "BNS 351": "criminal intimidation threat to cause injury",
    "BNS 308": "extortion putting person in fear of injury",
    "BNSS 35": "when police may arrest without warrant cognizable offence reasons recorded notice of appearance",
    "BNSS 173": "information in cognizable cases first information report FIR registration zero FIR",
    "BNSS 175": "power of police officer to investigate cognizable case",
    "BNSS 176": "procedure for investigation",
    "BNSS 183": "recording of confessions and statements by magistrate",
    "BNSS 482": "direction for grant of bail to person apprehending arrest anticipatory bail",
    "BSA 61": "electronic or digital record admissibility",
    "BSA 63": "admissibility of electronic records certificate",
}
_CURRENT_SECTION_RE = re.compile(
    r"\b(BNSS|BNS|BSA)\b[^0-9]{0,12}?(\d{1,4}[A-Z]{0,2})\b"
    r"|\b(?:section|sec\.?|धारा)\s*(\d{1,4}[A-Z]{0,2})\b[^0-9]{0,12}?\b(BNSS|BNS|BSA)\b",
    re.IGNORECASE,
)

# A FIR/Zero-FIR concept query ("FIR kya hota hai?") carries no digits at
# all, so `parse_section_lookup`/`_named_section_citation` never resolve it
# to a section number for an exact metadata lookup -- the `expansions` entry
# below only injects "BNSS Section 173" as extra TEXT for BM25/embedding
# similarity, which is confirmed live NOT enough on its own: BNSS Section
# 173 (the actual FIR-registration provision) still doesn't enter the top-8
# fused candidates, because its own text shares more incidental "cognizable
# offence" vocabulary with unrelated Acts than FIR-specific vocabulary with
# the query. `LegalRetriever.retrieve` uses this constant to also force an
# exact `find_by_section_number` lookup, the same way a named citation does
# (see its own comment for the full trace) -- exposed here, rather than a
# second independent regex in retriever.py, so the concept-to-section
# mapping has one source of truth.
FIR_CONCEPT_RE = re.compile(
    r"\bzero\s+fir\b|\b(fir|f\.i\.r|firr|police complaint|first information report)\b", re.IGNORECASE
)
FIR_BNSS_SECTION_NUMBER = "173"

# Confirmed live: "which section of BNS talks about rape?" answered correctly
# from Section 64 (punishment) alone, but never surfaced Section 63 (the
# definition) -- an answer covering only the punishment, with no definition,
# for a plain "what section covers rape" question. Both are verified BNS
# sections (see `CURRENT_SECTION_TOPICS` above); forcing both present the
# same way `FIR_CONCEPT_RE` does gives the LLM what it needs to distinguish
# them rather than leaving that entirely up to which one similarity ranking
# happened to surface.
RAPE_CONCEPT_RE = re.compile(r"\brape\b", re.IGNORECASE)
RAPE_BNS_SECTION_NUMBERS = ("63", "64")

# Confirmed live: "theft ki definition batao BNS par" returned "no verified
# document" even after BNS's own document was approved out of `needs_review`
# -- BNS Section 303 (theft) never entered the top-8 similarity-fused
# candidates for the same reason FIR's Section 173 didn't: it shares more
# incidental "cognizable offence"/penalty vocabulary with unrelated Acts
# than theft-specific vocabulary with a bare definitional query.
THEFT_CONCEPT_RE = re.compile(r"\btheft\b|\bchori\b", re.IGNORECASE)
THEFT_BNS_SECTION_NUMBER = "303"


def current_section_topic(normalized_query: str) -> str | None:
    """The statutory subject matter of a BNS/BNSS/BSA section named by number
    in `normalized_query`, or `None` when no known citation is present."""
    match = _CURRENT_SECTION_RE.search(normalized_query)
    if not match:
        return None
    act = (match.group(1) or match.group(4) or "").upper()
    number = (match.group(2) or match.group(3) or "").upper()
    if not act or not number:
        return None
    topic = CURRENT_SECTION_TOPICS.get(f"{act} {number}")
    if not topic:
        return None
    return f"{act} Section {number} {topic}"


class SmartQueryRewriter:
    expansions: ClassVar[list[tuple[re.Pattern[str], str]]] = [
        (re.compile(r"\b(salary|wage|payment).*(nahi|pending|unpaid|not paid)|salary nahi mili", re.IGNORECASE), "salary pending labour law india wages employment dispute"),
        (re.compile(r"\b(online fraud|cyber fraud|upi|otp|digital scam|internet fraud)\b", re.IGNORECASE), "cyber crime online payment fraud information technology act"),
        (re.compile(r"\b(property issue|land dispute|ownership|sale deed|gift deed)\b", re.IGNORECASE), "property ownership dispute transfer of property act registration act"),
        (re.compile(r"\bzero\s+fir\b", re.IGNORECASE), "Zero FIR registration at any police station jurisdiction transfer BNSS Section 173"),
        (re.compile(r"\b(fir|f\.i\.r|firr|police complaint|first information report)\b", re.IGNORECASE), "FIR first information report police complaint cognizable offence registration BNSS Section 173"),
        (re.compile(r"\b(cheque bounce|dishonou?r)\b", re.IGNORECASE), "cheque bounce dishonour negotiable instruments act section 138"),
        (re.compile(r"\b(divorce|talak|maintenance|custody)\b", re.IGNORECASE), "family law divorce maintenance child custody marriage act"),
        # Confirmed live (2026-09-25): "meri patni bina karan alag reh rahi
        # hai, mere legal options kya hain?" never retrieved the Hindu
        # Marriage Act at all -- unlike "divorce"/"talaq" just above, "living
        # separately without reasonable excuse" shares no vocabulary with any
        # statute's own wording, so the raw query alone gave BM25/embedding
        # search nothing to anchor on. The Act's own heading is "restitution
        # of conjugal rights" (Section 9), which this names directly, the
        # same trick the FIR/arrest/threat expansions above already use for
        # colloquial-vs-statutory vocabulary gaps.
        (
            re.compile(
                r"\b(wife|husband|spouse)\b[^.?!]{0,40}\b(?:living|staying|stays?)\s+separately\b"
                r"|\bliving\s+separately\b(?:\s+without\s+(?:reasonable\s+)?(?:cause|excuse|reason))?"
                r"|\balag\s+reh(?:ti|ta|rahi|raha|rahe)\b|\brestitution\s+of\s+conjugal\s+rights\b"
                r"|\bconjugal\s+rights\b"
                r"|पति.{0,15}अलग\s*रह|पत्नी.{0,15}अलग\s*रह|अलग\s*रह\s*रह[ीा]|वैवाहिक\s*अधिकार|दांपत्य\s*अधिकार",
                re.IGNORECASE,
            ),
            "restitution of conjugal rights Hindu Marriage Act Section 9 living separately without reasonable excuse",
        ),
        (re.compile(r"\b(consumer|refund|defective|warranty)\b", re.IGNORECASE), "consumer complaint consumer protection act deficiency in service"),
        (re.compile(r"\b(pg|rent|tenant|landlord|deposit|kiraya)\b", re.IGNORECASE), "security deposit tenant landlord rental deposit refund rent agreement property law"),
        (re.compile(r"\bbail\b|\bjamanat\b", re.IGNORECASE), "bail release from custody regular bail bail application bailable offence bail bond BNSS"),
        (re.compile(r"\banticipatory\s+bail\b", re.IGNORECASE), "anticipatory bail pre arrest bail apprehending arrest BNSS"),
        (re.compile(r"\b(recovery agent|loan recovery|debt collector)\b", re.IGNORECASE), "RBI recovery agents debt collection borrower harassment guidelines"),
        (re.compile(r"\b(unauthori[sz]ed transaction|wrong account|galat account|bank transaction)\b", re.IGNORECASE), "unauthorized electronic banking transaction UPI wrong transfer RBI customer liability"),
        (re.compile(r"\b(legal notice|notice kya|notice hota)\b", re.IGNORECASE), "legal notice formal demand notice advocate reply civil dispute"),
        # CPC s.80 (notice before suing the Government/a public officer) had no
        # rule at all, English or otherwise, so it depended entirely on the
        # embedding leg. Phrase-level on purpose: "government" alone is far too broad.
        (
            re.compile(
                r"\b(?:sue|suing|sued|suit|case|lawsuit)\s+(?:against\s+)?(?:the\s+)?(?:government|govt|public\s+officer)\b"
                r"|\b(?:government|govt)\s+(?:department\s+)?(?:notice|legal\s+notice)\s+before\s+(?:suing|filing)\b"
                r"|\bnotice\s+(?:before|prior\s+to)\s+(?:suing|filing\s+(?:a\s+)?(?:suit|case))\s+(?:against\s+)?(?:the\s+)?(?:government|govt)\b"
                r"|\bcpc\s+(?:section\s+)?80\b|\bsection\s+80\s+(?:of\s+)?(?:the\s+)?(?:cpc|code\s+of\s+civil\s+procedure)\b",
                re.IGNORECASE,
            ),
            "notice before suit against Government or public officer Code of Civil Procedure Section 80 suit against government",
        ),
        (re.compile(r"\b(encroachment|illegal kabza|possession|fake property documents|forged property)\b", re.IGNORECASE), "property possession encroachment forged documents illegal occupation specific relief transfer property"),
        (re.compile(r"\bblackmail\b", re.IGNORECASE), "blackmail extortion criminal intimidation cyber crime BNS"),
        (re.compile(r"\bcognizable\b", re.IGNORECASE), "cognizable non-cognizable offence classification criminal procedure BNSS"),
        # Confirmed live (2026-09-25): "Arbitration aur mediation mein kya
        # antar hai?" retrieved both Acts' chunks into the raw candidate pool
        # fine (the query already names them), but they had no reranker topic
        # bonus of their own (see `LegalReranker._topic_bonus`) and lost the
        # top_k=6 cut to unrelated State Acts sharing only generic words.
        # This expansion also carries the statutes' own vocabulary for a
        # query that names the concept without the English words at all.
        (
            re.compile(
                r"\barbitrat(?:ion|or|e|ed|ing)\b|\bmediat(?:ion|or|e|ed|ing)\b|\bconciliat(?:ion|or)\b"
                r"|\balternative\s+dispute\s+resolution\b|\badr\b",
                re.IGNORECASE,
            ),
            "arbitration mediation conciliation alternative dispute resolution Arbitration and Conciliation Act "
            "1996 Mediation Act 2023 arbitral tribunal arbitral award",
        ),
        # Topic -> section linking for the BNS offences users ask about by
        # NAME rather than by number. Confirmed live gap: "IPC 420 ab BNS ki
        # kaunsi dhara hai?" answered correctly (the crosswalk above supplies
        # "BNS Section 318" as a search variant), while the same provision
        # asked topically -- "What is the punishment for cheating under BNS?",
        # its Hindi equivalent, and "BNS 318 kya hai?" -- all returned "no
        # verified document". The corpus indexes the provision under its own
        # statutory heading ("Cheating and dishonestly inducing delivery of
        # property"), so a query using the everyday word plus the Act
        # abbreviation had too little lexical overlap to clear the relevance
        # gate. Each entry names the words the SECTION ITSELF uses, the same
        # way the RTI/GST/POSH expansions above do.
        (
            re.compile(r"\bcheat(?:ing|ed)?\b|\bdhokha|\bdhokhadhadi\b|\bthagi\b|\bfraud(?:ulent(?:ly)?)?\b", re.IGNORECASE),
            (
                "Bharatiya Nyaya Sanhita BNS Section 318 cheating dishonestly inducing delivery of property "
                "punishment imprisonment fine dishonest inducement"
            ),
        ),
        (
            re.compile(r"\bcriminal\s+breach\s+of\s+trust\b|\bmisappropriat", re.IGNORECASE),
            "Bharatiya Nyaya Sanhita BNS Section 316 criminal breach of trust dishonest misappropriation of property",
        ),
        (
            re.compile(r"\bforger(?:y|ies)\b|\bforged\s+document", re.IGNORECASE),
            "Bharatiya Nyaya Sanhita BNS Section 336 forgery false document valuable security punishment",
        ),
        # Part 58 "Answer Quality Audit" issue 4: an arrest-powers question
        # must be answerable WITH its statutory safeguards, not just the bare
        # power. Naming the safeguard provisions here is what puts them in
        # the retrieved context at all -- the system prompt's
        # "never state a coercive power without its conditions" rule can only
        # comply if the conditions were actually retrieved.
        (
            re.compile(
                r"\barrest\b.*\bwarrant\b|\bwarrant\b.*\barrest\b|\bwithout\s+(?:a\s+)?warrant\b"
                r"|\bgiraftar\b|\bhirasat\b|\bdetain(?:ed|ment)?\b",
                re.IGNORECASE,
            ),
            (
                "BNSS Section 35 arrest without warrant cognizable offence police officer reasons to be recorded "
                "in writing necessity of arrest notice of appearance Section 35(3) Section 47 grounds of arrest "
                "to be communicated Section 46 arrest how made Section 43 rights of arrested person"
            ),
        ),
        # Part 58 issue 16/23: threat/harassment/stalking questions returned
        # "no verified document" even though BNS covers criminal
        # intimidation, stalking, and harassment by communication -- the
        # colloquial words users type ("dhamki", "pareshan", "harass") share
        # almost no vocabulary with the statute's own headings.
        (
            re.compile(
                r"\bthreat(?:en(?:ing|ed)?|s)?\b|\bdhamki\b|\bdhamka\b|\bharass(?:ment|ing|ed)?\b|\bpareshan\b"
                r"|\bstalk(?:ing|er)?\b|\bblank\s+calls?\b|\bobscene\s+calls?\b"
                r"|धमकी|परेशान|पीछा",
                re.IGNORECASE,
            ),
            (
                "Bharatiya Nyaya Sanhita BNS criminal intimidation Section 351 stalking Section 78 "
                "insulting modesty by word gesture Section 79 extortion Section 308 "
                "Information Technology Act offensive message electronic communication police complaint "
                "BNSS Section 173"
            ),
        ),
        # These four mirror the FIR/bail/cheque-bounce pattern above for the
        # newly-added Acts (RTI, CGST, POSH, Income-tax 2025): the query
        # commonly uses the acronym or a colloquial phrase, but the actual
        # statute text never does -- "RTI" doesn't appear anywhere in the
        # Right to Information Act's own operative text, which always says
        # "this Act"/spells the name out only once in its own title. Without
        # an expanded variant carrying the words the text ACTUALLY uses,
        # BM25/embedding similarity has too little lexical overlap to rank
        # the real content above unrelated chunks that coincidentally share
        # a generic word like "application" with the bare query. Confirmed
        # live: retrieval for "RTI application kaise file karte hain?"
        # surfaced the RTI Act's own table-of-contents chunk as its only
        # candidate (real content never entered the pool at all) while an
        # unrelated historical document and a GST chunk outranked it after
        # reranking, purely on incidental word overlap.
        (re.compile(r"\brti\b|\bright\s+to\s+information\b", re.IGNORECASE), "Right to Information Act RTI application public authority information officer public information officer appeal"),
        (re.compile(r"\bgst\b|\bgoods\s+and\s+services\s+tax\b", re.IGNORECASE), "Goods and Services Tax GST Act CGST registration return filing input tax credit"),
        (re.compile(r"\bposh\b|\bsexual\s+harassment\b.*\bworkplace\b|\bworkplace\b.*\bharassment\b", re.IGNORECASE), "Sexual Harassment of Women at Workplace Act POSH internal committee complaint inquiry"),
        (re.compile(r"\bincome\s+tax\b|\bitr\b", re.IGNORECASE), "Income-tax Act return filing assessment penalty deduction"),
        # A dedicated, narrower expansion for the late-filing-fee shape
        # specifically -- mirrors "zero fir" having its own expansion
        # distinct from the generic "fir" one just above it. Confirmed
        # live: the generic income-tax expansion alone wasn't specific
        # enough to rank the Income-tax Act, 2025's actual Section 428
        # ("Fee for default in furnishing return of income") above other
        # Income-tax Act chunks for a colloquial "penalty for late filing"
        # phrasing, even though the section itself IS correctly indexed
        # (confirmed by direct search) -- explicitly naming its own
        # statutory language ("fee for default", "furnishing return")
        # gives BM25/embedding search the actual words the provision uses.
        (re.compile(r"\b(late|delay|deadline).{0,15}(file|filing|return)\b|\bitr.{0,10}(late|penalty|due date)\b", re.IGNORECASE), "fee for default in furnishing return of income Income-tax Act belated return due date section 263"),
        # Native-script anchors supply the English statutory vocabulary used
        # by the KB; multilingual embeddings still handle the full sentence.
        #
        # QA session 2026-09-24 ("Hindi retrieval investigation"): this
        # pattern's Devanagari alternation used to include सजा
        # ("saza" -- "punishment/sentence") alongside the two terms that are
        # actually specific to cheating/fraud (धोखाध
        # ड़ी "dhokhadhadi"/चीट "cheat"). "saza"
        # is generic -- it's the ordinary Hindi word for "punishment" and
        # appears in the overwhelming majority of Hindi criminal-law
        # questions regardless of topic ("... ki saza kya hai?" = "what is
        # the punishment for ...?") -- so this fired on ANY Devanagari
        # question mentioning punishment at all, not just cheating/fraud
        # ones. Live-reproduced: a plain cheque-bounce question ("चेक बाउंस
        # होने पर भारत में क्या सजा है?", no mention of fraud/cheating)
        # spuriously got "Bharatiya Nyaya Sanhita BNS Section 318 cheating
        # ..." appended to its expanded query -- diluting the correctly
        # retrieved Negotiable Instruments Act Section 138 chunk's
        # lexical-overlap reranking score against BNS/BNSS-family chunks
        # that share vocabulary with the spurious expansion ("Bharatiya",
        # "Sanhita", "Section"), and knocking it out of the LLM's final
        # context -- confirmed root cause of the QA report's "Hindi
        # retrieval unreliable" finding (BUG-106) for this specific query.
        # Removing the generic term restores the pattern's actual intent:
        # match only when the query names cheating/fraud specifically.
        (re.compile("[\\u0900-\\u097f].*(?:\\u0927\\u094b\\u0916\\u093e\\u0927\\u0921\\u093c\\u0940|\\u091a\\u0940\\u091f)|(?:\\u0927\\u094b\\u0916\\u093e\\u0927\\u0921\\u093c\\u0940|\\u091a\\u0940\\u091f).*[\\u0900-\\u097f]"), "Bharatiya Nyaya Sanhita BNS Section 318 cheating punishment dishonest inducement"),
        (re.compile("(?:\\u0911\\u0928\\u0932\\u093e\\u0907\\u0928|\\u0938\\u093e\\u0907\\u092c\\u0930|\\u092c\\u0948\\u0902\\u0915\\u093f\\u0902\\u0917).*(?:\\u092b\\u094d\\u0930\\u0949\\u0921|\\u0927\\u094b\\u0916\\u093e\\u0927\\u0921\\u093c\\u0940)|(?:\\u092b\\u094d\\u0930\\u0949\\u0921|\\u0927\\u094b\\u0916\\u093e\\u0927\\u0921\\u093c\\u0940).*(?:\\u0911\\u0928\\u0932\\u093e\\u0907\\u0928|\\u0938\\u093e\\u0907\\u092c\\u0930|\\u092c\\u0948\\u0902\\u0915\\u093f\\u0902\\u0917)"), "cyber crime online banking fraud UPI unauthorised electronic transaction immediate reporting"),
        (re.compile("(?:\\u092c\\u093f\\u0928\\u093e\\s*\\u0935\\u093e\\u0930\\u0902\\u091f|\\u0935\\u093e\\u0930\\u0902\\u091f).*(?:\\u0917\\u093f\\u0930\\u092b\\u094d\\u0924\\u093e\\u0930|\\u092a\\u0941\\u0932\\u093f\\u0938)|(?:\\u0917\\u093f\\u0930\\u092b\\u094d\\u0924\\u093e\\u0930|\\u092a\\u0941\\u0932\\u093f\\u0938).*(?:\\u092c\\u093f\\u0928\\u093e\\s*\\u0935\\u093e\\u0930\\u0902\\u091f|\\u0935\\u093e\\u0930\\u0902\\u091f)"), (
            "BNSS Section 35 police arrest without warrant cognizable offence reasons to be recorded in writing "
            "necessity of arrest notice of appearance Section 35(3) Section 47 grounds of arrest communicated "
            "Section 43 rights of arrested person"
        )),
    ]

    def __init__(self) -> None:
        self.normalizer = QueryNormalizer()

    async def rewrite(self, query: str, intent: str | None = None, context_summary: str | None = None) -> str:
        return self.combine(self.expand_queries(query, intent=intent, context_summary=context_summary))

    def expand_queries(self, query: str, intent: str | None = None, context_summary: str | None = None) -> list[str]:
        normalized = self.normalizer.normalize(query)
        queries = [normalized]
        ipc_match = IPC_SECTION_RE.search(normalized)
        if ipc_match:
            ipc_section = (ipc_match.group(1) or ipc_match.group(2) or "").lower()
            crosswalk = IPC_TO_BNS_CROSSWALK.get(ipc_section)
            if crosswalk:
                bns_section, topic = crosswalk
                queries.append(f"Bharatiya Nyaya Sanhita BNS Section {bns_section} {topic}")
        current_topic = current_section_topic(normalized)
        if current_topic:
            queries.append(current_topic)
        for pattern, expansion in self.expansions:
            if pattern.search(normalized):
                queries.append(expansion)
        # Phase 1 item 1: the legal concepts a non-English query names,
        # rendered in the statutory English the corpus is written in, so a
        # Hindi/Marathi/Gujarati/Tamil/Telugu/Bengali/Urdu question reaches the
        # BM25 leg (which has no multilingual capability at all) and
        # `_is_relevant_chunk`'s lexical gate with real English legal
        # vocabulary behind it, instead of being rejected for the sole reason
        # that it wasn't asked in English.
        #
        # Appended AFTER the pattern expansions on purpose. Each variant is
        # searched independently, so order is irrelevant to retrieval -- but
        # `combine()` de-duplicates tokens across the whole joined string, so
        # whichever variant comes first keeps its tokens and later ones lose
        # them. Putting these last leaves every existing English expansion
        # byte-for-byte unchanged (confirmed by the Zero-FIR expansion test,
        # which asserts the contiguous phrase "police station" survives
        # combining).
        queries.extend(legal_english_variants(normalized))
        # Phase 6 "Retrieval Recall Fix": every OTHER intent value in this system is
        # genuine natural-language text ("Cyber Crime", "Bail", "General Legal Query"
        # -- real English phrases that legitimately help BM25/embedding search when
        # appended as a second query variant). `SECTION_LOOKUP` (Phase 2) is the one
        # exception: a machine-readable classification label, not language. Confirmed
        # live: appending it as a search variant for "Section 420" contributes zero
        # relevant BM25/vector matches of its own, yet still participates in
        # `retriever.py`'s outer RRF fusion across variants -- diluting the genuine
        # chunk's combined score enough to push it from rank 10 (present) to absent
        # from the final top-10, even though the SAME chunk ranked BM25 #1 on the
        # real "Section 420" variant alone. Excluding only this one label (not
        # widening any candidate count/limit) restores that rank-10 presence.
        if intent and intent != "SECTION_LOOKUP" and intent.lower() not in normalized.lower():
            queries.append(intent)
        if context_summary:
            queries.append(context_summary[:300])
        seen: set[str] = set()
        unique_queries = []
        for item in queries:
            key = item.lower()
            if item and key not in seen:
                unique_queries.append(item)
                seen.add(key)
        return unique_queries

    def combine(self, queries: list[str]) -> str:
        combined = " ".join(queries)
        return " ".join(dict.fromkeys(combined.split()))
