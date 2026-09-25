import json
import re

import structlog

from app.drafting.field_label_translations import localized_field_label
from app.drafting.localized_dates import MONTH_WORD_TO_NUMBER
from app.drafting.templates.base import DraftTemplateDefinition
from app.llm.base import ChatMessage
from app.llm.deadline import has_time_for
from app.llm.factory import LLMFactory
from app.llm.prompts import prompt_registry

log = structlog.get_logger(__name__)

_AMOUNT_PATTERN = re.compile(r"(?:rs\.?|inr|₹)\s*([0-9][0-9,]*(?:\.\d{1,2})?)", re.IGNORECASE)
# The "<day> <month name> <year>" branch matched only Latin-alphabet month
# names ([A-Za-z]+) -- a Hindi/Tamil/Telugu/Kannada/Bengali date like
# "15 जुलाई 2026" never matched at all, so the field was never extracted in
# the first place (compounding the separate validation-rejection bug this
# was paired with). Widened to also accept any localized month word this
# codebase recognizes (`MONTH_WORD_TO_NUMBER`, shared with the date
# validator so both stay in sync).
_DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:[A-Za-z]+|"
    + "|".join(re.escape(word) for word in MONTH_WORD_TO_NUMBER)
    + r")\s+\d{4})\b"
)
_MOBILE_PATTERN = re.compile(r"(?<!\d)([6-9]\d{9})(?!\d)")
_EMAIL_PATTERN = re.compile(r"\b([\w.+-]+@[\w-]+\.[\w.-]+)\b")
_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Z][a-zA-Z.]+(?:\s+[A-Z][a-zA-Z.]+){0,3})", re.IGNORECASE),
    re.compile(r"\bthis is\s+([A-Z][a-zA-Z.]+(?:\s+[A-Z][a-zA-Z.]+){0,3})\s+(?:here|speaking)", re.IGNORECASE),
    re.compile(r"मेरा नाम\s+([^\s,।]+(?:\s+[^\s,।]+){0,3})\s+है"),
    re.compile(r"मैं\s+([^\s,।]+(?:\s+[^\s,।]+){0,3})\s+हूं"),
]
# Accept the standard chandrabindu spelling as well as the existing informal
# "हूं" spelling, without depending on the optional LLM extractor.
_NAME_PATTERNS.append(re.compile(r"मैं\s+([^\s,।]+(?:\s+[^\s,।]+){0,3})\s+हूँ"))

_DOCUMENT_IDENTIFIER_PATTERN = re.compile(
    r"(?:certificate|document|प्रमाणपत्र|दस्तावेज़)\s*(?:number|no\.?|संख्या)\s*[:\-]?\s*"
    r"([A-Z0-9][A-Z0-9./_-]{2,})",
    re.IGNORECASE,
)
_HINDI_LOST_DOCUMENT_PATTERN = re.compile(
    r"(?:मेरा|मेरी)\s+([^।\n]{2,80}?(?:प्रमाणपत्र|दस्तावेज़|डिग्री|मार्कशीट))\s+"
    r"\d{1,2}\s+[^\s]+\s+\d{4}\s+को",
    re.IGNORECASE,
)
_HINDI_LOSS_PLACE_PATTERN = re.compile(
    r"\d{1,2}\s+[^\s]+\s+\d{4}\s+को\s+(.{2,100}?)\s+(?:में|पर)\s+(?:कहीं\s+)?"
    r"(?:खो|गुम)\s+(?:गया|गयी|गई)",
    re.IGNORECASE,
)

# Small known-value lists used for zero-LLM extraction of otherwise-unstructured
# fields. Deliberately not exhaustive — anything not on these lists still gets
# asked for directly by the conversation engine, this is just a head start.
_KNOWN_BANKS = [
    "State Bank of India", "SBI", "HDFC Bank", "HDFC", "ICICI Bank", "ICICI", "Axis Bank",
    "Punjab National Bank", "PNB", "Bank of Baroda", "Canara Bank", "Union Bank of India",
    "Kotak Mahindra Bank", "Kotak", "IDBI Bank", "Yes Bank", "IndusInd Bank", "Bank of India",
    "Central Bank of India", "Indian Bank", "UCO Bank", "Paytm Payments Bank",
]
_KNOWN_CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune",
    "Ahmedabad", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna",
    "Surat", "Vadodara", "Chandigarh", "Ranchi", "Guwahati", "Coimbatore", "Nashik",
    "Meerut", "Varanasi", "Agra", "Amritsar", "Noida", "Gurgaon", "Gurugram",
]

# Which regex-derived value maps to which field key, per candidate field name.
_AMOUNT_FIELD_CANDIDATES = (
    "fraud_amount", "cheque_amount", "principal_amount", "amount_paid",
    "security_deposit_amount", "dues_amount",
)
_DATE_FIELD_CANDIDATES = ("incident_date", "cheque_date", "purchase_date", "loss_date")
# What separates a field LABEL from its value. The colon has three forms in
# real Indian input: ASCII `:`, the fullwidth `：` that a Devanagari or CJK
# keyboard produces, and the small-form variants. A message using any of them
# is the same form, and reading only ASCII meant a whole labelled reply was
# treated as one unstructured blob.
_LABEL_SEPARATOR = r"(?:\s*[:：﹕︓]\s*|\s+(?:is|are)\s+)"
# "Facts of the Case (in your own words)" is how the engine PRINTS the label;
# "Facts of the Case:" is how the user types it back.
_WITHOUT_PARENTHETICAL = re.compile(r"\s*\([^)]*\)\s*$")
_LLM_EXTRACTION_SIGNAL = re.compile(
    r"[:\n]|\d|\b(?:my name|i am|i live|incident|happened|paid|received|facts?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Narrative-field fidelity guard (draft-quality pass, Sept 2026)
#
# The incident: a user described a QR-code fraud across four sentences of
# Hindi. The LLM extraction pass returned `facts` as the single English
# sentence "A fraud of Rs 15,000 occurred via PhonePe yesterday, but the user
# does not have the UTR number." -- summarised, translated out of the user's
# language, and rewritten into the third person. That string was then stored
# as the FACTS OF THE CASE and rendered verbatim into a police complaint the
# user was invited to sign. Every specific the complaint needed (the phone
# call, the impersonation, the cashback pretext, the scan, the debit, the
# subsequent silence) was destroyed before drafting ever began, and no
# downstream length target could recover them.
#
# `entity_extraction_prompt.md` now forbids this explicitly, but a prompt is
# a request, not a guarantee -- and the cost of a silent regression here is a
# legal document that misrepresents its own author. So the result is checked
# in code as well: a narrative value that has lost the user's words, their
# language, or their voice is rejected, and the user's own text is used.
# ---------------------------------------------------------------------------

# Third-person referents that a first-person account should never contain. If
# the model produced one of these, it rewrote the user's voice.
_THIRD_PERSON_MARKERS = (
    "the user", "the applicant", "the complainant", "the customer", "the victim",
    "उपयोगकर्ता", "प्रयोगकर्ता", "आवेदक", "शिकायतकर्ता", "परिवादी",
    "वापरकर्ता", "अर्जदार", "તક્રારદાર", "ਵਰਤੋਂਕਾਰ", "ব্যবহারকারী",
    "பயனர்", "மனுதாரர்", "వినియోగదారు", "దరఖాస్తుదారు", "ಬಳಕೆದಾರ", "ಅರ್ಜಿದಾರ",
    "ഉപയോക്താവ്", "صارف", "درخواست گزار",
)
# A narrative value shorter than this fraction of the source message has
# summarised it away. Deliberately lenient: a user's message also contains
# greetings, other fields' answers, and asides, so a faithful `facts` value is
# legitimately shorter than the whole message -- just not four times shorter.
_MIN_NARRATIVE_RETENTION = 0.45


def _script_signature(text: str) -> frozenset[str]:
    """The set of Unicode script blocks present in `text`, coarsely.

    Compared between the source message and an extracted value to catch a
    silent translation: Devanagari in, pure Latin out.
    """
    blocks: set[str] = set()
    for char in text:
        code = ord(char)
        if char.isspace() or not char.isalpha():
            continue
        if code < 0x0250:
            blocks.add("latin")
        elif 0x0900 <= code <= 0x097F:
            blocks.add("devanagari")
        elif 0x0980 <= code <= 0x09FF:
            blocks.add("bengali")
        elif 0x0A00 <= code <= 0x0A7F:
            blocks.add("gurmukhi")
        elif 0x0A80 <= code <= 0x0AFF:
            blocks.add("gujarati")
        elif 0x0B00 <= code <= 0x0B7F:
            blocks.add("odia")
        elif 0x0B80 <= code <= 0x0BFF:
            blocks.add("tamil")
        elif 0x0C00 <= code <= 0x0C7F:
            blocks.add("telugu")
        elif 0x0C80 <= code <= 0x0CFF:
            blocks.add("kannada")
        elif 0x0D00 <= code <= 0x0D7F:
            blocks.add("malayalam")
        elif 0x0600 <= code <= 0x06FF:
            blocks.add("perso_arabic")
    return frozenset(blocks)


def is_faithful_narrative(value: str, source_message: str) -> bool:
    """Whether `value` is a faithful copy of the user's own narrative.

    Rejects three specific corruptions, each of which was observed in the
    reported incident:

    * **Summarised** -- the value retains too little of the source's length.
    * **Translated** -- the source carried a non-Latin script that the value
      has dropped entirely.
    * **Re-voiced** -- the value speaks about the user in the third person
      when the user did not.
    """
    value = (value or "").strip()
    source = (source_message or "").strip()
    if not value or not source:
        return bool(value)
    lowered_value = value.lower()
    lowered_source = source.lower()
    if any(
        marker in lowered_value and marker not in lowered_source
        for marker in _THIRD_PERSON_MARKERS
    ):
        return False
    source_scripts = _script_signature(source) - {"latin"}
    if source_scripts and not (source_scripts & _script_signature(value)):
        return False
    return len(value.split()) >= _MIN_NARRATIVE_RETENTION * len(source.split())


class DraftFieldExtractor:
    """Extracts template field values from a free-text chat message.

    Regex/heuristic extraction runs unconditionally, so structured fields
    (amount, date, mobile, email, a known bank/city) are picked up even with
    no LLM configured. When an LLM is available, a second structured-JSON
    extraction pass runs and its values win on conflict, since it can use
    context regex can't (e.g. correctly attributing a name or the narrative
    "facts" passage). This is best-effort NLU, not guaranteed-perfect
    extraction — anything missed is simply asked for directly by the
    conversation engine's follow-up questions, exactly like a field the user
    never mentioned at all.
    """

    def __init__(self) -> None:
        self.llm = LLMFactory.create()

    async def extract(
        self, text: str, template: DraftTemplateDefinition, language: str = "english"
    ) -> dict[str, str]:
        """Field values found in `text`, keyed by template field key.

        `language` is the language the FIELD LABELS were shown in, so a user
        answering the Hindi prompt with Hindi labels is understood. Defaults
        to English so every existing caller behaves exactly as before.
        """
        field_keys = template.field_keys()
        extracted = self._extract_with_regex(text, field_keys)
        for key, value in self._extract_by_label_prefix(text, template, language).items():
            if value and not extracted.get(key):
                extracted[key] = value
        # A form-shaped reply can already contain every required value. The
        # reported nine-field mobile-theft turn spent ~84 seconds on an
        # optional Gemini extraction pass even though deterministic label
        # parsing had everything it needed. Besides adding no information,
        # that call consumed most of the request budget before drafting even
        # began. Never call an LLM when the deterministic result is complete.
        missing_required = template.required_field_keys() - {
            key for key, value in extracted.items() if value
        }
        if not missing_required:
            return extracted
        # A short reply is normally one requested field value (for example a
        # department or police-station name). Sending every such message to a
        # provider is slow, unnecessary, and can turn a simple draft
        # continuation into a provider/network failure. Reserve the optional
        # LLM pass for genuinely narrative multi-detail input.
        if len(text.split()) >= 8 and _LLM_EXTRACTION_SIGNAL.search(text) and has_time_for(15.0):
            llm_extracted = await self._extract_with_llm(text, template)
            narrative_keys = {
                field.key for field in template.all_fields() if field.field_type == "textarea"
            }
            for key, value in llm_extracted.items():
                if not value:
                    continue
                # A narrative field is the document's substance -- the facts a
                # legal draft is built from. If the model summarised,
                # translated, or re-voiced it (see `is_faithful_narrative`),
                # the user's own words are used instead. Losing an extraction
                # is recoverable: the conversation engine simply asks for the
                # field. Silently substituting a paraphrase is not, because
                # nothing downstream can tell it apart from what the user
                # actually said.
                if key in narrative_keys and not is_faithful_narrative(value, text):
                    log.info(
                        "draft_narrative_extraction_rejected",
                        field=key,
                        template=template.draft_id,
                        source_words=len(text.split()),
                        extracted_words=len(value.split()),
                    )
                    # Only the primary narrative field falls back to the
                    # user's own words. Every OTHER narrative field the model
                    # returned used to be filled with the same whole message
                    # too, so a multi-message blob ended up as the applicant's
                    # address, relief and document list all at once -- and,
                    # being non-empty, blocked the real values later.
                    if not extracted.get(key) and key == "facts":
                        extracted[key] = text.strip()
                    continue
                extracted[key] = value
        return extracted

    def _extract_by_label_prefix(
        self, text: str, template: DraftTemplateDefinition, language: str = "english"
    ) -> dict[str, str]:
        """Catches explicitly-labeled answers regardless of how many fields
        are still missing, e.g. "Facts: I received a call..." or "Police
        Station: Hazratganj" -- the label the user themselves used (their
        field's label, key-as-words, or one of its `field_synonyms`) followed
        by "is"/"are"/":". Narrative (`textarea`) fields take everything up to
        the *next* labeled field in the same message (or the end of it).

        Short fields stop at the next comma/period/newline in a conversational
        message ("Police station Hazratganj, and it happened yesterday" --
        where the comma really is the end of the value). That rule is wrong
        for a FORM-SHAPED message, though, and form-shaped is exactly how a
        user answers when the engine has just printed a list of nine pending
        fields. Confirmed live, from a real session where the user pasted all
        nine labelled answers on one line:

          * `applicant_name` came back as "Rahul Sharma Expected Relief /
            What you want: Stolen mobile phone ko trace/recover karke..." --
            there is no comma between the name and the NEXT field's label, so
            the value ran straight through it;
          * `imei_number` came back as "352099123456789 Location Phone Was
            Stolen From: Ghanta Ghar Market", which then failed IMEI
            validation and was discarded, so the field was asked for again;
          * `police_station`, conversely, was truncated at its first comma to
            "Kotwali Nagar Police Station", losing "Ghaziabad".

        So when the message carries two or more labelled fields it is treated
        as a form: short values run to the next LABEL (or a sentence-ending
        period/newline) rather than to the next comma, which fixes the run-on
        and the truncation together. A single-label message keeps the
        original comma-stopping behaviour, where it is still the better read.
        """
        candidates_by_key = self._label_candidates(template, language)

        all_candidates = sorted(
            {candidate for candidates in candidates_by_key.values() for candidate in candidates},
            key=len,
            reverse=True,
        )
        boundary = "|".join(re.escape(candidate) for candidate in all_candidates)
        # A value ends where the NEXT label begins, or at a newline, or at the
        # end of the message. Deliberately NOT at a period: "ABC Electronics
        # Pvt. Ltd." was being truncated to "ABC Electronics Pvt", and a
        # respondent's name is not something to guess the end of.
        # Printed collection forms number each label ("1. Purpose", "2.
        # Address"). Permit that numbering before the next label so a textarea
        # value cannot swallow every later field in the pasted form.
        #
        # QA session 2026-09-24 ("BUG-104 follow-up"): `\d+` here (unbounded)
        # matched a 4-digit YEAR immediately followed by a sentence-ending
        # period as if it were list numbering -- "date of return memo is
        # 10-08-2026. reason for dishonour is ..." matched this whole
        # `next_label` pattern starting at "2026. reason for dishonour is",
        # so the value capture (which stops right before `next_label`) lost
        # everything from "2026" onward, extracting "10-08" instead of
        # "10-08-2026". Bounding the digit count to `\d{1,2}` alone was not
        # enough: it then matched just the LAST two digits of the year
        # ("26.") the same way, truncating to "10-08-20" instead. The
        # `(?<!\d)` lookbehind is the actual fix -- it requires the "list
        # number" not be immediately preceded by another digit, so it can
        # never match the tail end of a longer run like a year, while a
        # real list marker ("1.", "2)", ...) is never preceded by a digit
        # either and matches exactly as before.
        next_label = rf"\s*(?:(?<!\d)\d{{1,2}}[.)]\s*)?(?:{boundary}){_LABEL_SEPARATOR}" if boundary else ""
        form_stop_pattern = rf"(.+?)(?={next_label}|\n|$)" if boundary else r"(.+)$"
        # How many DISTINCT fields this message labels. Two or more means the
        # user is filling in a form, not talking, and short values must run to
        # the next label instead of the next comma (see the docstring).
        labelled_field_count = (
            sum(
                any(
                    re.search(re.escape(candidate) + _LABEL_SEPARATOR, text, re.IGNORECASE)
                    for candidate in candidates
                )
                for candidates in candidates_by_key.values()
            )
            if boundary
            else 0
        )
        short_stop_pattern = form_stop_pattern if labelled_field_count >= 2 else r"([^,.\n]+)"

        found: dict[str, str] = {}
        for draft_field in template.all_fields():
            stop_pattern = form_stop_pattern if draft_field.field_type == "textarea" else short_stop_pattern
            for candidate in sorted(candidates_by_key[draft_field.key], key=len, reverse=True):
                pattern = re.compile(
                    re.escape(candidate) + _LABEL_SEPARATOR + stop_pattern, re.IGNORECASE | re.DOTALL
                )
                match = pattern.search(text)
                if match:
                    value = match.group(1).strip().strip("-–—,;")
                    # A trailing period is never part of a date VALUE itself
                    # (it's the sentence's own full stop) the way it can
                    # legitimately be part of a name/textarea value ("ABC
                    # Electronics Pvt. Ltd." -- the general stripping above
                    # deliberately leaves internal/trailing periods alone for
                    # exactly that reason). Stripped only for `date` fields,
                    # narrowly: "date of return memo is 10-08-2026. reason
                    # for dishonour is ..." previously left the captured
                    # value as "10-08-2026." with the trailing period from
                    # its own sentence still attached, which then failed
                    # date-format validation and the field went back to
                    # "missing" despite having been correctly captured.
                    if draft_field.field_type == "date":
                        value = value.rstrip(".").strip()
                    if value:
                        found[draft_field.key] = value
                    break
        return found

    @staticmethod
    def _label_candidates(
        template: DraftTemplateDefinition, language: str = "english"
    ) -> dict[str, set[str]]:
        """Every wording a user might use to name each field.

        Four sources, and each one was a real gap:

        * the English label, and the label with any trailing parenthetical
          removed -- the engine prints "Facts of the Case (in your own
          words)" and the user types "Facts of the Case:", which matched
          nothing at all;
        * the field key as words ("respondent address");
        * the template's hand-authored `field_synonyms`;
        * the label as LOCALIZED for `language` -- the label the user was
          actually shown. A Hindi speaker answering the Hindi prompt with
          "आवेदक का नाम:" previously matched no candidate, so the whole
          message fell through to a single field or to RAG.
        """
        candidates_by_key: dict[str, set[str]] = {}
        for draft_field in template.all_fields():
            candidates = {
                draft_field.label.lower(),
                _WITHOUT_PARENTHETICAL.sub("", draft_field.label).strip().lower(),
                draft_field.key.replace("_", " ").lower(),
                # QA session 2026-09-24 ("BUG-104"): the RAW key (underscore
                # intact), not just the space-joined form above. `DraftSummary`/
                # `DraftTurnInfo.missing_fields` (what the API actually shows
                # the caller, e.g. `["dishonour_date"]`) always prints the raw
                # key, never the space-joined display form -- so a caller who
                # reasonably answers using the exact string the API told them
                # was missing ("dishonour_date is 10-08-2026") previously
                # matched no candidate at all, only the differently-spelled
                # "dishonour date" did. Live-reproduced: 3 straight attempts
                # using the API's own reported field name failed to fill it;
                # only discovering and switching to the human-facing label
                # ("date of return memo") worked.
                draft_field.key.lower(),
                draft_field.hindi_label.lower(),
                _WITHOUT_PARENTHETICAL.sub("", draft_field.hindi_label).strip().lower(),
            }
            if language and language != "english":
                localized = localized_field_label(template.draft_id, draft_field, language)
                candidates.add(localized.lower())
                candidates.add(_WITHOUT_PARENTHETICAL.sub("", localized).strip().lower())
            for synonym, field_key in template.field_synonyms.items():
                if field_key == draft_field.key:
                    candidates.add(synonym.lower())
            candidates_by_key[draft_field.key] = {
                candidate for candidate in candidates if len(candidate) >= 3
            }
        return candidates_by_key

    def _extract_with_regex(self, text: str, field_keys: set[str]) -> dict[str, str]:
        found: dict[str, str] = {}

        if "applicant_mobile" in field_keys:
            match = _MOBILE_PATTERN.search(text)
            if match:
                found["applicant_mobile"] = match.group(1)

        # "Mera address: 21, Andheri West, Mumbai - 400053. Mobile: ..." and
        # "Landlord ka address: 14, Bandra East, ..." -- the possessive
        # prefix stops the generic label matcher (which expects the field's
        # own label, "Applicant Address") from recognising them, and the
        # optional LLM pass is skipped for short replies or fails on quota.
        for owner, target in ((r"(?:mera|meri|my)", "applicant_address"),
                              (r"(?:landlord|malik|uska|unka|respondent|opposite\s+party)(?:\s+(?:ka|ki))?", "respondent_address")):
            if target in field_keys:
                m = re.search(
                    rf"\b{owner}\s+(?:ghar\s+ka\s+)?(?:pata|address)\s*[:：-]\s*"
                    r"(.+?)(?=\.\s+(?:mobile|phone|contact|landlord|mera|ab|email)\b|\n|$)",
                    text, re.IGNORECASE,
                )
                if m and m.group(1).strip(" .,"):
                    found[target] = m.group(1).strip(" .,")

        # A bare "Address: ..." in a labelled block ("Applicant: Ankit Verma.
        # Address: 45, Saraswati Vihar, ...") is the applicant's own; the
        # generic matcher only knew the field's full label "Applicant Address".
        if "applicant_address" in field_keys and "applicant_address" not in found:
            m = re.search(
                r"(?:^|[.\n]\s*)address\s*[:：]\s*(.+?)(?=\.\s+(?:mobile|phone|contact|police|place|email)\b|\n|$)",
                text, re.IGNORECASE,
            )
            if m and m.group(1).strip(" .,"):
                found["applicant_address"] = m.group(1).strip(" .,")

        # "... SHO ko FIR registration ke liye complaint draft karo": asking
        # for the FIR to be registered IS the relief sought.
        if (
            "expected_relief" in field_keys
            and "police_station" in field_keys
            and re.search(
                r"\bfir\b[^.]{0,40}\b(registration|register|darj)\b|\b(registration|register)\b[^.]{0,20}\bfir\b",
                text, re.IGNORECASE,
            )
        ):
            found["expected_relief"] = "Registration of an FIR and investigation into the reported incident."

        if "applicant_email" in field_keys:
            match = _EMAIL_PATTERN.search(text)
            if match:
                found["applicant_email"] = match.group(1)

        amount_key = next((key for key in _AMOUNT_FIELD_CANDIDATES if key in field_keys), None)
        if amount_key:
            match = _AMOUNT_PATTERN.search(text)
            if match:
                found[amount_key] = match.group(1)

        date_key = next((key for key in _DATE_FIELD_CANDIDATES if key in field_keys), None)
        if date_key:
            match = _DATE_PATTERN.search(text)
            if match:
                found[date_key] = match.group(1)

        if "bank_name" in field_keys:
            for bank in _KNOWN_BANKS:
                if re.search(rf"\b{re.escape(bank)}\b", text, re.IGNORECASE):
                    found["bank_name"] = bank
                    break

        if "city" in field_keys:
            for city in _KNOWN_CITIES:
                if re.search(rf"\b{re.escape(city)}\b", text, re.IGNORECASE):
                    found["city"] = city
                    break

        if "applicant_name" in field_keys:
            for pattern in _NAME_PATTERNS:
                match = pattern.search(text)
                if match:
                    found["applicant_name"] = match.group(1).strip().rstrip(".")
                    break

        if "document_identifier" in field_keys:
            match = _DOCUMENT_IDENTIFIER_PATTERN.search(text)
            if match:
                found["document_identifier"] = match.group(1).strip()

        if "lost_document_name" in field_keys:
            match = _HINDI_LOST_DOCUMENT_PATTERN.search(text)
            if match:
                found["lost_document_name"] = match.group(1).strip()

        if "loss_place" in field_keys:
            match = _HINDI_LOSS_PLACE_PATTERN.search(text)
            if match:
                found["loss_place"] = match.group(1).strip()

        return found

    async def _extract_with_llm(self, text: str, template: DraftTemplateDefinition) -> dict[str, str]:
        field_descriptions = "\n".join(
            f"- {field.key}: {field.label} ({field.hindi_label})" for field in template.all_fields()
        )
        prompt = prompt_registry.render("entity_extraction_prompt", field_descriptions=field_descriptions, text=text)
        try:
            response = await self.llm.chat([ChatMessage(role="user", content=prompt)], temperature=0.0)
        # LLM-assisted extraction is an optional enhancement over the
        # rule-based pass, so any provider failure degrades to "no extra
        # fields" rather than breaking drafting. Logged, because an empty
        # result is otherwise indistinguishable from a document that
        # genuinely had nothing to extract.
        except Exception as exc:  # noqa: BLE001 - optional enhancement; must never break drafting
            log.warning("llm_field_extraction_failed", error=str(exc), error_type=type(exc).__name__)
            return {}
        content = response.content.strip()
        if not content:
            return {}
        content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        if not isinstance(parsed, dict):
            return {}
        field_keys = template.field_keys()
        return {
            key: str(value).strip()
            for key, value in parsed.items()
            if key in field_keys and isinstance(value, (str, int, float)) and str(value).strip()
        }


# Notice-type documents whose relief IS the amount demanded.
_DEMAND_RELIEF_TEMPLATES = frozenset({"demand_notice", "recovery_notice", "rent_notice", "legal_notice"})


def infer_derivable_fields(template: DraftTemplateDefinition, fields: dict[str, str]) -> dict[str, str]:
    """Values that follow directly from facts the user ALREADY gave, so the
    draft is not held up asking for them again.

    * `place` -- the city in the applicant's own address ("21, Andheri West,
      Mumbai - 400053" -> "Mumbai"). Only a well-known city named in an
      address the user supplied; never guessed.
    * `expected_relief` of a demand-type notice -- payment of the amount the
      user stated as demanded/due. No amount, no inference.

    Never overwrites a value the user supplied.
    """
    from app.rag.jurisdiction import CITY_TO_STATE

    inferred: dict[str, str] = {}
    keys = template.field_keys()
    if "place" in keys and not fields.get("place"):
        cities = sorted(CITY_TO_STATE, key=len, reverse=True)
        for source in ("applicant_address", "complainant_address", "respondent_address"):
            address = (fields.get(source) or "").lower()
            city = next((c for c in cities if re.search(rf"\b{re.escape(c)}\b", address)), None)
            if city:
                inferred["place"] = city.title()
                break
    if (
        "expected_relief" in keys
        and not fields.get("expected_relief")
        and template.draft_id in _DEMAND_RELIEF_TEMPLATES
    ):
        amount = next((fields.get(k) for k in _AMOUNT_FIELD_CANDIDATES if fields.get(k)), None)
        if amount:
            inferred["expected_relief"] = f"Payment of Rs. {amount} as demanded in this notice."
    return inferred
