"""Phase 1 item 1: cross-lingual legal retrieval.

The knowledge base is overwhelmingly English (bare Acts, Gazette PDFs, FAQ
material), while a large share of real questions arrive in Hindi, Hinglish,
Marathi, Gujarati, Tamil, Telugu, Bengali, Urdu and the other Eighth Schedule
languages. Multilingual embeddings carry a lot of that gap, but not all of it:
retrieval also runs a BM25 leg (pure lexical -- a Gujarati query and an English
statute share literally zero tokens) and, downstream, `ChatService.
_is_relevant_chunk` applies a lexical-overlap gate before anything reaches the
LLM. Both are token-level, so a correct chunk could be retrieved and then
rejected for the sole reason that the question was not asked in English.

That produced the worst possible failure for a multilingual product: "no
verified document in the Knowledge Base" for a provision the knowledge base
demonstrably contains, purely because of the language it was asked in.

This module closes it by translating the LEGAL CONCEPTS in a query into the
statutory English vocabulary the corpus actually uses, and handing those to
the rewriter as extra search variants. It is deliberately a curated concept
bridge rather than a general-purpose translator:

* It only has to carry the handful of words that decide WHICH provision is
  relevant ("जमानत" -> bail, "છેતરપિંડી" -> cheating). Everything else in the
  sentence is already handled by the multilingual embedding leg.
* It is deterministic and auditable. A machine-translation call here would add
  latency and a failure mode to the retrieval hot path, and could silently
  mistranslate a term of art -- with no way to tell afterwards.
* An unrecognised query loses nothing: it falls through with the same
  behaviour as before this module existed.

Terms are matched as substrings, not `\\b`-delimited words: Python's `\\b`
is defined in terms of `\\w`, which excludes Indic combining marks (matras,
virama), so a word boundary essentially never exists at the end of a
Devanagari/Tamil/Telugu/Bengali word. The terms below are long and
distinctive enough that substring matching is safe; the few short romanized
ones are matched as whole words instead (see `_ROMANIZED_WORD_TERMS`).
"""

import re
import unicodedata

# One entry per legal concept: the English search vocabulary the corpus itself
# uses, and the words a user might reach for in any supported language.
#
# The English side deliberately names the statute's own phrasing (what BM25
# needs to match on), not a dictionary gloss -- "criminal intimidation" rather
# than "threatening", "deficiency in service" rather than "bad service".
# Carries the phrases `LegalReranker._topic_bonus` keys its CPC s.80 handling on.
_GOVERNMENT_SUIT_EXPANSION = (
    "notice before suit against Government or public officer Code of Civil Procedure Section 80 "
    "suit against government"
)

_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "first information report FIR police complaint cognizable offence registration BNSS Section 173",
        (
            "प्राथमिकी", "एफआईआर", "पुलिस शिकायत", "पुलिस में शिकायत", "थाने में शिकायत",
            "पोलीस तक्रार", "एफआयआर",
            "પોલીસ ફરિયાદ", "એફઆઈઆર",
            "காவல் புகார்", "முதல் தகவல் அறிக்கை",
            "పోలీసు ఫిర్యాదు", "ఎఫ్ఐఆర్",
            "পুলিশ অভিযোগ", "এফআইআর",
            "پولیس شکایت", "ایف آئی آر",
            "ಪೊಲೀಸ್ ದೂರು", "പോലീസ് പരാതി", "ਪੁਲਿਸ ਸ਼ਿਕਾਇਤ", "ਪੁਲਿਸ ਨੂੰ ਸ਼ਿਕਾਇਤ", "ପୋଲିସ ଅଭିଯୋଗ",
            # Added after qa-40q-multilingual-20260921: the abbreviation, the
            # "cognizable offence" / "police station" vocabulary that a
            # FIR-refusal question is actually phrased in, and the Hindi-
            # loanword "अभियोग" that Bodo and Maithili speakers use for a
            # police complaint. (The ZWNJ variant of the Kannada abbreviation
            # is handled by `_normalize_for_match`, not listed separately.)
            "एफ़आईआर", "जीरो एफआईआर", "प्रथम सूचना रिपोर्ट", "संज्ञेय अपराध", "पुलिस थाना", "पुलिस स्टेशन",
            "थाने में", "पुलिस अभियोग", "पुलिस रिपोर्ट", "रिपोर्ट दर्ज",
            "प्रथम खबर अहवाल", "दखलपात्र गुन्हा", "पोलीस ठाणे", "पोलीस स्टेशन",
            "પ્રથમ માહિતી અહેવાલ", "સંજ્ઞેય ગુનો", "પોલીસ સ્ટેશન", "ફરિયાદ નોંધ",
            "காவல் நிலையம்", "புகார் பதிவு", "அறியத்தக்க குற்றம்",
            "పోలీస్ స్టేషన్", "పోలీసు స్టేషన్", "మొదటి సమాచార నివేదిక", "సంజ్ఞేయ నేరం",
            "থানায়", "থানা", "পুলিশ স্টেশন", "প্রাথমিক তথ্য প্রতিবেদন", "আমলযোগ্য অপরাধ",
            "پولیس اسٹیشن", "تھانے", "ابتدائی اطلاعاتی رپورٹ",
            "ಎಫ್ಐಆರ್", "ಪ್ರಥಮ ಮಾಹಿತಿ ವರದಿ", "ಸಂಜ್ಞೇಯ ಅಪರಾಧ", "ಪೊಲೀಸ್ ಠಾಣೆ", "ದೂರು ದಾಖಲ",
            "എഫ്ഐആർ", "എഫ്.ഐ.ആർ", "പോലീസ് സ്റ്റേഷൻ", "എഫ്‌ഐആർ",
            "ਐਫ਼ਆਈਆਰ", "ਐਫਆਈਆਰ", "ਪੁਲਿਸ ਸਟੇਸ਼ਨ", "ਥਾਣੇ",
            "ଏଫଆଇଆର", "ଥାନା", "ପୋଲିସ୍ ଅଭିଯୋଗ",
            "प्रहरी उजुरी", "प्रहरीमा उजुरी", "प्रहरी कार्यालय",
            "এফআইআৰ", "আৰক্ষী থানা",
        ),
    ),
    (
        "bail bail application bailable offence bail bond release from custody BNSS",
        (
            "जमानत", "ज़मानत", "जमानत अर्जी",
            "जामीन", "જામીન", "பிணை", "బెయిల్", "জামিন", "ضمانت",
            "ಜಾಮೀನು", "ജാമ്യം", "ਜ਼ਮਾਨਤ", "ଜାମିନ",
        ),
    ),
    (
        "anticipatory bail pre-arrest bail apprehending arrest BNSS Section 482",
        (
            "अग्रिम जमानत", "अग्रिम ज़मानत", "अटकपूर्व जामीन", "આગોતરા જામીન",
            "முன்ஜாமீன்", "ముందస్తు బెయిల్", "আগাম জামিন", "پیشگی ضمانت",
        ),
    ),
    (
        "cheating dishonestly inducing delivery of property fraud Bharatiya Nyaya Sanhita Section 318",
        (
            "धोखाधड़ी", "ठगी", "धोखा", "छल", "फ्रॉड",
            # Marathi "फसवणूक" appears as "फसवणुकीसाठी"/"फसवणुकीची" in real
            # sentences (ू -> ु before the case suffix), and Telugu "మోసం"
            # as "మోసానికి" -- the citation form alone matches neither, so
            # the stem is listed alongside it.
            "फसवणूक", "फसवणुक", "फसवण", "છેતરપિંડી", "மோசடி", "ஏமாற்று", "మోసం", "మోస",
            "প্রতারণা", "জালিয়াতি", "دھوکہ", "فراڈ",
            "ವಂಚನೆ", "ವಂಚನ", "തട്ടിപ്പ്", "ਧੋਖਾਧੜੀ", "ଠକେଇ",
        ),
    ),
    (
        "punishment imprisonment fine sentence prescribed penalty",
        (
            "सजा", "सज़ा", "दंड", "दण्ड",
            "शिक्षा", "સજા", "தண்டனை", "శిక్ష", "শাস্তি", "سزا",
            "ಶಿಕ್ಷೆ", "ശിക്ഷ", "ਸਜ਼ਾ", "ଦଣ୍ଡ",
        ),
    ),
    (
        "cyber crime online fraud UPI unauthorised electronic transaction Information Technology Act",
        (
            "साइबर", "ऑनलाइन धोखाधड़ी", "ऑनलाइन ठगी",
            "सायबर", "સાયબર", "சைபர்", "సైబర్", "সাইবার", "سائبر",
            "ಸೈಬರ್", "സൈബർ", "ਸਾਈਬਰ", "ସାଇବର",
        ),
    ),
    (
        "consumer complaint Consumer Protection Act deficiency in service unfair trade practice refund",
        (
            "उपभोक्ता", "ग्राहक शिकायत", "रिफंड",
            "ग्राहक तक्रार", "ગ્રાહક ફરિયાદ", "நுகர்வோர்", "వినియోగదారు",
            "ভোক্তা", "صارف", "ಗ್ರಾಹಕ", "ഉപഭോക്തൃ", "ਖਪਤਕਾਰ", "ଗ୍ରାହକ",
            # Santali (Ol Chiki) writes the loanword phonetically; the
            # qa-40q-multilingual-20260921 Santali consumer question produced
            # no bridge at all and was queued as a missing Consumer Act.
            "ᱠᱚᱱᱥᱩᱢᱟᱨ",
        ),
    ),
    (
        "divorce dissolution of marriage judicial separation marriage act",
        (
            "तलाक", "विवाह विच्छेद",
            "घटस्फोट", "છૂટાછેડા", "விவாகரத்து", "విడాకులు",
            "বিবাহবিচ্ছেদ", "طلاق", "ವಿಚ್ಛೇದನ", "വിവാഹമോചനം", "ਤਲਾਕ", "ଛାଡପତ୍ର",
        ),
    ),
    (
        # Distinct from the divorce concept above: "husband/wife living
        # separately without reason" names Hindu Marriage Act Section 9
        # (restitution of conjugal rights), not Section 13. Confirmed live
        # (2026-09-25): a Hindi question phrased this way never retrieved the
        # Hindu Marriage Act at all, unlike a question naming "divorce"/
        # "talaq" directly.
        "restitution of conjugal rights Hindu Marriage Act Section 9 living separately without reasonable excuse",
        (
            "वैवाहिक अधिकार", "दांपत्य अधिकार", "अलग रह रही", "अलग रह रहा",
            "વૈવાહિક અધિકાર", "தாம்பத்திய உரிமை", "వైవాహిక హక్కులు",
            "দাম্পত্য অধিকার", "ازدواجی حقوق", "ವೈವಾಹಿಕ ಹಕ್ಕು", "ദാമ്പത്യാവകാശം",
            "ਵਿਆਹੁਤਾ ਅਧਿਕਾਰ", "ଦାମ୍ପତ୍ୟ ଅଧିକାର",
        ),
    ),
    (
        "maintenance of wife children and parents monthly allowance BNSS Section 144",
        (
            "भरण पोषण", "भरण-पोषण", "गुजारा भत्ता",
            "पोटगी", "ભરણપોષણ", "ஜீவனாம்சம்", "భరణం", "খোরপোশ", "نان نفقہ",
        ),
    ),
    (
        "security deposit tenant landlord rent agreement refund of deposit property law",
        (
            "किराया", "किरायेदार", "मकान मालिक", "सुरक्षा जमा",
            "भाडे", "भाडेकरू", "घरमालक",
            "ભાડું", "ભાડૂત", "મકાનમાલિક",
            "வாடகை", "குத்தகைதாரர்", "అద్దె", "కిరాయిదారు",
            "ভাড়া", "ভাড়াটে", "বাড়িওয়ালা", "کرایہ", "کرایہ دار",
            "ಬಾಡಿಗೆ", "വാടക", "ਕਿਰਾਇਆ", "ଭଡ଼ା",
        ),
    ),
    (
        "cheque bounce dishonour of cheque Negotiable Instruments Act Section 138",
        (
            "चेक बाउंस", "चेक अनादर", "चेक बाउन्स",
            "चेक बाऊन्स", "ચેક બાઉન્સ", "காசோலை", "చెక్కు", "চেক বাউন্স", "چیک باؤنس",
            # "cheque came back" is how most people say it; the bare instrument
            # noun plus a return verb is caught by `_PAIRED_CONCEPTS`, these
            # are the fixed phrases that need no second word.
            "चेक वापस", "चेक रिटर्न", "चेक डिसऑनर", "ચેક પરત", "ચેક રિટર્ન", "ચેક અનાદર",
            "ಚೆಕ್ ಬೌನ್ಸ್", "ಚೆಕ್ ಅಮಾನ್ಯ", "ചെക്ക് ബൗൺസ്", "ചെക്ക് മടങ്ങി", "ਚੈੱਕ ਬਾਊਂਸ", "ਚੈੱਕ ਵਾਪਸ",
            "ଚେକ ବାଉନ୍ସ", "ଚେକ ଫେରି", "காசோலை திரும்ப", "చెక్కు బౌన్స్", "চেক ফেরত",
            # See the paired-instrument-group comment in `_PAIRED_CONCEPTS`
            # below for why these short forms are separate, confirmed-real
            # spellings, not duplicates of the entries just above.
            "چيڪ باؤنس", "చెక్ బౌన్స్",
        ),
    ),
    (
        "arrest without warrant cognizable offence grounds of arrest rights of arrested person BNSS Section 35",
        (
            "गिरफ्तार", "गिरफ्तारी", "वारंट", "हिरासत",
            "अटक", "ધરપકડ", "கைது", "అరెస్టు", "গ্রেপ্তার", "گرفتار",
            "ಬಂಧನ", "അറസ്റ്റ്", "ਗ੍ਰਿਫ਼ਤਾਰ", "ଗିରଫ",
        ),
    ),
    (
        "criminal intimidation threat to cause injury stalking harassment Bharatiya Nyaya Sanhita",
        (
            "धमकी", "धमका", "परेशान", "प्रताड़ना", "पीछा कर",
            "धमकाव", "त्रास", "ધમકી", "પરેશાન",
            # Verb stems alongside the nouns: a user writes "மிரட்டுகிறார்"
            # ("he is threatening") and "బెదిరిస్తున్నారు", not the citation
            # nouns "மிரட்டல்"/"బెదిరింపు".
            "மிரட்டல்", "மிரட்ட", "தொல்லை", "బెదిరింపు", "బెదిరి", "వేధింపు",
            "হুমকি", "উত্ত্যক্ত", "دھمکی", "ہراسانی",
            "ಬೆದರಿಕೆ", "ഭീഷണി", "ਧਮਕੀ", "ଧମକ",
        ),
    ),
    (
        "domestic violence protection of women cruelty by husband or relatives dowry",
        (
            "घरेलू हिंसा", "दहेज", "क्रूरता",
            "घरगुती हिंसा", "हुंडा", "ઘરેલુ હિંસા", "દહેજ",
            "குடும்ப வன்முறை", "வரதட்சணை", "గృహ హింస", "కట్నం",
            "পারিবারিক সহিংসতা", "যৌতুক", "گھریلو تشدد", "جہیز",
            "ಕೌಟುಂಬಿಕ ಹಿಂಸೆ", "ವರದಕ್ಷಿಣೆ", "ഗാർഹിക പീഡനം", "സ്ത്രീധനം", "ਘਰੇਲੂ ਹਿੰਸਾ", "ਦਾਜ",
        ),
    ),
    (
        "theft dishonestly taking movable property Bharatiya Nyaya Sanhita Section 303",
        (
            "चोरी", "चोरी हो गई",
            "ચોરી", "திருட்டு", "దొంగతనం", "চুরি", "چوری",
            "ಕಳ್ಳತನ", "മോഷണം", "ਚੋਰੀ", "ଚୋରି",
            "ಕಳವು", "దొంగిలి", "চুৰি",
        ),
    ),
    (
        "Right to Information Act public authority public information officer first appeal",
        (
            "सूचना का अधिकार", "आरटीआई",
            "माहितीचा अधिकार", "માહિતી અધિકાર", "தகவல் அறியும் உரிமை",
            "సమాచార హక్కు", "তথ্য অধিকার", "حق اطلاعات",
            # Added after qa-40q-multilingual-20260921 (Odia RTI question got
            # no bridge at all): the local spelling of the Act's own title in
            # each remaining language, and the transliterated "RTI" acronym.
            "सूचना अधिकार", "आर टी आई", "सूचनाको हक", "सूचनाको अधिकार", "सूचनाक अधिकार", "सूचनाधिकार",
            "माहिती अधिकार", "માહિતીનો અધિકાર", "આરટીઆઈ",
            "ସୂଚନା ଅଧିକାର", "ଆରଟିଆଇ", "ಮಾಹಿತಿ ಹಕ್ಕು", "ಆರ್‌ಟಿಐ", "വിവരാവകാശ", "ആർടിഐ",
            "ਸੂਚਨਾ ਦਾ ਅਧਿਕਾਰ", "ਸੂਚਨਾ ਅਧਿਕਾਰ", "ਆਰਟੀਆਈ", "ஆர்டிஐ", "ఆర్టీఐ", "আরটিআই", "তথ্যৰ অধিকাৰ",
            "معلومات کا حق", "حق معلومات", "آر ٹی آئی",
        ),
    ),
    (
        "unpaid salary wages employment dues labour law appointment letter",
        (
            "वेतन", "तनख्वाह", "मजदूरी", "पगार",
            "પગાર", "சம்பளம்", "జీతం", "বেতন", "تنخواہ",
            "ಸಂಬಳ", "ശമ്പളം", "ਤਨਖਾਹ", "ଦରମା",
            # Regional spellings the first list missed: "तनखा" (Hindi/Maithili
            # colloquial -- not "तनख्वाह") and "दरमाहा" (Maithili/Bhojpuri),
            # plus the "wage"-root words (ವೇತನ, ஊதியம்...) used in formal
            # written complaints. "तलब" is deliberately absent even though it
            # means salary in Nepali: in Hindi/Urdu it also means "summoned"
            # ("अदालत ने तलब किया"), and a court-summons question must not be
            # bridged to a wages search.
            "तनखा", "तनख़ा", "तनखाह", "दरमाहा", "बकाया वेतन", "मासिक वेतन", "सैलरी",
            "पारिश्रमिक", "મજૂરી", "સેલરી", "ವೇತನ", "വേതനം", "ஊதியம்", "వేతనం",
            "মাইনে", "মজুরি", "ਤਨਖ਼ਾਹ", "ମଜୁରୀ", "اجرت", "দৰমহা",
        ),
    ),
    (
        "property ownership dispute land transfer of property registration act possession",
        (
            "संपत्ति", "सम्पत्ति", "जमीन", "ज़मीन", "कब्जा",
            "मालमत्ता", "મિલકત", "જમીન", "சொத்து", "நிலம்",
            "ఆస్తి", "భూమి", "সম্পত্তি", "জমি", "جائیداد", "زمین",
            "ಆಸ್ತಿ", "സ്വത്ത്", "ਜਾਇਦਾਦ", "ସମ୍ପତ୍ତି",
        ),
    ),
    (
        "legal notice formal demand notice advocate reply civil dispute",
        (
            "कानूनी नोटिस", "विधिक नोटिस", "कायदेशीर नोटीस",
            "કાનૂની નોટિસ", "சட்ட அறிவிப்பு", "చట్టపరమైన నోటీసు",
            "আইনি নোটিশ", "قانونی نوٹس",
            # "विधिसूचना" is the Sanskrit/formal-Hindi rendering of "legal
            # notice" (qa-40q-multilingual-20260921 Q19); the rest are the
            # transliterated "legal notice" and the local words for "legal".
            "विधिसूचना", "विधि सूचना", "विधिक सूचना", "कानूनी सूचना", "लीगल नोटिस", "वकील का नोटिस",
            "कानुनी नोटिस", "कानुनी सूचना", "लीगल नोटीस", "લીગલ નોટિસ", "કાનૂની નોટીસ",
            "ಕಾನೂನು ನೋಟಿಸ್", "ಲೀಗಲ್ ನೋಟಿಸ್", "നിയമ നോട്ടീസ്", "ലീഗൽ നോട്ടീസ്",
            "ਕਾਨੂੰਨੀ ਨੋਟਿਸ", "ਲੀਗਲ ਨੋਟਿਸ", "ଆଇନଗତ ନୋଟିସ", "ଲିଗାଲ ନୋଟିସ",
            "சட்ட நோட்டீஸ்", "లీగల్ నోటీసు", "লিগ্যাল নোটিশ", "لیگل نوٹس",
        ),
    ),
    (
        "sexual harassment of women at workplace internal committee POSH inquiry",
        (
            "कार्यस्थल पर उत्पीड़न", "यौन उत्पीड़न",
            "लैंगिक छळ", "જાતીય સતામણી", "பாலியல் துன்புறுத்தல்",
            "లైంగిక వేధింపు", "যৌন হয়রানি", "جنسی ہراسانی",
        ),
    ),
    (
        "legal rights of a citizen fundamental rights remedies available",
        (
            "अधिकार", "कानूनी अधिकार",
            "हक्क", "અધિકાર", "உரிமை", "హక్కు", "অধিকার", "حقوق",
            "ಹಕ್ಕು", "അവകാശം", "ਅਧਿਕਾਰ", "ଅଧିକାର",
        ),
    ),
    (
        "section of the Act provision statutory text",
        (
            "धारा", "कलम", "કલમ", "பிரிவு", "సెక్షన్", "ধারা", "دفعہ",
            "ವಿಭಾಗ", "വകുപ്പ്", "ਧਾਰਾ", "ଧାରା",
        ),
    ),
    (
        "procedure for filing an application before the authority required documents",
        (
            "प्रक्रिया", "कैसे दर्ज", "आवेदन कैसे",
            "प्रक्रिया काय", "પ્રક્રિયા", "நடைமுறை", "విధానం", "প্রক্রিয়া", "طریقہ کار",
        ),
    ),
    (
        # Confirmed live (2026-09-25): the raw candidate pool did contain
        # Arbitration and Conciliation Act / Mediation Act chunks (the query
        # already named them in English), but they had no reranker topic
        # bonus and lost the top-6 cut to unrelated State Acts. This bridge
        # is what carries the concept when asked in a native script instead.
        "arbitration mediation conciliation alternative dispute resolution Arbitration and Conciliation Act "
        "1996 Mediation Act 2023 arbitral tribunal arbitral award",
        (
            "मध्यस्थता", "पंचाट", "सुलह", "मध्यस्थ", "विवाद समाधान",
            "લવાદ", "મધ્યસ્થી", "நடுவர்", "மத்தியஸ்தம்", "మధ్యవర్తిత్వం", "మధ్యవర్తి",
            "মধ্যস্থতা", "সালিশ", "ثالثی",
            "ಮಧ್ಯಸ್ಥಿಕೆ", "മധ്യസ്ഥത", "ਸਾਲਸੀ", "ମଧ୍ୟସ୍ଥତା",
        ),
    ),
    # Appended (not inserted) so that `legal_intent_hint`'s first-match order
    # over the concepts above is unchanged.
    (
        # The trigger phrases here ("valid contract", "essential elements",
        # "offer and acceptance", "contract act") are the ones `LegalReranker.
        # _topic_bonus` and `relevance.is_relevant_chunk` key their contract
        # handling on -- a non-English contract question reaches Indian
        # Contract Act s.10 only if its expansion carries them.
        (
            "valid contract essential elements offer and acceptance free consent lawful consideration "
            "competent parties lawful object Indian Contract Act Section 10"
        ),
        (
            "करार", "अनुबंध", "अनुबन्ध", "संविदा", "सम्विदा", "इकरारनामा", "सम्झौता",
            "કરાર", "ஒப்பந்தம்", "ఒప్పందం", "ಒಪ್ಪಂದ", "കരാർ", "চুক্তি", "ਇਕਰਾਰਨਾਮਾ", "ਇਕਰਾਰ",
            "ଚୁକ୍ତି", "معاہدہ", "کنٹریکٹ",
        ),
    ),
    (
        # CPC Section 80: notice before suing the Government or a public
        # officer. Phrase-level triggers only -- a bare "सरकार" (government)
        # appears in questions about every other subject.
        _GOVERNMENT_SUIT_EXPANSION,
        (
            "सरकार पर मुकदमा", "सरकार के खिलाफ मुकदमा", "सरकार के विरुद्ध मुकदमा",
            "सरकार पर केस", "सरकार के खिलाफ केस", "सरकार के विरुद्ध वाद", "सरकार पर वाद",
            "सरकारी विभाग पर मुकदमा", "सरकारी अधिकारी पर मुकदमा",
        ),
    ),
)

# Romanized (Hinglish and the romanized forms of the other languages) triggers.
# Kept separate because these ARE plain ASCII words, so they can and should be
# matched with real word boundaries -- "na" inside "Nagar", "fir" inside
# "confirm", "sazaa" inside a name. Mapped to the same English expansions as
# their native-script counterparts above, by index into `_CONCEPTS`.
_ROMANIZED_WORD_TERMS: dict[str, tuple[str, ...]] = {
    "first information report FIR police complaint cognizable offence registration BNSS Section 173": (
        "fir", "thana", "thane", "police complaint", "police shikayat", "shikayat darj",
    ),
    "bail bail application bailable offence bail bond release from custody BNSS": (
        "jamanat", "zamanat", "jamin", "jaamin",
    ),
    "anticipatory bail pre-arrest bail apprehending arrest BNSS Section 482": (
        "agrim jamanat", "agrim zamanat", "anticipatory jamanat",
    ),
    "cheating dishonestly inducing delivery of property fraud Bharatiya Nyaya Sanhita Section 318": (
        "dhokha", "dhokhadhadi", "dhokadhadi", "thagi", "thug liya", "chhal", "fasavnuk",
    ),
    "punishment imprisonment fine sentence prescribed penalty": (
        "saza", "sazaa", "sajaa", "dand",
    ),
    "cyber crime online fraud UPI unauthorised electronic transaction Information Technology Act": (
        "cyber", "online fraud", "upi fraud", "otp fraud", "paisa kat gaya", "paise kat gaye",
    ),
    "consumer complaint Consumer Protection Act deficiency in service unfair trade practice refund": (
        "upbhokta", "grahak", "refund nahi",
    ),
    "divorce dissolution of marriage judicial separation marriage act": (
        "talak", "talaq", "ghatsphot",
    ),
    "restitution of conjugal rights Hindu Marriage Act Section 9 living separately without reasonable excuse": (
        "alag reh rahi", "alag reh raha", "vaivahik adhikar", "vivahik adhikar",
    ),
    "arbitration mediation conciliation alternative dispute resolution Arbitration and Conciliation Act "
    "1996 Mediation Act 2023 arbitral tribunal arbitral award": (
        "madhyasthata", "panchat", "sulah", "adr",
    ),
    "maintenance of wife children and parents monthly allowance BNSS Section 144": (
        "bharan poshan", "guzara bhatta", "gujara bhatta", "potgi",
    ),
    "security deposit tenant landlord rent agreement refund of deposit property law": (
        "kiraya", "kirayedar", "makan malik", "bhade", "bhadekaru",
    ),
    "cheque bounce dishonour of cheque Negotiable Instruments Act Section 138": (
        "cheque bounce", "check bounce", "chek baunce",
    ),
    "arrest without warrant cognizable offence grounds of arrest rights of arrested person BNSS Section 35": (
        "giraftar", "giraftari", "warrant", "hirasat", "atak",
    ),
    "criminal intimidation threat to cause injury stalking harassment Bharatiya Nyaya Sanhita": (
        "dhamki", "dhamka", "pareshan", "harass", "peecha kar", "blackmail",
    ),
    "domestic violence protection of women cruelty by husband or relatives dowry": (
        "gharelu hinsa", "dahej", "hunda", "crurta",
    ),
    "theft dishonestly taking movable property Bharatiya Nyaya Sanhita Section 303": (
        "chori", "chori ho gayi", "chori hui",
    ),
    "Right to Information Act public authority public information officer first appeal": (
        "rti", "suchna ka adhikar",
    ),
    "unpaid salary wages employment dues labour law appointment letter": (
        "vetan", "tankhwah", "tankhwah", "pagar", "salary nahi",
    ),
    "property ownership dispute land transfer of property registration act possession": (
        "sampatti", "zameen", "jameen", "kabza", "malmatta",
    ),
    "legal notice formal demand notice advocate reply civil dispute": (
        "kanooni notice", "legal notice bhejna",
    ),
    _GOVERNMENT_SUIT_EXPANSION: (
        "sarkar par mukadma", "sarkar par mukadama", "sarkar ke khilaf case", "sarkar ke khilaf mukadma",
        "sarkar pe case", "government par case",
    ),
    "legal rights of a citizen fundamental rights remedies available": (
        "adhikar", "kanooni adhikar", "haq",
    ),
    "section of the Act provision statutory text": (
        "dhara", "kalam",
    ),
}

# Concepts that only exist as a PAIR of ideas. A word like "cheque" or "bank"
# is not a legal concept on its own -- a Gujarati speaker writing "મારો ચેક
# “Funds Insufficient” કારણસર પરત આવ્યો" ("my cheque came back for insufficient
# funds") never says "cheque bounce" as a fixed phrase, so no contiguous term
# can be listed for it, and matching the bare noun would fire on every
# cheque-related sentence. The concept is recognised only when a word from
# EACH group is present. Group terms may be native-script (substring match, see
# `_contains_native_term`) or ASCII (whole-word match).
#
# The expansions are the same strings the single-term concepts above use
# wherever the concept is the same one, so a paired hit and a single-term hit
# de-duplicate rather than adding two copies of one search variant, and
# `_CONCEPT_INTENT` needs no second entry.
_CHEQUE_EXPANSION = "cheque bounce dishonour of cheque Negotiable Instruments Act Section 138"
_UNAUTHORISED_TRANSACTION_EXPANSION = (
    "unauthorised electronic banking transaction customer liability RBI zero liability "
    "report to bank within three working days UPI online fraud"
)
_POLICE_NOTICE_EXPANSION = (
    "police notice to appear notice of appearance BNSS Section 35 summons to attend police "
    "station arrest without warrant"
)

_PAIRED_CONCEPTS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        _CHEQUE_EXPANSION,
        # the instrument
        (
            "चेक", "ચેક", "ಚೆಕ್", "ചെക്ക്", "ਚੈੱਕ", "ਚੈਕ", "ଚେକ", "চেক", "چیک", "காசோலை", "చెక్కు",
            # QA retest 2026-09-24 (Sindhi/Telugu grounding investigation):
            # "چيڪ" is the Sindhi spelling of the loanword "cheque" -- it
            # ends in "ڪ" (the Sindhi-specific swash kaf, one of
            # `app/language/detector.py`'s `_SINDHI_ONLY_LETTERS`), not the
            # Urdu "ک" that "چیک" above already covers, so it is a genuinely
            # different Unicode string, not a duplicate. "చెక్" is the
            # commonly-typed short Telugu transliteration ("chek"), distinct
            # from "చెక్కు" ("checku") already listed -- both are real,
            # confirmed live: T020 test text "చెక్ బౌన్స్ అయితే..." matched
            # neither the main concept's own term list nor this paired
            # instrument group under the old spellings alone, so no
            # cheque-bounce expansion fired at all for either language.
            "چيڪ", "చెక్",
            "cheque",
        ),
        # ...coming back unpaid
        (
            "बाउंस", "बाउन्स", "वापस", "लौट", "रिटर्न", "अनादर", "खारिज", "डिसऑनर", "अपर्याप्त",
            "परत", "बाऊन्स",
            "બાઉન્સ", "પરત", "રિટર્ન", "અનાદર", "અપૂરતા",
            "ಬೌನ್ಸ್", "ವಾಪಸ್", "ಹಿಂತಿರುಗ", "ಅಮಾನ್ಯ",
            "ബൗൺസ്", "മടങ്ങി", "തിരിച്ചയ", "മടക്കി",
            "பவுன்ஸ்", "திரும்ப", "நிராகரி",
            "బౌన్స్", "తిరిగి", "తిరస్కర",
            "বাউন্স", "ফেরত", "প্রত্যাখ্যাত",
            "ਬਾਊਂਸ", "ਵਾਪਸ", "ਬਾਉਂਸ",
            "ବାଉନ୍ସ", "ଫେରି", "ଫେରସ୍ତ",
            "باؤنس", "واپس", "ڈس آنر",
            "return", "returned", "bounce", "bounced", "dishonour", "dishonoured", "dishonor",
            "dishonored", "insufficient", "unpaid",
        ),
    ),
    (
        _UNAUTHORISED_TRANSACTION_EXPANSION,
        # the bank / money movement
        (
            "बैंक", "खाते", "खाता", "लेनदेन", "लेन-देन", "ट्रांजैक्शन", "ट्रांज़ैक्शन", "एटीएम", "यूपीआई",
            "बँक", "खात्यातून", "व्यवहार",
            "બેંક", "ખાતા", "વ્યવહાર", "એટીએમ",
            "ಬ್ಯಾಂಕ್", "ಖಾತೆ", "ವಹಿವಾಟು",
            "ബാങ്ക്", "അക്കൗണ്ട്", "ഇടപാട്", "എടിഎം",
            "வங்கி", "கணக்கு", "பரிவர்த்தனை",
            "బ్యాంక్", "ఖాతా", "లావాదేవీ",
            "ব্যাংক", "অ্যাকাউন্ট", "লেনদেন",
            "ਬੈਂਕ", "ਖਾਤੇ", "ਲੈਣ-ਦੇਣ", "ਲੈਣ ਦੇਣ",
            "ବ୍ୟାଙ୍କ", "ଖାତା", "କାରବାର",
            "بینک", "کھاتے", "لین دین", "اے ٹی ایم",
            "bank", "atm", "upi", "transaction",
        ),
        # ...that the account holder did not authorise
        (
            "बिना अनुमति", "बिना मेरी", "अनधिकृत", "अनाधिकृत", "बिना अनुमती", "मेरी जानकारी के बिना",
            "मैंने नहीं", "बिना सहमति", "बिना इजाज़त", "बिना इजाजत",
            "परवानगीशिवाय", "संमतीशिवाय", "माझ्या संमती", "माझ्या परवानगी",
            "મંજૂરી વિના", "પરવાનગી વિના", "અનધિકૃત", "મારી જાણ બહાર", "સંમતિ વિના",
            "ಅನುಮತಿ ಇಲ್ಲದೆ", "ಅನಧಿಕೃತ", "ನನ್ನ ಅನುಮತಿ", "ನನ್ನ ಒಪ್ಪಿಗೆ",
            "അനുമതി നൽകാത്ത", "അനുമതിയില്ലാതെ", "അനധികൃത", "അറിവില്ലാതെ",
            "அனுமதி இல்லாமல்", "அனுமதியின்றி", "அங்கீகரிக்கப்படாத", "தெரியாமல்",
            "అనుమతి లేకుండా", "అనధికార", "నా అనుమతి",
            "অনুমতি ছাড়া", "অননুমোদিত", "অনুমতি ছাড়াই",
            "ਬਿਨਾਂ ਇਜਾਜ਼ਤ", "ਅਣਅਧਿਕਾਰਤ", "ਬਿਨਾ ਇਜਾਜ਼ਤ",
            "ଅନୁମତି ବିନା", "ଅନଧିକୃତ", "ଅନୁମତି ନ",
            "اجازت کے بغیر", "غیر مجاز", "میری اجازت",
            "अनुमति बिना", "अनुमतिबिना",
            "unauthorised", "unauthorized", "without permission", "without my consent",
            "without my permission", "not authorised", "not authorized", "fraudulent",
        ),
    ),
    (
        _POLICE_NOTICE_EXPANSION,
        # the police
        (
            "पुलिस", "पोलीस", "પોલીસ", "ಪೊಲೀಸ್", "ಪೊಲೀಸರ", "പോലീസ്", "போலீஸ்", "காவல்", "పోలీసు", "పోలీస్",
            "পুলিশ", "ਪੁਲਿਸ", "ପୋଲିସ", "ପୁଲିସ", "پولیس", "प्रहरी", "আৰক্ষী",
            "police",
        ),
        # ...a notice / summons
        (
            "नोटिस", "नोटीस", "समन", "सम्मन",
            "નોટિસ", "નોટીસ", "ನೋಟಿಸ್", "നോട്ടീസ്", "நோட்டீஸ்", "నోటీసు", "নোটিশ",
            "ਨੋਟਿਸ", "ਸੰਮਨ", "ନୋଟିସ", "نوٹس", "سمن",
            "notice", "summons",
        ),
    ),
)

_WORD_CHAR_CATEGORIES = frozenset({"Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Nd"})


def _is_word_char(char: str) -> bool:
    return unicodedata.category(char) in _WORD_CHAR_CATEGORIES


def _normalize_for_match(text: str) -> str:
    """Canonical form both the query and every term are compared in.

    Three real-user spellings defeated plain substring matching:

    * Zero-width joiner/non-joiner (U+200C/U+200D). Kannada, Malayalam, Bengali
      and Persian-script keyboards insert one inside abbreviations -- "ಎಫ್‌ಐಆರ್"
      typed as the citation form "ಎಫ್ಐಆರ್" plus an invisible ZWNJ is a different
      string to `str.find`, which is exactly how a Kannada FIR question missed
      the FIR concept in qa-40q-multilingual-20260921.
    * Decomposed vs. precomposed nukta letters ("ज़" as one code point or two).
      NFC on both sides makes them agree, whichever a keyboard emitted.
    * Arabic-script vowel marks (harakat). Urdu is normally written without
      them, but Kashmiri text carries them on almost every word ("نوٹِس"), so
      the same word never matched its Urdu spelling.

    Applied to the terms at import time and to the query on every call, so the
    two sides can never disagree about it.
    """
    decomposed = unicodedata.normalize("NFC", text)
    return "".join(
        char for char in decomposed
        if char not in _ZERO_WIDTH_CHARS and not ("ً" <= char <= "ٟ" or char == "ٰ")
    ).lower()


_ZERO_WIDTH_CHARS = frozenset({"‌", "‍"})


def _contains_native_term(haystack: str, needle: str) -> bool:
    """Substring containment with a light guard: the match must not be glued to
    another word character on BOTH sides at once (which would mean it is buried
    inside a longer, unrelated word). Deliberately more permissive than a word
    boundary, because Indic scripts inflect by suffixing directly onto the stem
    -- "जमानत" legitimately appears as "जमानतें"/"जमानती", and requiring a clean
    boundary would miss every inflected form.
    """
    index = haystack.find(needle)
    if index == -1:
        return False
    before_ok = index == 0 or not _is_word_char(haystack[index - 1])
    return before_ok


def _group_matches(haystack: str, terms: tuple[str | re.Pattern[str], ...]) -> bool:
    """Whether any term in a `_PAIRED_CONCEPTS` group occurs in `haystack`.
    ASCII terms were pre-compiled to whole-word patterns (so "atm" cannot match
    inside "batman"); native-script terms use the same inflection-tolerant
    containment check as the single-term concepts."""
    return any(
        term.search(haystack) if isinstance(term, re.Pattern) else _contains_native_term(haystack, term)
        for term in terms
    )


def _compile_group(terms: tuple[str, ...]) -> tuple[str | re.Pattern[str], ...]:
    compiled: list[str | re.Pattern[str]] = []
    for term in terms:
        normalized = _normalize_for_match(term)
        if normalized.isascii():
            compiled.append(re.compile(r"\b" + re.escape(normalized) + r"\b"))
        else:
            compiled.append(normalized)
    return tuple(compiled)


# Compiled/normalized once at import, so the per-request cost is only the
# query's own normalization plus the containment checks.
_NORMALIZED_CONCEPTS: tuple[tuple[str, tuple[str, ...]], ...] = tuple(
    (expansion, tuple(_normalize_for_match(term) for term in terms)) for expansion, terms in _CONCEPTS
)
_NORMALIZED_PAIRED_CONCEPTS: tuple[
    tuple[str, tuple[str | re.Pattern[str], ...], tuple[str | re.Pattern[str], ...]], ...
] = tuple(
    (expansion, _compile_group(first), _compile_group(second))
    for expansion, first, second in _PAIRED_CONCEPTS
)

# Compiled once: one alternation per concept over its romanized triggers.
_ROMANIZED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b", re.IGNORECASE), expansion)
    for expansion, terms in _ROMANIZED_WORD_TERMS.items()
)


def legal_english_variants(query: str) -> list[str]:
    """English legal search variants for the concepts `query` mentions.

    Returns `[]` when the query names no recognised legal concept, so callers
    can treat "nothing to add" and "this module isn't relevant" identically.
    Order follows `_CONCEPTS`, so the same query always produces the same
    variants (retrieval results stay reproducible, and the retrieval cache key
    stays stable).
    """
    if not query:
        return []
    lowered = _normalize_for_match(query)
    found: list[str] = []
    for expansion, terms in _NORMALIZED_CONCEPTS:
        if any(_contains_native_term(lowered, term) for term in terms):
            found.append(expansion)
    for expansion, first_group, second_group in _NORMALIZED_PAIRED_CONCEPTS:
        if expansion not in found and _group_matches(lowered, first_group) and _group_matches(lowered, second_group):
            found.append(expansion)
    for pattern, expansion in _ROMANIZED_PATTERNS:
        if expansion not in found and pattern.search(lowered):
            found.append(expansion)
    return found


# Phase 1 item 5: the `IntentDetector` rule each concept corresponds to, so a
# non-English question reaches the same intent (and therefore the same
# legal-category hint for retrieval and the reranker) as its English
# equivalent.
#
# Keyed by the expansion string, which is already this module's unique concept
# key. Deliberately an explicit mapping rather than string-matching the
# English expansion against the rule keywords: the expansions are written in
# statutory phrasing, which is full of generic legal nouns, and matching on
# those mis-routed real questions -- "मेरी बाइक चोरी हो गई" ("my bike was
# stolen") landed in Property Registration because the theft expansion
# contains the phrase "movable property", and the cheating expansion sent
# fraud questions there too via "delivery of property".
_CONCEPT_INTENT: dict[str, tuple[str, str]] = {
    "first information report FIR police complaint cognizable offence registration BNSS Section 173":
        ("FIR", "Criminal Law"),
    "bail bail application bailable offence bail bond release from custody BNSS":
        ("Bail", "Criminal Law"),
    "anticipatory bail pre-arrest bail apprehending arrest BNSS Section 482":
        ("Bail", "Criminal Law"),
    "cheating dishonestly inducing delivery of property fraud Bharatiya Nyaya Sanhita Section 318":
        ("Cheating and Fraud", "Criminal Law"),
    "cyber crime online fraud UPI unauthorised electronic transaction Information Technology Act":
        ("Cyber Crime", "Cyber Law"),
    "consumer complaint Consumer Protection Act deficiency in service unfair trade practice refund":
        ("Consumer Complaint", "Consumer Law"),
    "divorce dissolution of marriage judicial separation marriage act":
        ("Divorce", "Family Law"),
    "restitution of conjugal rights Hindu Marriage Act Section 9 living separately without reasonable excuse":
        ("Divorce", "Family Law"),
    "arbitration mediation conciliation alternative dispute resolution Arbitration and Conciliation Act "
    "1996 Mediation Act 2023 arbitral tribunal arbitral award":
        ("Alternative Dispute Resolution", "Civil Law"),
    "maintenance of wife children and parents monthly allowance BNSS Section 144":
        ("Divorce", "Family Law"),
    "security deposit tenant landlord rent agreement refund of deposit property law":
        ("Rental Dispute", "Property Law"),
    "cheque bounce dishonour of cheque Negotiable Instruments Act Section 138":
        ("Cheque Bounce", "Banking and Criminal Law"),
    "arrest without warrant cognizable offence grounds of arrest rights of arrested person BNSS Section 35":
        ("Arrest and Custody", "Criminal Law"),
    "criminal intimidation threat to cause injury stalking harassment Bharatiya Nyaya Sanhita":
        ("Threat and Harassment", "Criminal Law"),
    "domestic violence protection of women cruelty by husband or relatives dowry":
        ("Domestic Violence", "Family Law"),
    "theft dishonestly taking movable property Bharatiya Nyaya Sanhita Section 303":
        ("Theft", "Criminal Law"),
    "Right to Information Act public authority public information officer first appeal":
        ("RTI", "Public Law"),
    "unpaid salary wages employment dues labour law appointment letter":
        ("Salary Issue", "Labour Law"),
    "property ownership dispute land transfer of property registration act possession":
        ("Property Registration", "Property Law"),
    "legal notice formal demand notice advocate reply civil dispute":
        ("Legal Notice", "Civil Law"),
    _GOVERNMENT_SUIT_EXPANSION:
        ("Legal Notice", "Civil Law"),
    _UNAUTHORISED_TRANSACTION_EXPANSION:
        ("Cyber Crime", "Cyber Law"),
    _POLICE_NOTICE_EXPANSION:
        ("Arrest and Custody", "Criminal Law"),
    "sexual harassment of women at workplace internal committee POSH inquiry":
        ("POSH Complaint", "Employment Law"),
}


def legal_intent_hint(query: str) -> tuple[str, str] | None:
    """`(intent, legal_category)` for the most specific legal concept `query`
    names, or `None`.

    Used only as a FALLBACK by `IntentDetector`, after its own English keyword
    rules have failed -- an English question keeps resolving exactly as it
    always did. Concepts that carry no distinctive intent of their own
    ("punishment", "section", "rights", "procedure") are absent from the map on
    purpose: they say what KIND of question it is, not what it is about, and
    guessing an intent from them would be worse than the honest "General Legal
    Query" default.
    """
    for expansion in legal_english_variants(query):
        hint = _CONCEPT_INTENT.get(expansion)
        if hint:
            return hint
    return None


def has_legal_concept(query: str) -> bool:
    """Whether `query` names any legal concept this bridge recognises -- used
    to decide whether a cross-lingual retrieval allowance is justified, rather
    than granting it to any non-English text at all."""
    return bool(legal_english_variants(query))
