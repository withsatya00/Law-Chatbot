"""Part 52 "Multilingual Draft Engine": display-only translations for the
fixed, closed set of section headings used across every draft template
(`app.drafting.templates.base`).

Deliberately NOT used to change the `sections` dict's own keys -- those stay
the literal English strings everywhere else (LLM prompt/response parsing in
`LegalDraftEngine._parse_sections`, in-chat field edits like "change the
police station", PDF/DOCX/TXT export section iteration) because that is the
exact parsing contract the LLM is instructed to follow (see
`legal_drafting_prompt.md` rule 5, and Part 42's rationale for pinning a
fixed heading skeleton per category). Translating the dict keys themselves
would mean re-deriving that contract per language and risking the LLM
silently failing to follow an unfamiliar non-English heading format.

`translated_heading()` is applied only at the point a heading is actually
shown to a person -- chat preview text and every export format -- via a
small static lookup rather than a further LLM call, since the heading set is
small, fixed, and legally standard vocabulary (mistranslating a document's
own structural labels is worse than a mechanical, reviewed lookup table).
"""

_HEADING_TRANSLATIONS: dict[str, dict[str, str]] = {
    # Part 52 (Workflow Localization) fix: translating this heading to
    # "sewa mein" (Devanagari) collided with Hindi formal-letter convention,
    # where an addressee block's BODY conventionally opens with that same
    # phrase as its own first line -- an LLM writing genuine Hindi content
    # for this section naturally starts with it too, producing a visible
    # duplicate heading independent of the separate authority_label
    # duplication fixed in fallback_phrases.py. "prati" ("to"/"addressed
    # to") is a distinct label word used for exactly this purpose in Hindi
    # bureaucratic/legal documents (e.g. RTI forms), so it does not collide
    # with the body's own opening line.
    "To": {
        "hindi": "प्रति", "tamil": "பெறுநர்", "telugu": "గ్రహీత",
        "kannada": "ಸ್ವೀಕರಿಸುವವರು", "bengali": "প্রতি",
        "malayalam": "സ്വീകർത്താവ്", "marathi": "प्रति", "gujarati": "પ્રતિ",
        "punjabi": "ਪ੍ਰਤੀ", "odia": "ପ୍ରତି", "urdu": "بخدمت",
    },
    # Part 56 "Advocate-Style Draft Redesign": "Recipient" is the unified
    # Notice/Complaint skeleton's renamed "To" heading -- same translations,
    # different English key, since Application (RTI) still uses "To" itself.
    "Recipient": {
        "hindi": "प्रति", "tamil": "பெறுநர்", "telugu": "గ్రహీత",
        "kannada": "ಸ್ವೀಕರಿಸುವವರು", "bengali": "প্রতি",
        "malayalam": "സ്വീകർത്താവ്", "marathi": "प्रति", "gujarati": "પ્રતિ",
        "punjabi": "ਪ੍ਰਤੀ", "odia": "ପ୍ରତି", "urdu": "بخدمت",
    },
    "Introduction": {
        "hindi": "परिचय", "tamil": "அறிமுகம்", "telugu": "పరిచయం",
        "kannada": "ಪರಿಚಯ", "bengali": "ভূমিকা",
        "malayalam": "ആമുഖം", "marathi": "प्रस्तावना", "gujarati": "પરિચય",
        "punjabi": "ਜਾਣ-ਪਛਾਣ", "odia": "ପରିଚୟ", "urdu": "تعارف",
    },
    "Facts of the Case": {
        "hindi": "मामले के तथ्य", "tamil": "வழக்கின் உண்மைகள்", "telugu": "కేసు వాస్తవాలు",
        "kannada": "ಪ್ರಕರಣದ ಸತ್ಯಾಂಶಗಳು", "bengali": "মামলার ঘটনা",
        "malayalam": "കേസിന്റെ വസ്തുതകൾ", "marathi": "प्रकरणातील तथ्ये", "gujarati": "કેસની હકીકતો",
        "punjabi": "ਕੇਸ ਦੇ ਤੱਥ", "odia": "ମାମଲାର ତଥ୍ୟ", "urdu": "مقدمے کے حقائق",
    },
    "Legal Position": {
        "hindi": "कानूनी स्थिति", "tamil": "சட்ட நிலை", "telugu": "చట్టపరమైన స్థితి",
        "kannada": "ಕಾನೂನು ಸ್ಥಿತಿ", "bengali": "আইনি অবস্থান",
        "malayalam": "നിയമപരമായ നിലപാട്", "marathi": "कायदेशीर स्थिती", "gujarati": "કાનૂની સ્થિતિ",
        "punjabi": "ਕਾਨੂੰਨੀ ਸਥਿਤੀ", "odia": "ଆଇନଗତ ସ୍ଥିତି", "urdu": "قانونی حیثیت",
    },
    "Consequences": {
        "hindi": "परिणाम", "tamil": "விளைவுகள்", "telugu": "పరిణామాలు",
        "kannada": "ಪರಿಣಾಮಗಳು", "bengali": "পরিণতি",
        "malayalam": "അനന്തരഫലങ്ങൾ", "marathi": "परिणाम", "gujarati": "પરિણામો",
        "punjabi": "ਨਤੀਜੇ", "odia": "ପରିଣାମ", "urdu": "نتائج",
    },
    "Prayer": {
        "hindi": "प्रार्थना", "tamil": "வேண்டுகோள்", "telugu": "ప్రార్థన",
        "kannada": "ಪ್ರಾರ್ಥನೆ", "bengali": "প্রার্থনা",
        "malayalam": "അപേക്ഷ", "marathi": "विनंती", "gujarati": "પ્રાર્થના",
        "punjabi": "ਬੇਨਤੀ", "odia": "ପ୍ରାର୍ଥନା", "urdu": "استدعا",
    },
    # "Signature Block" is the unified skeleton's renamed "Signature" --
    # same translations, different English key (Affidavit/Application still
    # use "Signature" itself).
    "Signature Block": {
        "hindi": "हस्ताक्षर", "tamil": "கையொப்பம்", "telugu": "సంతకం",
        "kannada": "ಸಹಿ", "bengali": "স্বাক্ষর",
        "malayalam": "ഒപ്പ്", "marathi": "स्वाक्षरी", "gujarati": "સહી",
        "punjabi": "ਦਸਤਖਤ", "odia": "ହସ୍ତାକ୍ଷର", "urdu": "دستخط",
    },
    "Subject": {
        "hindi": "विषय", "tamil": "பொருள்", "telugu": "విషయం",
        "kannada": "ವಿಷಯ", "bengali": "বিষয়",
        "malayalam": "വിഷയം", "marathi": "विषय", "gujarati": "વિષય",
        "punjabi": "ਵਿਸ਼ਾ", "odia": "ବିଷୟ", "urdu": "موضوع",
    },
    "Notice": {
        "hindi": "सूचना", "tamil": "அறிவிப்பு", "telugu": "నోటీసు",
        "kannada": "ಸೂಚನೆ", "bengali": "নোটিশ",
        "malayalam": "അറിയിപ്പ്", "marathi": "सूचना", "gujarati": "સૂચના",
        "punjabi": "ਨੋਟਿਸ", "odia": "ବିଜ୍ଞପ୍ତି", "urdu": "نوٹس",
    },
    "Sender Details": {
        "hindi": "प्रेषक का विवरण", "tamil": "அனுப்புநர் விவரங்கள்", "telugu": "పంపినవారి వివరాలు",
        "kannada": "ಕಳುಹಿಸಿದವರ ವಿವರಗಳು", "bengali": "প্রেরকের বিবরণ",
        "malayalam": "അയച്ചയാളുടെ വിവരങ്ങൾ", "marathi": "प्रेषकाचा तपशील", "gujarati": "મોકલનારની વિગતો",
        "punjabi": "ਭੇਜਣ ਵਾਲੇ ਦਾ ਵੇਰਵਾ", "odia": "ପ୍ରେରକଙ୍କ ବିବରଣୀ", "urdu": "بھیجنے والے کی تفصیلات",
    },
    "Complainant Details": {
        "hindi": "परिवादी का विवरण", "tamil": "புகார்தாரர் விவரங்கள்", "telugu": "ఫిర్యాదుదారు వివరాలు",
        "kannada": "ದೂರುದಾರರ ವಿವರಗಳು", "bengali": "অভিযোগকারীর বিবরণ",
        "malayalam": "പരാതിക്കാരന്റെ വിവരങ്ങൾ", "marathi": "तक्रारदाराचा तपशील", "gujarati": "ફરિયાદીની વિગતો",
        "punjabi": "ਸ਼ਿਕਾਇਤਕਰਤਾ ਦਾ ਵੇਰਵਾ", "odia": "ଅଭିଯୋଗକାରୀଙ୍କ ବିବରଣୀ", "urdu": "شکایت کنندہ کی تفصیلات",
    },
    "Facts": {
        "hindi": "तथ्य", "tamil": "உண்மைகள்", "telugu": "వాస్తవాలు",
        "kannada": "ಸತ್ಯಾಂಶಗಳು", "bengali": "ঘটনা",
        "malayalam": "വസ്തുതകൾ", "marathi": "तथ्ये", "gujarati": "હકીકતો",
        "punjabi": "ਤੱਥ", "odia": "ତଥ୍ୟ", "urdu": "حقائق",
    },
    "Request": {
        "hindi": "अनुरोध", "tamil": "கோரிக்கை", "telugu": "అభ్యర్థన",
        "kannada": "ವಿನಂತಿ", "bengali": "অনুরোধ",
        "malayalam": "അഭ്യർത്ഥന", "marathi": "विनंती", "gujarati": "વિનંતી",
        "punjabi": "ਬੇਨਤੀ", "odia": "ଅନୁରୋଧ", "urdu": "درخواست",
    },
    "Court / Authority Name": {
        "hindi": "न्यायालय / प्राधिकरण का नाम", "tamil": "நீதிமன்றம் / அதிகார அமைப்பு பெயர்",
        "telugu": "కోర్టు / అధికార సంస్థ పేరు", "kannada": "ನ್ಯಾಯಾಲಯ / ಪ್ರಾಧಿಕಾರದ ಹೆಸರು",
        "bengali": "আদালত / কর্তৃপক্ষের নাম",
        "malayalam": "കോടതി / അധികാരസ്ഥാപനത്തിന്റെ പേര്", "marathi": "न्यायालय / प्राधिकरणाचे नाव",
        "gujarati": "અદાલત / સત્તામંડળનું નામ", "punjabi": "ਅਦਾਲਤ / ਅਥਾਰਟੀ ਦਾ ਨਾਮ",
        "odia": "ନ୍ୟାୟାଳୟ / କର୍ତ୍ତୃପକ୍ଷଙ୍କ ନାମ", "urdu": "عدالت / اتھارٹی کا نام",
    },
    "Deponent Details": {
        "hindi": "शपथकर्ता का विवरण", "tamil": "சத்தியக்கடதாசி அளிப்பவர் விவரங்கள்",
        "telugu": "డిపొనెంట్ వివరాలు", "kannada": "ಪ್ರಮಾಣಪತ್ರದಾರರ ವಿವರಗಳು", "bengali": "হলফনামাকারীর বিবরণ",
        "malayalam": "സത്യപ്രസ്താവന നൽകുന്നയാളുടെ വിവരങ്ങൾ", "marathi": "शपथकर्त्याचा तपशील",
        "gujarati": "સોગંદનામું આપનારની વિગતો", "punjabi": "ਹਲਫ਼ਨਾਮਾ ਦੇਣ ਵਾਲੇ ਦਾ ਵੇਰਵਾ",
        "odia": "ଶପଥକାରୀଙ୍କ ବିବରଣୀ", "urdu": "بیان حلفی دینے والے کی تفصیلات",
    },
    "Statements": {
        "hindi": "कथन", "tamil": "அறிக்கைகள்", "telugu": "ప్రకటనలు",
        "kannada": "ಹೇಳಿಕೆಗಳು", "bengali": "বিবৃতি",
        "malayalam": "പ്രസ്താവനകൾ", "marathi": "विधाने", "gujarati": "નિવેદનો",
        "punjabi": "ਬਿਆਨ", "odia": "ବିବୃତି", "urdu": "بیانات",
    },
    "Verification": {
        "hindi": "सत्यापन", "tamil": "சான்று", "telugu": "ధృవీకరణ",
        "kannada": "ಪರಿಶೀಲನೆ", "bengali": "যাচাইকরণ",
        "malayalam": "സാക്ഷ്യപ്പെടുത്തൽ", "marathi": "पडताळणी", "gujarati": "ચકાસણી",
        "punjabi": "ਪੁਸ਼ਟੀਕਰਨ", "odia": "ଯାଞ୍ଚ", "urdu": "تصدیق",
    },
    "Applicant Details": {
        "hindi": "आवेदक का विवरण", "tamil": "விண்ணப்பதாரர் விவரங்கள்", "telugu": "దరఖాస్తుదారు వివరాలు",
        "kannada": "ಅರ್ಜಿದಾರರ ವಿವರಗಳು", "bengali": "আবেদনকারীর বিবরণ",
        "malayalam": "അപേക്ഷകന്റെ വിവരങ്ങൾ", "marathi": "अर्जदाराचा तपशील", "gujarati": "અરજદારની વિગતો",
        "punjabi": "ਬਿਨੈਕਾਰ ਦਾ ਵੇਰਵਾ", "odia": "ଆବେଦନକାରୀଙ୍କ ବିବରଣୀ", "urdu": "درخواست گزار کی تفصیلات",
    },
    "Place": {
        "hindi": "स्थान", "tamil": "இடம்", "telugu": "స్థలం",
        "kannada": "ಸ್ಥಳ", "bengali": "স্থান",
        "malayalam": "സ്ഥലം", "marathi": "ठिकाण", "gujarati": "સ્થળ",
        "punjabi": "ਸਥਾਨ", "odia": "ସ୍ଥାନ", "urdu": "مقام",
    },
    "Date": {
        "hindi": "दिनांक", "tamil": "தேதி", "telugu": "తేదీ",
        "kannada": "ದಿನಾಂಕ", "bengali": "তারিখ",
        "malayalam": "തീയതി", "marathi": "दिनांक", "gujarati": "તારીખ",
        "punjabi": "ਮਿਤੀ", "odia": "ତାରିଖ", "urdu": "تاریخ",
    },
    "Signature": {
        "hindi": "हस्ताक्षर", "tamil": "கையொப்பம்", "telugu": "సంతకం",
        "kannada": "ಸಹಿ", "bengali": "স্বাক্ষর",
        "malayalam": "ഒപ്പ്", "marathi": "स्वाक्षरी", "gujarati": "સહી",
        "punjabi": "ਦਸਤਖਤ", "odia": "ହସ୍ତାକ୍ଷର", "urdu": "دستخط",
    },
    "Annexures": {
        "hindi": "संलग्नक", "tamil": "இணைப்புகள்", "telugu": "జతపరుపులు",
        "kannada": "ಲಗತ್ತುಗಳು", "bengali": "সংযুক্তি",
        "malayalam": "അനുബന്ധങ്ങൾ", "marathi": "जोडपत्रे", "gujarati": "જોડાણો",
        "punjabi": "ਅਨੁਲੱਗ", "odia": "ସଂଲଗ୍ନକ", "urdu": "منسلکات",
    },
    "Disclaimer": {
        "hindi": "अस्वीकरण", "tamil": "மறுப்பு", "telugu": "నిరాకరణ",
        "kannada": "ಹಕ್ಕುನಿರಾಕರಣೆ", "bengali": "দাবিত্যাগ",
        "malayalam": "നിരാകരണം", "marathi": "अस्वीकरण", "gujarati": "અસ્વીકરણ",
        "punjabi": "ਬੇਦਾਅਵਾ", "odia": "ଅସ୍ୱୀକରଣ", "urdu": "دستبرداری",
    },
    # Part 57 "Drafting Lifecycle Redesign": the new Contract-category
    # skeleton (`templates/base.py::CONTRACT_SECTIONS`).
    "Title": {
        "hindi": "शीर्षक", "tamil": "தலைப்பு", "telugu": "శీర్షిక",
        "kannada": "ಶೀರ್ಷಿಕೆ", "bengali": "শিরোনাম",
        "malayalam": "ശീർഷകം", "marathi": "शीर्षक", "gujarati": "શીર્ષક",
        "punjabi": "ਸਿਰਲੇਖ", "odia": "ଶୀର୍ଷକ", "urdu": "عنوان",
    },
    "Parties": {
        "hindi": "पक्षकार", "tamil": "தரப்பினர்", "telugu": "పక్షాలు",
        "kannada": "ಪಕ್ಷಗಳು", "bengali": "পক্ষগণ",
        "malayalam": "കക്ഷികൾ", "marathi": "पक्षकार", "gujarati": "પક્ષકારો",
        "punjabi": "ਧਿਰਾਂ", "odia": "ପକ୍ଷମାନେ", "urdu": "فریقین",
    },
    "Recitals": {
        "hindi": "प्रस्तावना", "tamil": "முன்னுரை", "telugu": "ప్రవేశిక",
        "kannada": "ಪೀಠಿಕೆ", "bengali": "প্রস্তাবনা",
        "malayalam": "ആമുഖ പ്രസ്താവന", "marathi": "प्रस्तावना", "gujarati": "પ્રસ્તાવના",
        "punjabi": "ਪ੍ਰਸਤਾਵਨਾ", "odia": "ପ୍ରସ୍ତାବନା", "urdu": "تمہید",
    },
    "Terms and Conditions": {
        "hindi": "नियम एवं शर्तें", "tamil": "விதிமுறைகள் மற்றும் நிபந்தனைகள்", "telugu": "నిబంధనలు మరియు షరతులు",
        "kannada": "ನಿಯಮಗಳು ಮತ್ತು ಷರತ್ತುಗಳು", "bengali": "শর্তাবলী",
        "malayalam": "നിബന്ധനകളും വ്യവസ്ഥകളും", "marathi": "अटी व शर्ती", "gujarati": "નિયમો અને શરતો",
        "punjabi": "ਨਿਯਮ ਅਤੇ ਸ਼ਰਤਾਂ", "odia": "ସର୍ତ୍ତାବଳୀ", "urdu": "شرائط و ضوابط",
    },
    "Term and Termination": {
        "hindi": "अवधि एवं समाप्ति", "tamil": "காலம் மற்றும் முடிவு", "telugu": "కాలవ్యవధి మరియు ముగింపు",
        "kannada": "ಅವಧಿ ಮತ್ತು ಮುಕ್ತಾಯ", "bengali": "মেয়াদ ও সমাপ্তি",
        "malayalam": "കാലാവധിയും അവസാനിപ്പിക്കലും", "marathi": "कालावधी व समाप्ती", "gujarati": "મુદત અને સમાપ્તિ",
        "punjabi": "ਮਿਆਦ ਅਤੇ ਸਮਾਪਤੀ", "odia": "ମିଆଦ ଏବଂ ସମାପ୍ତି", "urdu": "مدت اور خاتمہ",
    },
    "Governing Law and Jurisdiction": {
        "hindi": "शासी कानून एवं क्षेत्राधिकार", "tamil": "நிர்வாகச் சட்டம் மற்றும் அதிகார வரம்பு",
        "telugu": "పాలక చట్టం మరియు అధికార పరిధి", "kannada": "ಆಡಳಿತಾತ್ಮಕ ಕಾನೂನು ಮತ್ತು ವ್ಯಾಪ್ತಿ",
        "bengali": "শাসক আইন ও এখতিয়ার",
        "malayalam": "ഭരണ നിയമവും അധികാരപരിധിയും", "marathi": "शासकीय कायदा व अधिकारक्षेत्र",
        "gujarati": "સંચાલક કાયદો અને અધિકારક્ષેત્ર", "punjabi": "ਸ਼ਾਸਕੀ ਕਾਨੂੰਨ ਅਤੇ ਅਧਿਕਾਰ ਖੇਤਰ",
        "odia": "ଶାସକ ଆଇନ ଏବଂ ଅଧିକାରକ୍ଷେତ୍ର", "urdu": "حاکم قانون اور دائرہ اختیار",
    },
    "Signatures": {
        "hindi": "हस्ताक्षर", "tamil": "கையொப்பங்கள்", "telugu": "సంతకాలు",
        "kannada": "ಸಹಿಗಳು", "bengali": "স্বাক্ষরসমূহ",
        "malayalam": "ഒപ്പുകൾ", "marathi": "स्वाक्षऱ्या", "gujarati": "સહીઓ",
        "punjabi": "ਦਸਤਖਤ", "odia": "ହସ୍ତାକ୍ଷରଗୁଡ଼ିକ", "urdu": "دستخطیں",
    },
}


def translated_heading(heading: str, language: str) -> str:
    """The display label for `heading` in `language`.

    Falls back to the original English heading when the language isn't
    covered by this table (e.g. "english"/"hinglish", where the heading is
    conventionally left in English anyway, or any language outside the fixed
    set above) -- never raises, since an untranslated-but-correct heading is
    always safer for a legal document than a blank or fabricated one.
    """
    return _HEADING_TRANSLATIONS.get(heading, {}).get(language, heading)
