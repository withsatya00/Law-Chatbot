"""Part 54 "Complete Multilingual Drafting Pass" item 1: Tamil/Telugu/
Kannada/Bengali translations for every drafting field's `label` (Hindi
already has its own `hindi_label` authored directly on each `DraftField` in
the template YAMLs).

Keyed by field `key` rather than duplicated per-template into each YAML,
since the same field key (`applicant_name`, `facts`, `place`, ...) repeats
across most of the 11 templates with an identical English label -- the same
"small, closed lookup table" pattern already used by `fallback_phrases.py`/
`heading_translations.py`/`title_translations.py`/`wrapper_messages.py`,
rather than growing ~400 near-duplicate lines across 11 YAML files.

A handful of keys carry a template-specific English label wording
("Police Station" vs. "Police Station (nearest, for jurisdiction)" vs.
"Concerned Police Station" -- all the same `police_station` key) --
`_FIELD_LABEL_OVERRIDES` covers those by `(draft_id, key)` instead of
forcing one generic translation onto a more specific English label.
"""

from app.drafting.templates.base import DraftField

_FIELD_LABEL_TRANSLATIONS: dict[str, dict[str, str]] = {
    "applicant_name": {
        "tamil": "விண்ணப்பதாரர் பெயர்", "telugu": "దరఖాస్తుదారు పేరు",
        "kannada": "ಅರ್ಜಿದಾರರ ಹೆಸರು", "bengali": "আবেদনকারীর নাম",
        "malayalam": "അപേക്ഷകന്റെ പേര്", "marathi": "अर्जदाराचे नाव", "gujarati": "અરજદારનું નામ", "punjabi": "ਬਿਨੈਕਾਰ ਦਾ ਨਾਮ", "odia": "ଆବେଦନକାରୀଙ୍କ ନାମ", "urdu": "درخواست گزار کا نام",
    },
    "applicant_address": {
        "tamil": "விண்ணப்பதாரர் முகவரி", "telugu": "దరఖాస్తుదారు చిరునామా",
        "kannada": "ಅರ್ಜಿದಾರರ ವಿಳಾಸ", "bengali": "আবেদনকারীর ঠিকানা",
        "malayalam": "അപേക്ഷകന്റെ വിലാസം", "marathi": "अर्जदाराचा पत्ता", "gujarati": "અરજદારનું સરનામું", "punjabi": "ਬਿਨੈਕਾਰ ਦਾ ਪਤਾ", "odia": "ଆବେଦନକାରୀଙ୍କ ଠିକଣା", "urdu": "درخواست گزار کا پتہ",
    },
    "applicant_mobile": {
        "tamil": "மொபைல் எண்", "telugu": "మొబైల్ నంబర్",
        "kannada": "ಮೊಬೈಲ್ ಸಂಖ್ಯೆ", "bengali": "মোবাইল নম্বর",
        "malayalam": "മൊബൈൽ നമ്പർ", "marathi": "मोबाईल क्रमांक", "gujarati": "મોબાઇલ નંબર", "punjabi": "ਮੋਬਾਈਲ ਨੰਬਰ", "odia": "ମୋବାଇଲ୍ ନମ୍ବର", "urdu": "موبائل نمبر",
    },
    "applicant_email": {
        "tamil": "மின்னஞ்சல்", "telugu": "ఇమెయిల్",
        "kannada": "ಇಮೇಲ್", "bengali": "ইমেইল",
        "malayalam": "ഇമെയിൽ", "marathi": "ईमेल", "gujarati": "ઇમેઇલ", "punjabi": "ਈਮੇਲ", "odia": "ଇମେଲ୍", "urdu": "ای میل",
    },
    "respondent_name": {
        "tamil": "பிரதிவாதி / எதிர்தரப்பினர் பெயர்", "telugu": "ప్రతివాది / ప్రత్యర్థి పేరు",
        "kannada": "ಪ್ರತಿವಾದಿ / ಎದುರಾಳಿ ಹೆಸರು", "bengali": "বিবাদী / বিপরীত পক্ষের নাম",
        "malayalam": "എതിർകക്ഷിയുടെ പേര്", "marathi": "प्रतिवादीचे नाव", "gujarati": "પ્રતિવાદીનું નામ", "punjabi": "ਪ੍ਰਤੀਵਾਦੀ ਦਾ ਨਾਮ", "odia": "ପ୍ରତିବାଦୀଙ୍କ ନାମ", "urdu": "مدعا علیہ کا نام",
    },
    "respondent_address": {
        "tamil": "பிரதிவாதி முகவரி", "telugu": "ప్రతివాది చిరునామా",
        "kannada": "ಪ್ರತಿವಾದಿ ವಿಳಾಸ", "bengali": "বিবাদীর ঠিকানা",
        "malayalam": "എതിർകക്ഷിയുടെ വിലാസം", "marathi": "प्रतिवादीचा पत्ता", "gujarati": "પ્રતિવાદીનું સરનામું", "punjabi": "ਪ੍ਰਤੀਵਾਦੀ ਦਾ ਪਤਾ", "odia": "ପ୍ରତିବାଦୀଙ୍କ ଠିକଣା", "urdu": "مدعا علیہ کا پتہ",
    },
    "facts": {
        "tamil": "வழக்கின் உண்மைகள் (உங்கள் சொந்த வார்த்தைகளில்)",
        "telugu": "కేసు వాస్తవాలు (మీ సొంత మాటల్లో)",
        "kannada": "ಪ್ರಕರಣದ ಸತ್ಯಾಂಶಗಳು (ನಿಮ್ಮ ಸ್ವಂತ ಮಾತುಗಳಲ್ಲಿ)",
        "bengali": "মামলার তথ্য (নিজের ভাষায়)",
        "malayalam": "കേസിന്റെ വസ്തുതകൾ (നിങ്ങളുടെ സ്വന്തം വാക്കുകളിൽ)", "marathi": "प्रकरणातील तथ्ये (आपल्या स्वतःच्या शब्दांत)", "gujarati": "કેસની હકીકતો (તમારા પોતાના શબ્દોમાં)", "punjabi": "ਕੇਸ ਦੇ ਤੱਥ (ਆਪਣੇ ਸ਼ਬਦਾਂ ਵਿੱਚ)", "odia": "ମାମଲାର ତଥ୍ୟ (ନିଜ ଭାଷାରେ)", "urdu": "مقدمے کے حقائق (اپنے الفاظ میں)",
    },
    "expected_relief": {
        "tamil": "எதிர்பார்க்கும் நிவாரணம் / நீங்கள் விரும்புவது",
        "telugu": "ఆశించే ఉపశమనం / మీకు కావలసినది",
        "kannada": "ನಿರೀಕ್ಷಿತ ಪರಿಹಾರ / ನಿಮಗೆ ಬೇಕಾದುದು",
        "bengali": "প্রত্যাশিত প্রতিকার / আপনি যা চান",
        "malayalam": "പ്രതീക്ഷിക്കുന്ന പരിഹാരം / നിങ്ങൾക്ക് വേണ്ടത്", "marathi": "अपेक्षित दिलासा / तुम्हाला काय हवे आहे", "gujarati": "અપેક્ષિત રાહત / તમારે શું જોઈએ છે", "punjabi": "ਲੋੜੀਂਦੀ ਰਾਹਤ / ਤੁਹਾਨੂੰ ਕੀ ਚਾਹੀਦਾ ਹੈ", "odia": "ପ୍ରତ୍ୟାଶିତ ପ୍ରତିକାର / ଆପଣ କଣ ଚାହାଁନ୍ତି", "urdu": "متوقع ریلیف / آپ کیا چاہتے ہیں",
    },
    "place": {
        "tamil": "இடம்", "telugu": "స్థలం", "kannada": "ಸ್ಥಳ", "bengali": "স্থান",
        "malayalam": "സ്ഥലം", "marathi": "ठिकाण", "gujarati": "સ્થળ", "punjabi": "ਸਥਾਨ", "odia": "ସ୍ଥାନ", "urdu": "مقام",
    },
    "available_documents": {
        "tamil": "கிடைக்கும் ஆவணங்கள் / ஆதாரங்கள்", "telugu": "అందుబాటులో ఉన్న పత్రాలు / ఆధారాలు",
        "kannada": "ಲಭ್ಯವಿರುವ ದಾಖಲೆಗಳು / ಪುರಾವೆಗಳು", "bengali": "উপলব্ধ নথি / প্রমাণ",
        "malayalam": "ലഭ്യമായ രേഖകൾ / തെളിവുകൾ", "marathi": "उपलब्ध कागदपत्रे / पुरावे", "gujarati": "ઉપલબ્ધ દસ્તાવેજો / પુરાવા", "punjabi": "ਉਪਲਬਧ ਦਸਤਾਵੇਜ਼ / ਸਬੂਤ", "odia": "ଉପଲବ୍ଧ ଦସ୍ତାବିଜ / ପ୍ରମାଣ", "urdu": "دستیاب دستاویزات / شواہد",
    },
    "witnesses": {
        "tamil": "சாட்சிகள் (இருந்தால்)", "telugu": "సాక్షులు (ఉంటే)",
        "kannada": "ಸಾಕ್ಷಿಗಳು (ಇದ್ದರೆ)", "bengali": "সাক্ষী (যদি থাকে)",
        "malayalam": "സാക്ഷികൾ (ഉണ്ടെങ്കിൽ)", "marathi": "साक्षीदार (असल्यास)", "gujarati": "સાક્ષીઓ (જો હોય તો)", "punjabi": "ਗਵਾਹ (ਜੇ ਕੋਈ ਹੋਵੇ)", "odia": "ସାକ୍ଷୀ (ଥିଲେ)", "urdu": "گواہ (اگر کوئی ہو)",
    },
    "incident_location": {
        "tamil": "சம்பவ இடம்", "telugu": "సంఘటన స్థలం",
        "kannada": "ಘಟನೆಯ ಸ್ಥಳ", "bengali": "ঘটনার স্থান",
        "malayalam": "സംഭവ സ്ഥലം", "marathi": "घटनास्थळ", "gujarati": "ઘટનાનું સ્થળ", "punjabi": "ਘਟਨਾ ਦਾ ਸਥਾਨ", "odia": "ଘଟଣା ସ୍ଥଳ", "urdu": "واقعے کی جگہ",
    },
    "incident_date": {
        "tamil": "சம்பவ தேதி", "telugu": "సంఘటన తేదీ",
        "kannada": "ಘಟನೆಯ ದಿನಾಂಕ", "bengali": "ঘটনার তারিখ",
        "malayalam": "സംഭവ തീയതി", "marathi": "घटनेची तारीख", "gujarati": "ઘટનાની તારીખ", "punjabi": "ਘਟਨਾ ਦੀ ਮਿਤੀ", "odia": "ଘଟଣା ତାରିଖ", "urdu": "واقعے کی تاریخ",
    },
    "incident_time": {
        "tamil": "சம்பவ நேரம்", "telugu": "సంఘటన సమయం",
        "kannada": "ಘಟನೆಯ ಸಮಯ", "bengali": "ঘটনার সময়",
        "malayalam": "സംഭവ സമയം", "marathi": "घटनेची वेळ", "gujarati": "ઘટનાનો સમય", "punjabi": "ਘਟਨਾ ਦਾ ਸਮਾਂ", "odia": "ଘଟଣା ସମୟ", "urdu": "واقعے کا وقت",
    },
    "accused_details": {
        "tamil": "குற்றம் சாட்டப்பட்டவர் / எதிர் தரப்பினர் விவரங்கள் (தெரிந்தால்)",
        "telugu": "నిందితుడు / ప్రత్యర్థి వివరాలు (తెలిస్తే)",
        "kannada": "ಆರೋಪಿ / ಪ್ರತಿವಾದಿಯ ವಿವರಗಳು (ತಿಳಿದಿದ್ದರೆ)",
        "bengali": "অভিযুক্ত / বিপরীত পক্ষের বিবরণ (জানা থাকলে)",
        "malayalam": "പ്രതി / എതിർകക്ഷിയുടെ വിവരങ്ങൾ (അറിയാമെങ്കിൽ)", "marathi": "आरोपी / प्रतिपक्षाचा तपशील (माहीत असल्यास)", "gujarati": "આરોપી / પ્રતિપક્ષની વિગતો (જાણ હોય તો)", "punjabi": "ਦੋਸ਼ੀ / ਵਿਰੋਧੀ ਧਿਰ ਦਾ ਵੇਰਵਾ (ਜੇ ਪਤਾ ਹੋਵੇ)", "odia": "ଅଭିଯୁକ୍ତ / ବିପକ୍ଷଙ୍କ ବିବରଣୀ (ଜଣାଥିଲେ)", "urdu": "ملزم / مخالف فریق کی تفصیلات (اگر معلوم ہو)",
    },
    "police_station": {
        "tamil": "காவல் நிலையம்", "telugu": "పోలీసు స్టేషన్",
        "kannada": "ಪೊಲೀಸ್ ಠಾಣೆ", "bengali": "থানা",
        "malayalam": "പോലീസ് സ്റ്റേഷൻ", "marathi": "पोलीस स्टेशन", "gujarati": "પોલીસ સ્ટેશન", "punjabi": "ਪੁਲਿਸ ਸਟੇਸ਼ਨ", "odia": "ପୋଲିସ୍ ଷ୍ଟେସନ୍", "urdu": "پولیس اسٹیشن",
    },
    "public_authority_name": {
        "tamil": "பொது அதிகார அமைப்பு / துறையின் பெயர்", "telugu": "ప్రజా అధికార సంస్థ / శాఖ పేరు",
        "kannada": "ಸಾರ್ವಜನಿಕ ಪ್ರಾಧಿಕಾರ / ಇಲಾಖೆಯ ಹೆಸರು", "bengali": "সরকারি কর্তৃপক্ষ / বিভাগের নাম",
        "malayalam": "പൊതു അധികാരസ്ഥാപനം / വകുപ്പിന്റെ പേര്", "marathi": "सार्वजनिक प्राधिकरण / विभागाचे नाव", "gujarati": "જાહેર સત્તામંડળ / વિભાગનું નામ", "punjabi": "ਜਨਤਕ ਅਥਾਰਟੀ / ਵਿਭਾਗ ਦਾ ਨਾਮ", "odia": "ସର୍ବସାଧାରଣ କର୍ତ୍ତୃପକ୍ଷ / ବିଭାଗର ନାମ", "urdu": "عوامی ادارے / محکمے کا نام",
    },
    "information_sought": {
        "tamil": "கோரப்படும் தகவல் (ஒவ்வொரு புள்ளியையும் பட்டியலிடவும்)",
        "telugu": "కోరిన సమాచారం (ప్రతి అంశాన్ని జాబితా చేయండి)",
        "kannada": "ಕೋರಿದ ಮಾಹಿತಿ (ಪ್ರತಿ ಅಂಶವನ್ನು ಪಟ್ಟಿ ಮಾಡಿ)",
        "bengali": "চাওয়া তথ্য (প্রতিটি বিষয় তালিকাভুক্ত করুন)",
        "malayalam": "ആവശ്യപ്പെടുന്ന വിവരം (ഓരോ ഇനവും പട്ടികപ്പെടുത്തുക)", "marathi": "मागितलेली माहिती (प्रत्येक मुद्दा नमूद करा)", "gujarati": "માંગેલી માહિતી (દરેક મુદ્દો યાદી કરો)", "punjabi": "ਮੰਗੀ ਗਈ ਜਾਣਕਾਰੀ (ਹਰੇਕ ਨੁਕਤਾ ਸੂਚੀਬੱਧ ਕਰੋ)", "odia": "ମାଗିଥିବା ସୂଚନା (ପ୍ରତ୍ୟେକ ବିଷୟ ତାଲିକାଭୁକ୍ତ କରନ୍ତୁ)", "urdu": "مطلوبہ معلومات (ہر نکتہ درج کریں)",
    },
    "period_concerned": {
        "tamil": "தகவல் தொடர்பான காலம்", "telugu": "సమాచారం సంబంధించిన కాలం",
        "kannada": "ಮಾಹಿತಿ ಸಂಬಂಧಿಸಿದ ಅವಧಿ", "bengali": "তথ্য সংশ্লিষ্ট সময়কাল",
        "malayalam": "ബന്ധപ്പെട്ട കാലയളവ്", "marathi": "संबंधित कालावधी", "gujarati": "સંબંધિત સમયગાળો", "punjabi": "ਸੰਬੰਧਿਤ ਸਮਾਂ", "odia": "ସମ୍ବନ୍ଧିତ ସମୟ", "urdu": "متعلقہ مدت",
    },
    "bpl_status": {
        "tamil": "வறுமைக் கோட்டிற்குக் கீழ் (கட்டணம் விலக்கு)?",
        "telugu": "దారిద్ర్య రేఖకు దిగువన (రుసుము మినహాయింపు)?",
        "kannada": "ಬಡತನ ರೇಖೆಗಿಂತ ಕೆಳಗೆ (ಶುಲ್ಕ ವಿನಾಯಿತಿ)?",
        "bengali": "দারিদ্র্যসীমার নিচে (ফি মওকুফ)?",
        "malayalam": "ദാരിദ്ര്യരേഖയ്ക്ക് താഴെ (ഫീസ് ഇളവ്)?", "marathi": "दारिद्र्यरेषेखालील (शुल्क सूट)?", "gujarati": "ગરીબી રેખા નીચે (ફી માફી)?", "punjabi": "ਗਰੀਬੀ ਰੇਖਾ ਤੋਂ ਹੇਠਾਂ (ਫੀਸ ਛੋਟ)?", "odia": "ଦାରିଦ୍ର୍ୟ ସୀମାରେଖା ତଳେ (ଫି ମୁକ୍ତି)?", "urdu": "خط غربت سے نیچے (فیس میں چھوٹ)؟",
    },
    "city": {
        "tamil": "நகரம்", "telugu": "నగరం", "kannada": "ನಗರ", "bengali": "শহর",
        "malayalam": "നഗരം", "marathi": "शहर", "gujarati": "શહેર", "punjabi": "ਸ਼ਹਿਰ", "odia": "ସହର", "urdu": "شہر",
    },
    "fraud_amount": {
        "tamil": "மோசடி தொகை (ரூ.)", "telugu": "మోసం మొత్తం (రూ.)",
        "kannada": "ವಂಚನೆ ಮೊತ್ತ (ರೂ.)", "bengali": "প্রতারণার পরিমাণ (টাকা)",
        "malayalam": "തട്ടിപ്പ് തുക (₹)", "marathi": "फसवणुकीची रक्कम (₹)", "gujarati": "છેતરપિંડીની રકમ (₹)", "punjabi": "ਧੋਖਾਧੜੀ ਦੀ ਰਕਮ (₹)", "odia": "ଠକେଇ ପରିମାଣ (₹)", "urdu": "فراڈ کی رقم (₹)",
    },
    "bank_name": {
        "tamil": "வங்கியின் பெயர்", "telugu": "బ్యాంకు పేరు",
        "kannada": "ಬ್ಯಾಂಕ್ ಹೆಸರು", "bengali": "ব্যাংকের নাম",
        "malayalam": "ബാങ്കിന്റെ പേര്", "marathi": "बँकेचे नाव", "gujarati": "બેંકનું નામ", "punjabi": "ਬੈਂਕ ਦਾ ਨਾਮ", "odia": "ବ୍ୟାଙ୍କ ନାମ", "urdu": "بینک کا نام",
    },
    "transaction_id": {
        "tamil": "பரிவர்த்தனை ஐடி / UTR எண்", "telugu": "లావాదేవీ ఐడీ / UTR సంఖ్య",
        "kannada": "ವಹಿವಾಟು ಐಡಿ / UTR ಸಂಖ್ಯೆ", "bengali": "লেনদেন আইডি / ইউটিআর নম্বর",
        "malayalam": "ഇടപാട് ഐഡി / UTR നമ്പർ", "marathi": "व्यवहार आयडी / UTR क्रमांक", "gujarati": "વ્યવહાર આઈડી / UTR નંબર", "punjabi": "ਲੈਣ-ਦੇਣ ਆਈਡੀ / UTR ਨੰਬਰ", "odia": "କାରବାର ID / UTR ନମ୍ବର", "urdu": "ٹرانزیکشن آئی ڈی / یو ٹی آر نمبر",
    },
    "fraud_type": {
        "tamil": "சைபர் மோசடியின் வகை", "telugu": "సైబర్ మోసం రకం",
        "kannada": "ಸೈಬರ್ ವಂಚನೆಯ ಪ್ರಕಾರ", "bengali": "সাইবার প্রতারণার ধরন",
        "malayalam": "സൈബർ തട്ടിപ്പിന്റെ തരം", "marathi": "सायबर फसवणुकीचा प्रकार", "gujarati": "સાયબર છેતરપિંડીનો પ્રકાર", "punjabi": "ਸਾਈਬਰ ਧੋਖਾਧੜੀ ਦੀ ਕਿਸਮ", "odia": "ସାଇବର ଠକେଇ ପ୍ରକାର", "urdu": "سائبر فراڈ کی قسم",
    },
    "rented_premises_address": {
        "tamil": "வாடகை சொத்தின் முகவரி", "telugu": "అద్దె ఆస్తి చిరునామా",
        "kannada": "ಬಾಡಿಗೆ ಆಸ್ತಿಯ ವಿಳಾಸ", "bengali": "ভাড়া করা সম্পত্তির ঠিকানা",
        "malayalam": "വാടക സ്വത്തിന്റെ വിലാസം", "marathi": "भाड्याच्या मालमत्तेचा पत्ता", "gujarati": "ભાડાની મિલકતનું સરનામું", "punjabi": "ਕਿਰਾਏ ਦੀ ਜਾਇਦਾਦ ਦਾ ਪਤਾ", "odia": "ଭଡା ସମ୍ପତ୍ତିର ଠିକଣା", "urdu": "کرائے کی جائیداد کا پتہ",
    },
    "sale_consideration_amount": {
        "tamil": "மொத்த விற்பனை பரிசீலனை (ரூ.)", "telugu": "మొత్తం అమ్మకం విలువ (రూ.)",
        "kannada": "ಒಟ್ಟು ಮಾರಾಟ ಪರಿಗಣನೆ (ರೂ.)", "bengali": "মোট বিক্রয় মূল্য (টাকা)",
        "malayalam": "മൊത്തം വിൽപ്പന പ്രതിഫലം (₹)", "marathi": "एकूण विक्री मोबदला (₹)",
        "gujarati": "કુલ વેચાણ વળતર (₹)", "punjabi": "ਕੁੱਲ ਵਿਕਰੀ ਮੁੱਲ (₹)",
        "odia": "ମୋଟ ବିକ୍ରୟ ପ୍ରତିଫଳ (₹)", "urdu": "کل فروخت کی قیمت (₹)",
    },
    "monthly_rent_amount": {
        "tamil": "மாதாந்திர வாடகை (ரூ.)", "telugu": "నెలవారీ అద్దె (రూ.)",
        "kannada": "ಮಾಸಿಕ ಬಾಡಿಗೆ (ರೂ.)", "bengali": "মাসিক ভাড়া (টাকা)",
        "malayalam": "പ്രതിമാസ വാടക (₹)", "marathi": "मासिक भाडे (₹)", "gujarati": "માસિક ભાડું (₹)",
        "punjabi": "ਮਹੀਨਾਵਾਰ ਕਿਰਾਇਆ (₹)", "odia": "ମାସିକ ଭଡା (₹)", "urdu": "ماہانہ کرایہ (₹)",
    },
    "security_deposit_amount": {
        "tamil": "பாதுகாப்பு வைப்புத்தொகை (ரூ.)", "telugu": "సెక్యూరిటీ డిపాజిట్ మొత్తం (రూ.)",
        "kannada": "ಭದ್ರತಾ ಠೇವಣಿ ಮೊತ್ತ (ರೂ.)", "bengali": "জামানতের পরিমাণ (টাকা)",
        "malayalam": "സെക്യൂരിറ്റി ഡെപ്പോസിറ്റ് തുക (₹)", "marathi": "सुरक्षा ठेव रक्कम (₹)", "gujarati": "સિક્યોરિટી ડિપોઝિટ રકમ (₹)", "punjabi": "ਸੁਰੱਖਿਆ ਜਮ੍ਹਾਂ ਰਕਮ (₹)", "odia": "ସୁରକ୍ଷା ଜମା ପରିମାଣ (₹)", "urdu": "سیکیورٹی ڈپازٹ کی رقم (₹)",
    },
    "tenancy_start_date": {
        "tamil": "குடியிருப்பு தொடங்கிய தேதி", "telugu": "అద్దె ప్రారంభ తేదీ",
        "kannada": "ಬಾಡಿಗೆ ಪ್ರಾರಂಭ ದಿನಾಂಕ", "bengali": "ভাড়া শুরুর তারিখ",
        "malayalam": "വാടക ആരംഭിച്ച തീയതി", "marathi": "भाडेकरार सुरू झाल्याची तारीख", "gujarati": "ભાડા શરૂ થવાની તારીખ", "punjabi": "ਕਿਰਾਏਦਾਰੀ ਸ਼ੁਰੂ ਹੋਣ ਦੀ ਮਿਤੀ", "odia": "ଭଡା ଆରମ୍ଭ ତାରିଖ", "urdu": "کرایہ داری شروع ہونے کی تاریخ",
    },
    "tenancy_end_date": {
        "tamil": "குடியிருப்பு முடிவு / காலி செய்யும் தேதி", "telugu": "అద్దె ముగింపు / ఖాళీ చేసే తేదీ",
        "kannada": "ಬಾಡಿಗೆ ಮುಕ್ತಾಯ / ಖಾಲಿ ಮಾಡುವ ದಿನಾಂಕ", "bengali": "ভাড়া শেষ / খালি করার তারিখ",
        "malayalam": "വാടക അവസാനിക്കുന്ന / ഒഴിയേണ്ട തീയതി", "marathi": "भाडेकरार संपण्याची / जागा रिकामी करण्याची तारीख", "gujarati": "ભાડું સમાપ્ત / ખાલી કરવાની તારીખ", "punjabi": "ਕਿਰਾਏਦਾਰੀ ਖਤਮ / ਖਾਲੀ ਕਰਨ ਦੀ ਮਿਤੀ", "odia": "ଭଡା ସମାପ୍ତି / ଖାଲି କରିବା ତାରିଖ", "urdu": "کرایہ داری ختم / خالی کرنے کی تاریخ",
    },
    "affidavit_purpose": {
        "tamil": "சத்தியக்கடதாசியின் நோக்கம்", "telugu": "అఫిడవిట్ ప్రయోజనం",
        "kannada": "ಪ್ರಮಾಣಪತ್ರದ ಉದ್ದೇಶ", "bengali": "হলফনামার উদ্দেশ্য",
        "malayalam": "സത്യപ്രസ്താവനയുടെ ഉദ്ദേശ്യം", "marathi": "प्रतिज्ञापत्राचा उद्देश", "gujarati": "સોગંદનામાનો હેતુ", "punjabi": "ਹਲਫ਼ਨਾਮੇ ਦਾ ਮਕਸਦ", "odia": "ଶପଥପତ୍ରର ଉଦ୍ଦେଶ୍ୟ", "urdu": "حلف نامے کا مقصد",
    },
    "applicant_father_name": {
        "tamil": "தந்தை / கணவர் பெயர்", "telugu": "తండ్రి / భర్త పేరు",
        "kannada": "ತಂದೆ / ಪತಿಯ ಹೆಸರು", "bengali": "পিতা / স্বামীর নাম",
        "malayalam": "പിതാവിന്റെ / ഭർത്താവിന്റെ പേര്", "marathi": "वडिलांचे / पतीचे नाव", "gujarati": "પિતા / પતિનું નામ", "punjabi": "ਪਿਤਾ / ਪਤੀ ਦਾ ਨਾਮ", "odia": "ପିତା / ସ୍ୱାମୀଙ୍କ ନାମ", "urdu": "والد / شوہر کا نام",
    },
    "deponent_age": {
        "tamil": "வயது", "telugu": "వయస్సు", "kannada": "ವಯಸ್ಸು", "bengali": "বয়স",
        "malayalam": "വയസ്സ്", "marathi": "वय", "gujarati": "ઉંમર", "punjabi": "ਉਮਰ", "odia": "ବୟସ", "urdu": "عمر",
    },
    "prior_complaint_reference": {
        "tamil": "முந்தைய புகார் குறிப்பு (தேதி/டைரி எண், இருந்தால்)",
        "telugu": "మునుపటి ఫిర్యాదు సూచన (తేదీ/డైరీ నంబర్, ఉంటే)",
        "kannada": "ಹಿಂದಿನ ದೂರಿನ ಉಲ್ಲೇಖ (ದಿನಾಂಕ/ಡೈರಿ ಸಂಖ್ಯೆ, ಇದ್ದರೆ)",
        "bengali": "পূর্ববর্তী অভিযোগের রেফারেন্স (তারিখ/ডায়েরি নম্বর, যদি থাকে)",
        "malayalam": "മുൻ പരാതി പരാമർശം (തീയതി/ഡയറി നമ്പർ, ഉണ്ടെങ്കിൽ)", "marathi": "आधीच्या तक्रारीचा संदर्भ (तारीख/डायरी क्रमांक, असल्यास)", "gujarati": "અગાઉની ફરિયાદનો સંદર્ભ (તારીખ/ડાયરી નંબર, હોય તો)", "punjabi": "ਪਿਛਲੀ ਸ਼ਿਕਾਇਤ ਦਾ ਹਵਾਲਾ (ਮਿਤੀ/ਡਾਇਰੀ ਨੰਬਰ, ਜੇ ਹੋਵੇ)", "odia": "ପୂର୍ବ ଅଭିଯୋଗ ସନ୍ଦର୍ଭ (ତାରିଖ/ଡାଏରୀ ନମ୍ବର, ଥିଲେ)", "urdu": "پچھلی شکایت کا حوالہ (تاریخ/ڈائری نمبر، اگر ہو)",
    },
    "cheque_number": {
        "tamil": "காசோலை எண்", "telugu": "చెక్ నంబర్",
        "kannada": "ಚೆಕ್ ಸಂಖ್ಯೆ", "bengali": "চেক নম্বর",
        "malayalam": "ചെക്ക് നമ്പർ", "marathi": "धनादेश क्रमांक", "gujarati": "ચેક નંબર", "punjabi": "ਚੈੱਕ ਨੰਬਰ", "odia": "ଚେକ୍ ନମ୍ବର", "urdu": "چیک نمبر",
    },
    "cheque_amount": {
        "tamil": "காசோலை தொகை (ரூ.)", "telugu": "చెక్ మొత్తం (రూ.)",
        "kannada": "ಚೆಕ್ ಮೊತ್ತ (ರೂ.)", "bengali": "চেকের পরিমাণ (টাকা)",
        "malayalam": "ചെക്ക് തുക (₹)", "marathi": "धनादेश रक्कम (₹)", "gujarati": "ચેક રકમ (₹)", "punjabi": "ਚੈੱਕ ਰਕਮ (₹)", "odia": "ଚେକ୍ ପରିମାଣ (₹)", "urdu": "چیک کی رقم (₹)",
    },
    "cheque_date": {
        "tamil": "காசோலை தேதி", "telugu": "చెక్ తేదీ",
        "kannada": "ಚೆಕ್ ದಿನಾಂಕ", "bengali": "চেকের তারিখ",
        "malayalam": "ചെക്ക് തീയതി", "marathi": "धनादेश तारीख", "gujarati": "ચેક તારીખ", "punjabi": "ਚੈੱਕ ਮਿਤੀ", "odia": "ଚେକ୍ ତାରିଖ", "urdu": "چیک کی تاریخ",
    },
    "dishonour_reason": {
        "tamil": "நிராகரிக்கப்பட்ட காரணம்", "telugu": "తిరస్కరణకు కారణం",
        "kannada": "ತಿರಸ್ಕಾರಕ್ಕೆ ಕಾರಣ", "bengali": "অসম্মানের কারণ",
        "malayalam": "നിരസിക്കാനുള്ള കാരണം", "marathi": "नाकारण्याचे कारण", "gujarati": "અસ્વીકારનું કારણ", "punjabi": "ਅਸਵੀਕਾਰ ਕਰਨ ਦਾ ਕਾਰਨ", "odia": "ପ୍ରତ୍ୟାଖ୍ୟାନ କାରଣ", "urdu": "بے عزتی کی وجہ",
    },
    "dishonour_date": {
        "tamil": "திரும்ப அனுப்பப்பட்ட மெமோ தேதி", "telugu": "రిటర్న్ మెమో తేదీ",
        "kannada": "ರಿಟರ್ನ್ ಮೆಮೊ ದಿನಾಂಕ", "bengali": "রিটার্ন মেমোর তারিখ",
        "malayalam": "റിട്ടേൺ മെമ്മോ തീയതി", "marathi": "परतावा मेमो तारीख", "gujarati": "રિટર્ન મેમો તારીખ", "punjabi": "ਵਾਪਸੀ ਮੈਮੋ ਮਿਤੀ", "odia": "ରିଟର୍ନ ମେମୋ ତାରିଖ", "urdu": "ریٹرن میمو کی تاریخ",
    },
    "principal_amount": {
        "tamil": "முதன்மைத் தொகை நிலுவை (ரூ.)", "telugu": "అసలు బకాయి మొత్తం (రూ.)",
        "kannada": "ಮೂಲ ಬಾಕಿ ಮೊತ್ತ (ರೂ.)", "bengali": "মূল বকেয়া পরিমাণ (টাকা)",
        "malayalam": "മുതൽ കുടിശ്ശിക തുക (₹)", "marathi": "मुद्दल थकबाकी रक्कम (₹)", "gujarati": "મુદ્દલ બાકી રકમ (₹)", "punjabi": "ਮੂਲ ਬਕਾਇਆ ਰਕਮ (₹)", "odia": "ମୂଳ ବକେୟା ପରିମାଣ (₹)", "urdu": "اصل بقایا رقم (₹)",
    },
    "amount_due_since": {
        "tamil": "தொகை நிலுவையில் உள்ள தேதி", "telugu": "మొత్తం బకాయి ఉన్న తేదీ",
        "kannada": "ಮೊತ್ತ ಬಾಕಿ ಇರುವ ದಿನಾಂಕ", "bengali": "যে তারিখ থেকে টাকা বকেয়া",
        "malayalam": "തുക കുടിശ്ശികയായ തീയതി", "marathi": "रक्कम थकीत असल्याची तारीख", "gujarati": "રકમ બાકી હોવાની તારીખ", "punjabi": "ਰਕਮ ਬਕਾਇਆ ਹੋਣ ਦੀ ਮਿਤੀ", "odia": "ପରିମାଣ ବକେୟା ତାରିଖ", "urdu": "رقم واجب الادا ہونے کی تاریخ",
    },
    "interest_claimed": {
        "tamil": "கோரப்படும் வட்டி (இருந்தால்)", "telugu": "క్లెయిమ్ చేసిన వడ్డీ (ఉంటే)",
        "kannada": "ಕ್ಲೈಮ್ ಮಾಡಿದ ಬಡ್ಡಿ (ಇದ್ದರೆ)", "bengali": "দাবিকৃত সুদ (যদি থাকে)",
        "malayalam": "ആവശ്യപ്പെടുന്ന പലിശ (ഉണ്ടെങ്കിൽ)", "marathi": "दावा केलेले व्याज (असल्यास)", "gujarati": "દાવો કરેલ વ્યાજ (હોય તો)", "punjabi": "ਦਾਅਵਾ ਕੀਤਾ ਵਿਆਜ (ਜੇ ਹੋਵੇ)", "odia": "ଦାବି କରାଯାଇଥିବା ସୁଧ (ଥିଲେ)", "urdu": "دعویٰ کردہ سود (اگر ہو)",
    },
    "product_or_service": {
        "tamil": "தொடர்புடைய பொருள் / சேவை", "telugu": "సంబంధిత ఉత్పత్తి / సేవ",
        "kannada": "ಸಂಬಂಧಿತ ಉತ್ಪನ್ನ / ಸೇವೆ", "bengali": "সংশ্লিষ্ট পণ্য / পরিষেবা",
        "malayalam": "ബന്ധപ്പെട്ട ഉൽപ്പന്നം / സേവനം", "marathi": "संबंधित उत्पादन / सेवा", "gujarati": "સંબંધિત ઉત્પાદન / સેવા", "punjabi": "ਸੰਬੰਧਿਤ ਉਤਪਾਦ / ਸੇਵਾ", "odia": "ସମ୍ବନ୍ଧିତ ଉତ୍ପାଦ / ସେବା", "urdu": "متعلقہ پروڈکٹ / سروس",
    },
    "purchase_date": {
        "tamil": "கொள்முதல் / சேவை தேதி", "telugu": "కొనుగోలు / సేవ తేదీ",
        "kannada": "ಖರೀದಿ / ಸೇವಾ ದಿನಾಂಕ", "bengali": "ক্রয় / পরিষেবার তারিখ",
        "malayalam": "വാങ്ങിയ / സേവന തീയതി", "marathi": "खरेदी / सेवा तारीख", "gujarati": "ખરીદી / સેવા તારીખ", "punjabi": "ਖਰੀਦ / ਸੇਵਾ ਮਿਤੀ", "odia": "କ୍ରୟ / ସେବା ତାରିଖ", "urdu": "خریداری / سروس کی تاریخ",
    },
    "amount_paid": {
        "tamil": "செலுத்திய தொகை (ரூ.)", "telugu": "చెల్లించిన మొత్తం (రూ.)",
        "kannada": "ಪಾವತಿಸಿದ ಮೊತ್ತ (ರೂ.)", "bengali": "প্রদত্ত পরিমাণ (টাকা)",
        "malayalam": "അടച്ച തുക (₹)", "marathi": "भरलेली रक्कम (₹)", "gujarati": "ચૂકવેલ રકમ (₹)", "punjabi": "ਅਦਾ ਕੀਤੀ ਰਕਮ (₹)", "odia": "ପ୍ରଦତ୍ତ ପରିମାଣ (₹)", "urdu": "ادا شدہ رقم (₹)",
    },
    "designation": {
        "tamil": "பதவி", "telugu": "హోదా", "kannada": "ಹುದ್ದೆ", "bengali": "পদবি",
        "malayalam": "ഉദ്യോഗപ്പേര്", "marathi": "पदनाम", "gujarati": "હોદ્દો", "punjabi": "ਅਹੁਦਾ", "odia": "ପଦବୀ", "urdu": "عہدہ",
    },
    "employment_start_date": {
        "tamil": "சேரும் தேதி", "telugu": "చేరిన తేదీ",
        "kannada": "ಸೇರಿದ ದಿನಾಂಕ", "bengali": "যোগদানের তারিখ",
        "malayalam": "ജോലിയിൽ ചേർന്ന തീയതി", "marathi": "रुजू झाल्याची तारीख", "gujarati": "જોડાવાની તારીખ", "punjabi": "ਸ਼ਾਮਲ ਹੋਣ ਦੀ ਮਿਤੀ", "odia": "ଯୋଗଦାନ ତାରିଖ", "urdu": "شمولیت کی تاریخ",
    },
    "employment_end_date": {
        "tamil": "பணி நீக்கம் / ராஜினாமா தேதி", "telugu": "తొలగింపు / రాజీనామా తేదీ",
        "kannada": "ವಜಾ / ರಾಜೀನಾಮೆ ದಿನಾಂಕ", "bengali": "বরখাস্ত / পদত্যাগের তারিখ",
        "malayalam": "പിരിച്ചുവിടൽ / രാജി തീയതി", "marathi": "बडतर्फी / राजीनामा तारीख", "gujarati": "બરતરફી / રાજીનામાની તારીખ", "punjabi": "ਬਰਖਾਸਤਗੀ / ਅਸਤੀਫ਼ਾ ਮਿਤੀ", "odia": "ବରଖାସ୍ତ / ଇସ୍ତଫା ତାରିଖ", "urdu": "برخاستگی / استعفیٰ کی تاریخ",
    },
    "dues_amount": {
        "tamil": "நிலுவைத் தொகை (ரூ.)", "telugu": "బకాయి మొత్తం (రూ.)",
        "kannada": "ಬಾಕಿ ಮೊತ್ತ (ರೂ.)", "bengali": "বকেয়া পরিমাণ (টাকা)",
        "malayalam": "കുടിശ്ശിക തുക (₹)", "marathi": "थकबाकी रक्कम (₹)", "gujarati": "બાકી રકમ (₹)", "punjabi": "ਬਕਾਇਆ ਰਕਮ (₹)", "odia": "ବକେୟା ପରିମାଣ (₹)", "urdu": "بقایا رقم (₹)",
    },
}

# Per-(draft_id, key) overrides for the handful of fields whose English
# `label` wording differs by template even though the field `key` (and its
# core underlying concept) is the same -- e.g. `police_station` reads plain
# "Police Station" on the Police Complaint template but "Police Station
# (nearest, for jurisdiction)" on Cyber Crime Complaint and "Concerned
# Police Station" on the SP escalation complaint. Falling back to the
# generic `_FIELD_LABEL_TRANSLATIONS` entry for these would translate the
# concept correctly but silently drop the template-specific qualifier the
# English label carries.
_FIELD_LABEL_OVERRIDES: dict[tuple[str, str], dict[str, str]] = {
    ("cyber_crime_complaint", "police_station"): {
        "tamil": "காவல் நிலையம் (அருகிலுள்ளது, அதிகார எல்லைக்காக)",
        "telugu": "పోలీసు స్టేషన్ (సమీపంలోని, పరిధి కోసం)",
        "kannada": "ಪೊಲೀಸ್ ಠಾಣೆ (ಹತ್ತಿರದ, ವ್ಯಾಪ್ತಿಗಾಗಿ)",
        "bengali": "থানা (নিকটতম, এখতিয়ারের জন্য)",
    },
    ("sp_complaint", "police_station"): {
        "tamil": "சம்பந்தப்பட்ட காவல் நிலையம்", "telugu": "సంబంధిత పోలీసు స్టేషన్",
        "kannada": "ಸಂಬಂಧಿತ ಪೊಲೀಸ್ ಠಾಣೆ", "bengali": "সংশ্লিষ্ট থানা",
    },
    ("cheque_bounce_notice", "bank_name"): {
        "tamil": "காசோலை வழங்கிய வங்கி பெயர்", "telugu": "చెక్ జారీ చేసిన బ్యాంకు పేరు",
        "kannada": "ಚೆಕ್ ನೀಡಿದ ಬ್ಯಾಂಕ್ ಹೆಸರು", "bengali": "চেক প্রদানকারী ব্যাংকের নাম",
    },
}


def localized_field_label(draft_id: str, draft_field: DraftField, language: str) -> str:
    """The display label for `draft_field` in `language`.

    "hindi" reuses `draft_field.hindi_label` (already authored per field in
    the template YAML, not duplicated here). Falls back to
    `draft_field.label` (the English original) for any language outside
    the fixed tamil/telugu/kannada/bengali set, or any field key not yet
    covered above -- never raises, never fabricates a translation.
    """
    if language == "hindi":
        return draft_field.hindi_label
    override = _FIELD_LABEL_OVERRIDES.get((draft_id, draft_field.key))
    if override and language in override:
        return override[language]
    return _FIELD_LABEL_TRANSLATIONS.get(draft_field.key, {}).get(language, draft_field.label)
