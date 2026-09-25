"""Problem-first discovery follow-up: a translated one-line `description`
for the templates that actually participate in the recommendation engine
(`app/drafting/recommendation.py`) -- the "{usage}" half of
`discovery_recommend_one` ("So 'X' is the suitable document. {usage}"),
previously always `template.description` verbatim regardless of the
conversation's language, so even a fully Tamil/Telugu/... reply ended with
one leftover English sentence.

Scoped to the 8 templates onboarded with discovery metadata in this pass
(see `app/drafting/templates/*.yaml`'s `domain`/`user_roles`/etc. fields),
not all 55 -- translating every template's description into ten languages is
a much larger, separately-scoped effort (550 sentences) disproportionate to
what discovery currently surfaces; the other 47 templates simply show their
English description, exactly as before this module existed.

Statutory references (act names, section numbers -- e.g. "Negotiable
Instruments Act, 1881", "Section 138") are deliberately left in English
inside every translation below rather than transliterated or translated:
these are proper nouns naming a specific, exact legal citation, and
mistranslating or subtly altering one would misstate the law, which this
whole feature exists to never do (see the "SAFETY AND LEGAL QUALITY"
requirement to never invent statutory content). Only the surrounding
descriptive language is translated.
"""

from app.drafting.templates.base import DraftTemplateDefinition

_DESCRIPTION_TRANSLATIONS: dict[str, dict[str, str]] = {
    "rent_agreement": {
        "tamil": (
            "வீட்டு உரிமையாளருக்கும் வாடகைதாரருக்கும் இடையேயான வாடகை ஒப்பந்தம், வாடகை, "
            "பாதுகாப்பு வைப்புத்தொகை, காலம் மற்றும் நிபந்தனைகளை நிர்ணயிக்கிறது."
        ),
        "telugu": (
            "ఇంటి యజమాని మరియు అద్దెదారు మధ్య అద్దె ఒప్పందం, అద్దె, సెక్యూరిటీ డిపాజిట్, "
            "కాలవ్యవధి మరియు షరతులను నిర్ణయిస్తుంది."
        ),
        "kannada": (
            "ಮನೆ ಮಾಲೀಕ ಮತ್ತು ಬಾಡಿಗೆದಾರರ ನಡುವಿನ ಬಾಡಿಗೆ ಒಪ್ಪಂದ, ಬಾಡಿಗೆ, ಭದ್ರತಾ ಠೇವಣಿ, ಅವಧಿ ಮತ್ತು "
            "ಷರತ್ತುಗಳನ್ನು ನಿಗದಿಪಡಿಸುತ್ತದೆ."
        ),
        "bengali": "বাড়িওয়ালা ও ভাড়াটিয়ার মধ্যে ভাড়া চুক্তি, যা ভাড়া, নিরাপত্তা জামানত, মেয়াদ এবং শর্তাবলী নির্ধারণ করে।",
        "malayalam": (
            "വീട്ടുടമയും വാടകക്കാരനും തമ്മിലുള്ള വാടക കരാർ, വാടക, സെക്യൂരിറ്റി ഡെപ്പോസിറ്റ്, "
            "കാലാവധി, നിബന്ധനകൾ എന്നിവ നിശ്ചയിക്കുന്നു."
        ),
        "marathi": "घरमालक आणि भाडेकरू यांच्यातील भाडे करार, जो भाडे, सुरक्षा ठेव, कालावधी आणि अटी निश्चित करतो.",
        "gujarati": "મકાનમાલિક અને ભાડુઆત વચ્ચેનો ભાડા કરાર, જે ભાડું, સિક્યુરિટી ડિપોઝિટ, મુદત અને શરતો નક્કી કરે છે.",
        "punjabi": "ਮਕਾਨ ਮਾਲਕ ਅਤੇ ਕਿਰਾਏਦਾਰ ਵਿਚਕਾਰ ਕਿਰਾਏ ਦਾ ਸਮਝੌਤਾ, ਜੋ ਕਿਰਾਇਆ, ਸੁਰੱਖਿਆ ਜਮ੍ਹਾਂ, ਮਿਆਦ ਅਤੇ ਸ਼ਰਤਾਂ ਤੈਅ ਕਰਦਾ ਹੈ।",
        "odia": "ମାଲିକ ଏବଂ ଭଡ଼ାଟିଆଙ୍କ ମଧ୍ୟରେ ଭଡ଼ା ଚୁକ୍ତି, ଯାହା ଭଡ଼ା, ସୁରକ୍ଷା ଜମା, ଅବଧି ଏବଂ ସର୍ତ୍ତାବଳୀ ନିର୍ଦ୍ଧାରଣ କରେ।",
        "urdu": "مکان مالک اور کرایہ دار کے درمیان کرایہ نامہ، جو کرایہ، حفاظتی رقم، مدت اور شرائط طے کرتا ہے۔",
    },
    "tenancy_termination_notice": {
        "tamil": (
            "குடியிருப்பை முடிவுக்குக் கொண்டுவரும் சட்ட அறிவிப்பு, வாடகைதாரர் வீட்டை காலி "
            "செய்து ஒப்படைக்கக் கோருகிறது."
        ),
        "telugu": "అద్దెను ముగించి, ఆస్తిని ఖాళీ చేసి అప్పగించాలని అద్దెదారుని కోరే చట్టపరమైన నోటీసు.",
        "kannada": "ಬಾಡಿಗೆಯನ್ನು ಕೊನೆಗೊಳಿಸಿ, ಆಸ್ತಿಯನ್ನು ಖಾಲಿ ಮಾಡಿ ಒಪ್ಪಿಸಲು ಬಾಡಿಗೆದಾರರನ್ನು ಕೇಳುವ ಕಾನೂನು ನೋಟಿಸ್.",
        "bengali": "ভাড়া বাতিল করে সম্পত্তি খালি করে হস্তান্তর করতে ভাড়াটিয়াকে অনুরোধ করা আইনি নোটিশ।",
        "malayalam": "വാടക അവസാനിപ്പിച്ച് വസ്തു ഒഴിഞ്ഞ് കൈമാറാൻ വാടകക്കാരനോട് ആവശ്യപ്പെടുന്ന നിയമ നോട്ടീസ്.",
        "marathi": "भाडेकरार संपुष्टात आणून, जागा रिकामी करून ताबा देण्याची विनंती करणारी कायदेशीर नोटीस.",
        "gujarati": "ભાડું સમાપ્ત કરીને મિલકત ખાલી કરી સોંપવા ભાડુઆતને વિનંતી કરતી કાનૂની નોટિસ.",
        "punjabi": "ਕਿਰਾਏਦਾਰੀ ਖਤਮ ਕਰਕੇ ਜਾਇਦਾਦ ਖਾਲੀ ਕਰਨ ਅਤੇ ਸੌਂਪਣ ਦੀ ਬੇਨਤੀ ਕਰਨ ਵਾਲਾ ਕਾਨੂੰਨੀ ਨੋਟਿਸ।",
        "odia": "ଭଡ଼ାଟିଆକୁ ସମ୍ପତ୍ତି ଖାଲି କରି ହସ୍ତାନ୍ତର କରିବାକୁ ଅନୁରୋଧ କରୁଥିବା ଏବଂ ଭଡ଼ା ସମାପ୍ତ କରୁଥିବା ଆଇନଗତ ବିଜ୍ଞପ୍ତି।",
        "urdu": "کرایہ داری ختم کرکے جائیداد خالی کرنے اور حوالے کرنے کی درخواست کرنے والا قانونی نوٹس۔",
    },
    "cheque_bounce_notice": {
        "tamil": (
            "காசோலை நிராகரிக்கப்பட்ட பிறகு பணம் செலுத்த வேண்டுமெனக் கோரும், Negotiable "
            "Instruments Act, 1881-ன் Section 138-ன் கீழான சட்டப்பூர்வ கோரிக்கை அறிவிப்பு."
        ),
        "telugu": (
            "చెక్ తిరస్కరించబడిన తర్వాత చెల్లింపు కోరుతూ, Negotiable Instruments Act, 1881లోని "
            "Section 138 ప్రకారం చట్టబద్ధమైన డిమాండ్ నోటీసు."
        ),
        "kannada": (
            "ಚೆಕ್ ತಿರಸ್ಕರಿಸಲ್ಪಟ್ಟ ನಂತರ ಪಾವತಿಯನ್ನು ಕೇಳುವ, Negotiable Instruments Act, 1881ರ "
            "Section 138ರ ಅಡಿಯಲ್ಲಿ ಶಾಸನಬದ್ಧ ಬೇಡಿಕೆ ನೋಟಿಸ್."
        ),
        "bengali": (
            "চেক প্রত্যাখ্যাত হওয়ার পর অর্থ প্রদানের দাবিতে, Negotiable Instruments Act, "
            "1881-এর Section 138 অনুযায়ী বিধিবদ্ধ দাবি নোটিশ।"
        ),
        "malayalam": (
            "ചെക്ക് നിരസിച്ചതിന് ശേഷം പണം ആവശ്യപ്പെടുന്ന, Negotiable Instruments Act, 1881-ലെ "
            "Section 138 പ്രകാരമുള്ള നിയമപരമായ ഡിമാൻഡ് നോട്ടീസ്."
        ),
        "marathi": (
            "चेक नाकारल्यानंतर पैसे देण्याची मागणी करणारी, Negotiable Instruments Act, 1881 च्या "
            "Section 138 अंतर्गत वैधानिक मागणी नोटीस."
        ),
        "gujarati": (
            "ચેક નકારાયા પછી ચુકવણીની માંગ કરતી, Negotiable Instruments Act, 1881 ની Section "
            "138 હેઠળની વૈધાનિક માંગ નોટિસ."
        ),
        "punjabi": (
            "ਚੈੱਕ ਰੱਦ ਹੋਣ ਤੋਂ ਬਾਅਦ ਭੁਗਤਾਨ ਦੀ ਮੰਗ ਕਰਦਾ, Negotiable Instruments Act, 1881 ਦੀ "
            "Section 138 ਤਹਿਤ ਕਾਨੂੰਨੀ ਮੰਗ ਨੋਟਿਸ।"
        ),
        "odia": (
            "ଚେକ୍ ପ୍ରତ୍ୟାଖ୍ୟାନ ହେବା ପରେ ଦେୟ ଦାବି କରୁଥିବା, Negotiable Instruments Act, 1881 ର "
            "Section 138 ଅଧୀନରେ ବିଧିବଦ୍ଧ ଦାବି ବିଜ୍ଞପ୍ତି।"
        ),
        "urdu": (
            "چیک مسترد ہونے کے بعد ادائیگی کا مطالبہ کرنے والا، Negotiable Instruments Act, "
            "1881 کی Section 138 کے تحت قانونی مطالبہ نوٹس۔"
        ),
    },
    "police_complaint": {
        "tamil": (
            "FIR பதிவு செய்யவோ அல்லது பொருத்தமான காவல்துறை நடவடிக்கை எடுக்கவோ கோரும், "
            "Station House Officer-க்கு எழுத்துப்பூர்வ புகார்."
        ),
        "telugu": "FIR నమోదు చేయాలని లేదా తగిన పోలీసు చర్య తీసుకోవాలని కోరుతూ, Station House Officer-కు లిఖితపూర్వక ఫిర్యాదు.",
        "kannada": "FIR ನೋಂದಾಯಿಸಲು ಅಥವಾ ಸೂಕ್ತ ಪೊಲೀಸ್ ಕ್ರಮ ಕೈಗೊಳ್ಳಲು ಕೋರುವ, Station House Officer ಗೆ ಲಿಖಿತ ದೂರು.",
        "bengali": "FIR নিবন্ধন বা উপযুক্ত পুলিশি ব্যবস্থার অনুরোধ জানিয়ে, Station House Officer-এর কাছে লিখিত অভিযোগ।",
        "malayalam": "FIR രജിസ്റ്റർ ചെയ്യാനോ ഉചിതമായ പോലീസ് നടപടിക്കോ ആവശ്യപ്പെടുന്ന, Station House Officer-ക്ക് എഴുതിയ പരാതി.",
        "marathi": "FIR नोंदवण्याची किंवा योग्य पोलीस कारवाईची विनंती करणारी, Station House Officer यांना लेखी तक्रार.",
        "gujarati": "FIR નોંધવા અથવા યોગ્ય પોલીસ કાર્યવાહી માટે વિનંતી કરતી, Station House Officer ને લેખિત ફરિયાદ.",
        "punjabi": "FIR ਦਰਜ ਕਰਨ ਜਾਂ ਢੁਕਵੀਂ ਪੁਲਿਸ ਕਾਰਵਾਈ ਦੀ ਬੇਨਤੀ ਕਰਦੀ, Station House Officer ਨੂੰ ਲਿਖਤੀ ਸ਼ਿਕਾਇਤ।",
        "odia": "FIR ପଞ୍ଜୀକରଣ କିମ୍ବା ଉପଯୁକ୍ତ ପୋଲିସ୍ କାର୍ଯ୍ୟାନୁଷ୍ଠାନ ପାଇଁ ଅନୁରୋଧ କରୁଥିବା, Station House Officer ଙ୍କୁ ଲିଖିତ ଅଭିଯୋଗ।",
        "urdu": "FIR درج کرنے یا مناسب پولیس کارروائی کی درخواست کرنے والی، Station House Officer کو تحریری شکایت۔",
    },
    "consumer_complaint": {
        "tamil": (
            "குறைபாடுள்ள பொருள், குறைபாடான சேவை அல்லது நியாயமற்ற வர்த்தக நடைமுறை தொடர்பாக "
            "Consumer Disputes Redressal Commission-க்கு புகார்."
        ),
        "telugu": (
            "లోపభూయిష్ట ఉత్పత్తి, లోపభూయిష్ట సేవ లేదా అన్యాయమైన వాణిజ్య పద్ధతి గురించి "
            "Consumer Disputes Redressal Commission-కు ఫిర్యాదు."
        ),
        "kannada": (
            "ದೋಷಪೂರಿತ ಉತ್ಪನ್ನ, ಕೊರತೆಯ ಸೇವೆ ಅಥವಾ ಅನ್ಯಾಯದ ವ್ಯಾಪಾರ ಪದ್ಧತಿ ಕುರಿತು Consumer "
            "Disputes Redressal Commission ಗೆ ದೂರು."
        ),
        "bengali": (
            "ত্রুটিপূর্ণ পণ্য, ত্রুটিপূর্ণ পরিষেবা বা অন্যায্য বাণিজ্য চর্চা সংক্রান্ত Consumer "
            "Disputes Redressal Commission-এ অভিযোগ।"
        ),
        "malayalam": (
            "കേടായ ഉൽപ്പന്നം, കുറവുള്ള സേവനം അല്ലെങ്കിൽ അന്യായമായ വ്യാപാര രീതി സംബന്ധിച്ച് "
            "Consumer Disputes Redressal Commission-ന് പരാതി."
        ),
        "marathi": (
            "सदोष उत्पादन, त्रुटीपूर्ण सेवा किंवा अनुचित व्यापार पद्धतीबाबत Consumer Disputes "
            "Redressal Commission कडे तक्रार."
        ),
        "gujarati": (
            "ખામીયુક્ત ઉત્પાદન, ખામીયુક્ત સેવા અથવા અયોગ્ય વેપાર પ્રથા અંગે Consumer Disputes "
            "Redressal Commission ને ફરિયાદ."
        ),
        "punjabi": (
            "ਨੁਕਸਦਾਰ ਉਤਪਾਦ, ਖਾਮੀ ਵਾਲੀ ਸੇਵਾ ਜਾਂ ਗਲਤ ਵਪਾਰਕ ਅਭਿਆਸ ਬਾਰੇ Consumer Disputes "
            "Redressal Commission ਨੂੰ ਸ਼ਿਕਾਇਤ।"
        ),
        "odia": (
            "ତ୍ରୁଟିଯୁକ୍ତ ଉତ୍ପାଦ, ତ୍ରୁଟିଯୁକ୍ତ ସେବା କିମ୍ବା ଅନୁଚିତ ବାଣିଜ୍ୟ ଅଭ୍ୟାସ ସମ୍ବନ୍ଧରେ Consumer "
            "Disputes Redressal Commission କୁ ଅଭିଯୋଗ।"
        ),
        "urdu": (
            "ناقص پروڈکٹ، ناقص خدمت یا غیر منصفانہ تجارتی طریقہ کار سے متعلق Consumer Disputes "
            "Redressal Commission کو شکایت۔"
        ),
    },
    "domestic_violence_complaint": {
        "tamil": (
            "குடும்ப வன்முறையைப் புகாரளித்து, Protection of Women from Domestic Violence Act, "
            "2005-ன் கீழ் பாதுகாப்பு கோரும் எழுத்துப்பூர்வ புகார்."
        ),
        "telugu": (
            "గృహ హింసను నివేదించి, Protection of Women from Domestic Violence Act, 2005 "
            "ప్రకారం రక్షణ కోరుతూ లిఖితపూర్వక ఫిర్యాదు."
        ),
        "kannada": (
            "ಗೃಹ ಹಿಂಸೆಯ ಬಗ್ಗೆ ವರದಿ ಮಾಡಿ, Protection of Women from Domestic Violence Act, 2005 "
            "ಅಡಿಯಲ್ಲಿ ರಕ್ಷಣೆ ಕೋರುವ ಲಿಖಿತ ದೂರು."
        ),
        "bengali": (
            "গার্হস্থ্য সহিংসতার প্রতিবেদন করে, Protection of Women from Domestic Violence "
            "Act, 2005 অনুযায়ী সুরক্ষা চেয়ে লিখিত অভিযোগ।"
        ),
        "malayalam": (
            "ഗാർഹിക പീഡനം റിപ്പോർട്ട് ചെയ്ത്, Protection of Women from Domestic Violence Act, "
            "2005 പ്രകാരം സംരക്ഷണം ആവശ്യപ്പെടുന്ന എഴുതിയ പരാതി."
        ),
        "marathi": (
            "घरगुती हिंसाचाराची तक्रार करून, Protection of Women from Domestic Violence Act, "
            "2005 अंतर्गत संरक्षणाची विनंती करणारी लेखी तक्रार."
        ),
        "gujarati": (
            "ઘરેલુ હિંસાની જાણ કરી, Protection of Women from Domestic Violence Act, 2005 હેઠળ "
            "સુરક્ષા માંગતી લેખિત ફરિયાદ."
        ),
        "punjabi": (
            "ਘਰੇਲੂ ਹਿੰਸਾ ਦੀ ਰਿਪੋਰਟ ਕਰਦੀ ਅਤੇ Protection of Women from Domestic Violence Act, "
            "2005 ਤਹਿਤ ਸੁਰੱਖਿਆ ਮੰਗਦੀ ਲਿਖਤੀ ਸ਼ਿਕਾਇਤ।"
        ),
        "odia": (
            "ଘରୋଇ ହିଂସାର ରିପୋର୍ଟ କରି, Protection of Women from Domestic Violence Act, 2005 "
            "ଅଧୀନରେ ସୁରକ୍ଷା ମାଗୁଥିବା ଲିଖିତ ଅଭିଯୋଗ।"
        ),
        "urdu": (
            "گھریلو تشدد کی اطلاع دیتی اور Protection of Women from Domestic Violence Act, "
            "2005 کے تحت تحفظ مانگتی تحریری شکایت۔"
        ),
    },
    "service_agreement": {
        "tamil": (
            "சேவை வழங்குநருக்கும் வாடிக்கையாளருக்கும் இடையேயான ஒப்பந்தம், சேவைகளின் நோக்கம், "
            "கட்டணம் மற்றும் நிபந்தனைகளை நிர்ணயிக்கிறது."
        ),
        "telugu": "సేవా ప్రదాత మరియు క్లయింట్ మధ్య ఒప్పందం, సేవల పరిధి, రుసుము మరియు నిబంధనలను నిర్ణయిస్తుంది.",
        "kannada": "ಸೇವಾ ಪೂರೈಕೆದಾರ ಮತ್ತು ಗ್ರಾಹಕರ ನಡುವಿನ ಒಪ್ಪಂದ, ಸೇವೆಗಳ ವ್ಯಾಪ್ತಿ, ಶುಲ್ಕ ಮತ್ತು ನಿಯಮಗಳನ್ನು ನಿಗದಿಪಡಿಸುತ್ತದೆ.",
        "bengali": "পরিষেবা প্রদানকারী ও ক্লায়েন্টের মধ্যে চুক্তি, যা পরিষেবার পরিধি, ফি এবং শর্তাবলী নির্ধারণ করে।",
        "malayalam": "സേവന ദാതാവും ക്ലയന്റും തമ്മിലുള്ള കരാർ, സേവനങ്ങളുടെ വ്യാപ്തി, ഫീസ്, നിബന്ധനകൾ എന്നിവ നിശ്ചയിക്കുന്നു.",
        "marathi": "सेवा प्रदाता आणि क्लायंट यांच्यातील करार, जो सेवांची व्याप्ती, शुल्क आणि अटी निश्चित करतो.",
        "gujarati": "સેવા પ્રદાતા અને ક્લાયન્ટ વચ્ચેનો કરાર, જે સેવાઓનો વ્યાપ, ફી અને શરતો નક્કી કરે છે.",
        "punjabi": "ਸੇਵਾ ਪ੍ਰਦਾਤਾ ਅਤੇ ਗਾਹਕ ਵਿਚਕਾਰ ਸਮਝੌਤਾ, ਜੋ ਸੇਵਾਵਾਂ ਦਾ ਦਾਇਰਾ, ਫੀਸ ਅਤੇ ਸ਼ਰਤਾਂ ਤੈਅ ਕਰਦਾ ਹੈ।",
        "odia": "ସେବା ପ୍ରଦାନକାରୀ ଏବଂ ଗ୍ରାହକଙ୍କ ମଧ୍ୟରେ ଚୁକ୍ତି, ଯାହା ସେବାର ପରିସର, ଶୁଳ୍କ ଏବଂ ସର୍ତ୍ତାବଳୀ ନିର୍ଦ୍ଧାରଣ କରେ।",
        "urdu": "سروس فراہم کنندہ اور کلائنٹ کے درمیان معاہدہ، جو خدمات کا دائرہ، فیس اور شرائط طے کرتا ہے۔",
    },
    "recovery_notice": {
        "tamil": "நிலுவையில் உள்ள கடன், தனிப்பட்ட கடன் அல்லது செலுத்தப்படாத நிலுவைத் தொகையை வசூலிக்கக் கோரும் அறிவிப்பு.",
        "telugu": "బకాయి ఉన్న రుణం, వ్యక్తిగత అప్పు లేదా చెల్లించని బకాయిల వసూలు కోసం డిమాండ్ నోటీసు.",
        "kannada": "ಬಾಕಿ ಇರುವ ಸಾಲ, ವೈಯಕ್ತಿಕ ಸಾಲ ಅಥವಾ ಪಾವತಿಸದ ಬಾಕಿಯ ವಸೂಲಾತಿಗಾಗಿ ಬೇಡಿಕೆ ನೋಟಿಸ್.",
        "bengali": "বকেয়া ঋণ, ব্যক্তিগত ঋণ বা অপরিশোধিত বকেয়া আদায়ের জন্য দাবি নোটিশ।",
        "malayalam": "കുടിശ്ശികയുള്ള വായ്പ, വ്യക്തിഗത കടം അല്ലെങ്കിൽ അടയ്ക്കാത്ത കുടിശ്ശിക തിരിച്ചുപിടിക്കാനുള്ള ഡിമാൻഡ് നോട്ടീസ്.",
        "marathi": "थकीत कर्ज, वैयक्तिक कर्ज किंवा न भरलेल्या रकमेच्या वसुलीसाठी मागणी नोटीस.",
        "gujarati": "બાકી લોન, અંગત દેવું અથવા ન ચૂકવેલ લેણાંની વસૂલાત માટે માંગ નોટિસ.",
        "punjabi": "ਬਕਾਇਆ ਕਰਜ਼ਾ, ਨਿੱਜੀ ਕਰਜ਼ਾ ਜਾਂ ਅਦਾਇਗੀ ਨਾ ਕੀਤੇ ਬਕਾਏ ਦੀ ਵਸੂਲੀ ਲਈ ਮੰਗ ਨੋਟਿਸ।",
        "odia": "ବକେୟା ଋଣ, ବ୍ୟକ୍ତିଗତ ଋଣ କିମ୍ବା ଅପରିଶୋଧିତ ବକେୟା ଆଦାୟ ପାଇଁ ଦାବି ବିଜ୍ଞପ୍ତି।",
        "urdu": "بقایا قرض، ذاتی قرض یا ادا نہ کیے گئے واجبات کی وصولی کے لیے مطالبہ نوٹس۔",
    },
}


def localized_description(template: DraftTemplateDefinition, language: str) -> str:
    """The one-line description to show for `template` in `language`.

    Falls back to `template.description` (the English original) for hindi
    (which has no separate translated description field on
    `DraftTemplateDefinition` -- unlike `hindi_name`, this was never
    authored per-template), any language not covered above, or any
    template not in `_DESCRIPTION_TRANSLATIONS` (47 of the 55 -- see this
    module's docstring). Never fabricates a translation.
    """
    translated = _DESCRIPTION_TRANSLATIONS.get(template.draft_id, {}).get(language)
    return translated or template.description
