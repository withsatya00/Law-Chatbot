"""Part 53 "Fallback Draft Localization" (title localization): the document
title shown at the top of an exported PDF/DOCX/TXT (e.g. "LEGAL NOTICE" ->
"सामान्य कानूनी नोटिस"), applied at export time regardless of whether the
draft's `sections` came from the LLM or the deterministic fallback -- the
title itself is template metadata, not generated content, so both paths
share this one fix.

The `template_name`/`draft["template_name"]` stored on the draft record and
returned by the API stays the canonical English name everywhere else (draft
history, chat wrapper text, in-chat edit-command matching) -- only the
export-time display title is localized here, matching how
`heading_translations.py` only localizes section headings for display,
never the underlying data.

Hindi already has a translated name on every template definition
(`DraftTemplateDefinition.hindi_name`, authored per template's YAML) --
reused here rather than duplicated. Tamil/Telugu/Kannada/Bengali are added
below, keyed by `draft_id` (a small, closed set of 11 templates) since a
template title is a whole translated phrase, not a generic reusable word.
"""

from app.drafting.eighth_schedule_names import NAMES_BY_DRAFT_ID as _EIGHTH_SCHEDULE_NAMES
from app.drafting.templates.base import DraftTemplateDefinition

_TITLE_TRANSLATIONS: dict[str, dict[str, str]] = {
    "affidavit": {
        "tamil": "பொது சத்தியக்கடதாசி", "telugu": "సాధారణ అఫిడవిట్",
        "kannada": "ಸಾಮಾನ್ಯ ಪ್ರಮಾಣಪತ್ರ", "bengali": "সাধারণ হলফনামা",
        "malayalam": "പൊതു സത്യപ്രസ്താവന", "marathi": "सर्वसाधारण प्रतिज्ञापत्र", "gujarati": "સામાન્ય સોગંદનામું", "punjabi": "ਆਮ ਹਲਫ਼ਨਾਮਾ", "odia": "ସାଧାରଣ ଶପଥପତ୍ର", "urdu": "عام حلف نامہ",
    },
    "cheque_bounce_notice": {
        "tamil": "காசோலை மறுப்பு அறிவிப்பு", "telugu": "చెక్ బౌన్స్ నోటీసు",
        "kannada": "ಚೆಕ್ ಬೌನ್ಸ್ ನೋಟಿಸ್", "bengali": "চেক প্রত্যাখ্যান নোটিশ",
        "malayalam": "ചെക്ക് മടങ്ങൽ അറിയിപ്പ്", "marathi": "धनादेश परत सूचना", "gujarati": "ચેક બાઉન્સ સૂચના", "punjabi": "ਚੈੱਕ ਬਾਊਂਸ ਨੋਟਿਸ", "odia": "ଚେକ୍ ବାଉନ୍ସ ବିଜ୍ଞପ୍ତି", "urdu": "چیک باؤنس نوٹس",
    },
    "consumer_complaint": {
        "tamil": "நுகர்வோர் புகார்", "telugu": "వినియోగదారు ఫిర్యాదు",
        "kannada": "ಗ್ರಾಹಕ ದೂರು", "bengali": "ভোক্তা অভিযোগ",
        "malayalam": "ഉപഭോക്തൃ പരാതി", "marathi": "ग्राहक तक्रार", "gujarati": "ગ્રાહક ફરિયાદ", "punjabi": "ਖਪਤਕਾਰ ਸ਼ਿਕਾਇਤ", "odia": "ଗ୍ରାହକ ଅଭିଯୋଗ", "urdu": "صارف کی شکایت",
    },
    "cyber_crime_complaint": {
        "tamil": "இணையவழி குற்றப் புகார்", "telugu": "సైబర్ నేర ఫిర్యాదు",
        "kannada": "ಸೈಬರ್ ಅಪರಾಧ ದೂರು", "bengali": "সাইবার অপরাধ অভিযোগ",
        "malayalam": "സൈബർ ക്രൈം പരാതി", "marathi": "सायबर गुन्हा तक्रार", "gujarati": "સાયબર ક્રાઇમ ફરિયાદ", "punjabi": "ਸਾਈਬਰ ਅਪਰਾਧ ਸ਼ਿਕਾਇਤ", "odia": "ସାଇବର କ୍ରାଇମ୍ ଅଭିଯୋଗ", "urdu": "سائبر کرائم شکایت",
    },
    "employment_notice": {
        "tamil": "வேலை நிலுவைத் தொகை / தவறான பணிநீக்க அறிவிப்பு",
        "telugu": "ఉద్యోగ బకాయిలు / తప్పుడు తొలగింపు నోటీసు",
        "kannada": "ಉದ್ಯೋಗ ಬಾಕಿ / ತಪ್ಪಾದ ವಜಾ ನೋಟಿಸ್",
        "bengali": "চাকরির বকেয়া / বেআইনি বরখাস্ত নোটিশ",
        "malayalam": "തൊഴിൽ കുടിശ്ശിക / തെറ്റായ പിരിച്ചുവിടൽ അറിയിപ്പ്", "marathi": "नोकरी थकबाकी / अन्यायकारक बडतर्फी सूचना", "gujarati": "રોજગાર બાકી / ખોટી બરતરફી સૂચના", "punjabi": "ਰੁਜ਼ਗਾਰ ਬਕਾਇਆ / ਗਲਤ ਬਰਖਾਸਤਗੀ ਨੋਟਿਸ", "odia": "ନିଯୁକ୍ତି ବକେୟା / ଅନ୍ୟାୟ ବରଖାସ୍ତ ବିଜ୍ଞପ୍ତି", "urdu": "ملازمت واجبات / غلط برخاستگی نوٹس",
    },
    "legal_notice": {
        "tamil": "பொது சட்ட அறிவிப்பு", "telugu": "సాధారణ న్యాయ నోటీసు",
        "kannada": "ಸಾಮಾನ್ಯ ಕಾನೂನು ನೋಟಿಸ್", "bengali": "সাধারণ আইনি নোটিশ",
        "malayalam": "പൊതു നിയമ അറിയിപ്പ്", "marathi": "सर्वसाधारण कायदेशीर सूचना", "gujarati": "સામાન્ય કાનૂની સૂચના", "punjabi": "ਆਮ ਕਾਨੂੰਨੀ ਨੋਟਿਸ", "odia": "ସାଧାରଣ ଆଇନଗତ ବିଜ୍ଞପ୍ତି", "urdu": "عام قانونی نوٹس",
    },
    "police_complaint": {
        "tamil": "காவல்துறை புகார் (தாணா பொறுப்பாளருக்கு விண்ணப்பம்)",
        "telugu": "పోలీసు ఫిర్యాదు (స్టేషన్ హౌస్ ఆఫీసర్‌కు దరఖాస్తు)",
        "kannada": "ಪೊಲೀಸ್ ದೂರು (ಠಾಣಾ ಉಸ್ತುವಾರಿಗೆ ಅರ್ಜಿ)",
        "bengali": "পুলিশ অভিযোগ (থানা ইনচার্জের কাছে আবেদন)",
        "malayalam": "പോലീസ് പരാതി (സ്റ്റേഷൻ ഹൗസ് ഓഫീസർക്ക് അപേക്ഷ)", "marathi": "पोलीस तक्रार (ठाणे प्रभारींना अर्ज)", "gujarati": "પોલીસ ફરિયાદ (સ્ટેશન હાઉસ ઓફિસરને અરજી)", "punjabi": "ਪੁਲਿਸ ਸ਼ਿਕਾਇਤ (ਸਟੇਸ਼ਨ ਹਾਊਸ ਅਫ਼ਸਰ ਨੂੰ ਬਿਨੈ)", "odia": "ପୋଲିସ୍ ଅଭିଯୋଗ (ଷ୍ଟେସନ୍ ହାଉସ୍ ଅଫିସରଙ୍କୁ ଆବେଦନ)", "urdu": "پولیس شکایت (اسٹیشن ہاؤس افسر کو درخواست)",
    },
    "recovery_notice": {
        "tamil": "பணம் திரும்பப் பெறுதல் அறிவிப்பு", "telugu": "డబ్బు రికవరీ నోటీసు",
        "kannada": "ಹಣ ವಸೂಲಿ ನೋಟಿಸ್", "bengali": "অর্থ আদায় নোটিশ",
        "malayalam": "പണം തിരിച്ചുപിടിക്കൽ അറിയിപ്പ്", "marathi": "रक्कम वसुली सूचना", "gujarati": "રકમ વસૂલાત સૂચના", "punjabi": "ਰਕਮ ਵਸੂਲੀ ਨੋਟਿਸ", "odia": "ଅର୍ଥ ଆଦାୟ ବିଜ୍ଞପ୍ତି", "urdu": "رقم کی وصولی کا نوٹس",
    },
    "rent_notice": {
        "tamil": "வாடகை / பாதுகாப்பு வைப்புத்தொகை திரும்பப் பெறுதல் அறிவிப்பு",
        "telugu": "అద్దె / సెక్యూరిటీ డిపాజిట్ రికవరీ నోటీసు",
        "kannada": "ಬಾಡಿಗೆ / ಭದ್ರತಾ ಠೇವಣಿ ವಸೂಲಿ ನೋಟಿಸ್",
        "bengali": "ভাড়া / জামানত ফেরত আদায় নোটিশ",
        "malayalam": "വാടക / സെക്യൂരിറ്റി ഡെപ്പോസിറ്റ് തിരിച്ചുപിടിക്കൽ അറിയിപ്പ്", "marathi": "भाडे / सुरक्षा ठेव वसुली सूचना", "gujarati": "ભાડું / સિક્યોરિટી ડિપોઝિટ વસૂલાત સૂચના", "punjabi": "ਕਿਰਾਇਆ / ਸੁਰੱਖਿਆ ਜਮ੍ਹਾਂ ਵਸੂਲੀ ਨੋਟਿਸ", "odia": "ଭଡା / ସୁରକ୍ଷା ଜମା ଆଦାୟ ବିଜ୍ଞପ୍ତି", "urdu": "کرایہ / سیکیورٹی ڈپازٹ کی وصولی کا نوٹس",
    },
    "rti_application": {
        # "RTI" (Right to Information Act) is an official acronym -- kept
        # untranslated, only "application" is localized, same convention as
        # preserving Act names in the main drafting prompt.
        "tamil": "RTI விண்ணப்பம்", "telugu": "RTI దరఖాస్తు",
        "kannada": "RTI ಅರ್ಜಿ", "bengali": "RTI আবেদন",
        "malayalam": "RTI അപേക്ഷ", "marathi": "RTI अर्ज", "gujarati": "RTI અરજી", "punjabi": "RTI ਬਿਨੈ", "odia": "RTI ଆବେଦନ", "urdu": "RTI درخواست",
    },
    "sp_complaint": {
        "tamil": "காவல் கண்காணிப்பாளருக்கு புகார்", "telugu": "పోలీసు సూపరింటెండెంట్‌కు ఫిర్యాదు",
        "kannada": "ಪೊಲೀಸ್ ಅಧೀಕ್ಷಕರಿಗೆ ದೂರು", "bengali": "পুলিশ সুপারের কাছে অভিযোগ",
        "malayalam": "പോലീസ് സൂപ്രണ്ടിന് പരാതി", "marathi": "पोलीस अधीक्षकांना तक्रार", "gujarati": "પોલીસ સુપ્રિન્ટેન્ડન્ટને ફરિયાદ", "punjabi": "ਪੁਲਿਸ ਸੁਪਰਡੈਂਟ ਨੂੰ ਸ਼ਿਕਾਇਤ", "odia": "ପୋଲିସ୍ ସୁପରିଣ୍ଟେଣ୍ଡେଣ୍ଟଙ୍କୁ ଅଭିଯୋଗ", "urdu": "پولیس سپرنٹنڈنٹ کو شکایت",
    },
}


# The 43 templates below never got a `_TITLE_TRANSLATIONS`/
# `_SHORT_TRIGGER_TRANSLATIONS` entry at all (only ~11 "flagship" templates
# did, across the several rounds of multilingual work above) -- confirmed
# live: naming ANY of these 43 templates in Tamil/Telugu/Kannada/Bengali/
# Malayalam/Marathi/Gujarati/Punjabi/Odia/Urdu, in any phrasing, could never
# match (`DraftIntentDetector._score_templates` had nothing to score against
# for that language), regardless of whether a drafting verb was present. One
# shared table, not duplicated into both tables above: none of these 43
# names has a separate long/short form worth distinguishing (unlike e.g.
# "police_complaint"'s full "... Application to SHO" vs. short "Police
# Complaint"), so the same phrase serves both the export title and the
# short conversational/matching form. `all_localized_names`/`localized_title`/
# `localized_document_name` below fall back to this table for any
# `draft_id` not already covered by the two tables above.
_ADDITIONAL_TEMPLATE_NAME_TRANSLATIONS: dict[str, dict[str, str]] = {
    "account_closure_application": {
        "tamil": "கணக்கு மூடல் விண்ணப்பம்", "telugu": "ఖాతా మూసివేత దరఖాస్తు",
        "kannada": "ಖಾತೆ ಮುಚ್ಚುವಿಕೆ ಅರ್ಜಿ", "bengali": "অ্যাকাউন্ট বন্ধ আবেদন",
        "malayalam": "അക്കൗണ്ട് ക്ലോഷർ അപേക്ഷ", "marathi": "खाते बंद अर्ज", "gujarati": "ખાતું બંધ કરવા અરજી", "punjabi": "ਖਾਤਾ ਬੰਦ ਕਰਨ ਦੀ ਬਿਨੈ", "odia": "ଖାତା ବନ୍ଦ ଆବେଦନ", "urdu": "اکاؤنٹ بندش کی درخواست",
    },
    "address_affidavit": {
        "tamil": "முகவரி சத்தியக்கடதாசி", "telugu": "చిరునామా అఫిడవిట్",
        "kannada": "ವಿಳಾಸ ಪ್ರಮಾಣಪತ್ರ", "bengali": "ঠিকানা হলফনামা",
        "malayalam": "വിലാസ സത്യപ്രസ്താവന", "marathi": "पत्ता प्रतिज्ञापत्र", "gujarati": "સરનામું સોગંદનામું", "punjabi": "ਪਤਾ ਹਲਫ਼ਨਾਮਾ", "odia": "ଠିକଣା ଶପଥପତ୍ର", "urdu": "پتہ حلف نامہ",
    },
    "bank_fraud_complaint": {
        "tamil": "வங்கி மோசடி புகார்", "telugu": "బ్యాంకు మోసం ఫిర్యాదు",
        "kannada": "ಬ್ಯಾಂಕ್ ವಂಚನೆ ದೂರು", "bengali": "ব্যাংক জালিয়াতি অভিযোগ",
        "malayalam": "ബാങ്ക് തട്ടിപ്പ് പരാതി", "marathi": "बँक फसवणूक तक्रार", "gujarati": "બેંક છેતરપિંડી ફરિયાદ", "punjabi": "ਬੈਂਕ ਧੋਖਾਧੜੀ ਸ਼ਿਕਾਇਤ", "odia": "ବ୍ୟାଙ୍କ ଠକେଇ ଅଭିଯୋଗ", "urdu": "بینک فراڈ شکایت",
    },
    "birth_certificate_application": {
        "tamil": "பிறப்புச் சான்று விண்ணப்பம்", "telugu": "జనన ధృవీకరణ పత్రం దరఖాస్తు",
        "kannada": "ಜನನ ಪ್ರಮಾಣಪತ್ರ ಅರ್ಜಿ", "bengali": "জন্ম সনদ আবেদন",
        "malayalam": "ജനന സർട്ടിഫിക്കറ്റ് അപേക്ഷ", "marathi": "जन्म दाखला अर्ज", "gujarati": "જન્મ પ્રમાણપત્ર અરજી", "punjabi": "ਜਨਮ ਸਰਟੀਫਿਕੇਟ ਬਿਨੈ", "odia": "ଜନ୍ମ ପ୍ରମାଣପତ୍ର ଆବେଦନ", "urdu": "پیدائش سرٹیفکیٹ درخواست",
    },
    "bonafide_certificate_application": {
        "tamil": "நம்பகத்தன்மைச் சான்று விண்ணப்பம்", "telugu": "బోనఫైడ్ సర్టిఫికేట్ దరఖాస్తు",
        "kannada": "ಬೊನಫೈಡ್ ಪ್ರಮಾಣಪತ್ರ ಅರ್ಜಿ", "bengali": "বোনাফাইড সনদ আবেদন",
        "malayalam": "ബോണഫൈഡ് സർട്ടിഫിക്കറ്റ് അപേക്ഷ", "marathi": "बोनाफाईड प्रमाणपत्र अर्ज", "gujarati": "બોનાફાઇડ પ્રમાણપત્ર અરજી", "punjabi": "ਬੋਨਾਫਾਈਡ ਸਰਟੀਫਿਕੇਟ ਬਿਨੈ", "odia": "ବୋନାଫାଇଡ୍ ପ୍ରମାଣପତ୍ର ଆବେଦନ", "urdu": "بونا فائیڈ سرٹیفکیٹ درخواست",
    },
    "business_proposal": {
        "tamil": "வணிக முன்மொழிவு", "telugu": "వ్యాపార ప్రతిపాదన",
        "kannada": "ವ್ಯವಹಾರ ಪ್ರಸ್ತಾವನೆ", "bengali": "ব্যবসায়িক প্রস্তাব",
        "malayalam": "ബിസിനസ്സ് നിർദ്ദേശം", "marathi": "व्यवसाय प्रस्ताव", "gujarati": "વ્યવસાય દરખાસ્ત", "punjabi": "ਕਾਰੋਬਾਰੀ ਪ੍ਰਸਤਾਵ", "odia": "ବ୍ୟବସାୟ ପ୍ରସ୍ତାବ", "urdu": "کاروباری تجویز",
    },
    "consumer_notice": {
        "tamil": "நுகர்வோர் அறிவிப்பு", "telugu": "వినియోగదారు నోటీసు",
        "kannada": "ಗ್ರಾಹಕ ನೋಟಿಸ್", "bengali": "ভোক্তা নোটিশ",
        "malayalam": "ഉപഭോക്തൃ അറിയിപ്പ്", "marathi": "ग्राहक सूचना", "gujarati": "ગ્રાહક સૂચના", "punjabi": "ਖਪਤਕਾਰ ਨੋਟਿਸ", "odia": "ଗ୍ରାହକ ବିଜ୍ଞପ୍ତି", "urdu": "صارف نوٹس",
    },
    "contract_breach_notice": {
        "tamil": "ஒப்பந்த மீறல் அறிவிப்பு", "telugu": "ఒప్పంద ఉల్లంఘన నోటీసు",
        "kannada": "ಒಪ್ಪಂದ ಉಲ್ಲಂಘನೆ ನೋಟಿಸ್", "bengali": "চুক্তি লঙ্ঘন নোটিশ",
        "malayalam": "കരാർ ലംഘന അറിയിപ്പ്", "marathi": "करार भंग सूचना", "gujarati": "કરાર ભંગ સૂચના", "punjabi": "ਇਕਰਾਰਨਾਮਾ ਉਲੰਘਣਾ ਨੋਟਿਸ", "odia": "ଚୁକ୍ତି ଉଲ୍ଲଂଘନ ବିଜ୍ଞପ୍ତି", "urdu": "معاہدہ خلاف ورزی نوٹس",
    },
    "defamation_notice": {
        "tamil": "அவதூறு அறிவிப்பு", "telugu": "పరువు నష్టం నోటీసు",
        "kannada": "ಮಾನಹಾನಿ ನೋಟಿಸ್", "bengali": "মানহানি নোটিশ",
        "malayalam": "അപകീർത്തി അറിയിപ്പ്", "marathi": "बदनामी सूचना", "gujarati": "બદનક્ષી સૂચના", "punjabi": "ਮਾਣਹਾਨੀ ਨੋਟਿਸ", "odia": "ମାନହାନି ବିଜ୍ଞପ୍ତି", "urdu": "ہتک عزت نوٹس",
    },
    "demand_notice": {
        "tamil": "பொது கோரிக்கை அறிவிப்பு", "telugu": "సాధారణ డిమాండ్ నోటీసు",
        "kannada": "ಸಾಮಾನ್ಯ ಬೇಡಿಕೆ ನೋಟಿಸ್", "bengali": "সাধারণ দাবি নোটিশ",
        "malayalam": "പൊതു ഡിമാൻഡ് അറിയിപ്പ്", "marathi": "सर्वसाधारण मागणी सूचना", "gujarati": "સામાન્ય માંગ સૂચના", "punjabi": "ਆਮ ਮੰਗ ਨੋਟਿਸ", "odia": "ସାଧାରଣ ଦାବି ବିଜ୍ଞପ୍ତି", "urdu": "عام مطالبہ نوٹس",
    },
    "domestic_violence_complaint": {
        "tamil": "குடும்ப வன்முறை புகார்", "telugu": "గృహ హింస ఫిర్యాదు",
        "kannada": "ಗೃಹ ಹಿಂಸೆ ದೂರು", "bengali": "পারিবারিক নির্যাতন অভিযোগ",
        "malayalam": "ഗാർഹിക പീഡന പരാതി", "marathi": "घरगुती हिंसाचार तक्रार", "gujarati": "ઘરેલુ હિંસા ફરિયાદ", "punjabi": "ਘਰੇਲੂ ਹਿੰਸਾ ਸ਼ਿਕਾਇਤ", "odia": "ଗାର୍ହସ୍ଥ୍ୟ ହିଂସା ଅଭିଯୋଗ", "urdu": "گھریلو تشدد شکایت",
    },
    "ecommerce_complaint": {
        "tamil": "மின்வணிக புகார்", "telugu": "ఈ-కామర్స్ ఫిర్యాదు",
        "kannada": "ಇ-ಕಾಮರ್ಸ್ ದೂರು", "bengali": "ই-কমার্স অভিযোগ",
        "malayalam": "ഇ-കൊമേഴ്‌സ് പരാതി", "marathi": "ई-कॉमर्स तक्रार", "gujarati": "ઈ-કોમર્સ ફરિયાદ", "punjabi": "ਈ-ਕਾਮਰਸ ਸ਼ਿਕਾਇਤ", "odia": "ଇ-କମର୍ସ ଅଭିଯୋଗ", "urdu": "ای کامرس شکایت",
    },
    "education_leave_application": {
        "tamil": "பள்ளி / கல்லூரி விடுப்பு விண்ணப்பம்", "telugu": "పాఠశాల / కళాశాల సెలవు దరఖాస్తు",
        "kannada": "ಶಾಲೆ / ಕಾಲೇಜು ರಜೆ ಅರ್ಜಿ", "bengali": "স্কুল / কলেজ ছুটির আবেদন",
        "malayalam": "സ്കൂൾ / കോളേജ് അവധി അപേക്ഷ", "marathi": "शाळा / महाविद्यालय रजा अर्ज", "gujarati": "શાળા / કોલેજ રજા અરજી", "punjabi": "ਸਕੂਲ / ਕਾਲਜ ਛੁੱਟੀ ਬਿਨੈ", "odia": "ବିଦ୍ୟାଳୟ / ମହାବିଦ୍ୟାଳୟ ଛୁଟି ଆବେଦନ", "urdu": "اسکول / کالج چھٹی درخواست",
    },
    "employment_leave_application": {
        "tamil": "பணியாளர் விடுப்பு விண்ணப்பம்", "telugu": "ఉద్యోగి సెలవు దరఖాస్తు",
        "kannada": "ನೌಕರರ ರಜೆ ಅರ್ಜಿ", "bengali": "কর্মচারী ছুটির আবেদন",
        "malayalam": "ജീവനക്കാരൻ അവധി അപേക്ഷ", "marathi": "कर्मचारी रजा अर्ज", "gujarati": "કર્મચારી રજા અરજી", "punjabi": "ਕਰਮਚਾਰੀ ਛੁੱਟੀ ਬਿਨੈ", "odia": "କର୍ମଚାରୀ ଛୁଟି ଆବେଦନ", "urdu": "ملازم چھٹی درخواست",
    },
    "employment_noc_request": {
        "tamil": "வேலைவாய்ப்பு ஆட்சேபனையின்மைச் சான்று கோரிக்கை", "telugu": "ఉద్యోగ NOC అభ్యర్థన",
        "kannada": "ಉದ್ಯೋಗ NOC ವಿನಂತಿ", "bengali": "চাকরি এনওসি অনুরোধ",
        "malayalam": "തൊഴിൽ NOC അഭ്യർത്ഥന", "marathi": "नोकरी एनओसी विनंती", "gujarati": "રોજગાર એનઓસી વિનંતી", "punjabi": "ਰੁਜ਼ਗਾਰ ਐਨਓਸੀ ਬੇਨਤੀ", "odia": "ନିଯୁକ୍ତି NOC ଅନୁରୋଧ", "urdu": "ملازمت این او سی درخواست",
    },
    "experience_certificate_request": {
        "tamil": "அனுபவச் சான்று கோரிக்கை", "telugu": "అనుభవ ధృవీకరణ పత్రం అభ్యర్థన",
        "kannada": "ಅನುಭವ ಪ್ರಮಾಣಪತ್ರ ವಿನಂತಿ", "bengali": "অভিজ্ঞতা সনদ অনুরোধ",
        "malayalam": "പരിചയ സർട്ടിഫിക്കറ്റ് അഭ്യർത്ഥന", "marathi": "अनुभव प्रमाणपत्र विनंती", "gujarati": "અનુભવ પ્રમાણપત્ર વિનંતી", "punjabi": "ਤਜਰਬਾ ਸਰਟੀਫਿਕੇਟ ਬੇਨਤੀ", "odia": "ଅଭିଜ୍ଞତା ପ୍ରମାଣପତ୍ର ଅନୁରୋଧ", "urdu": "تجربہ سرٹیفکیٹ درخواست",
    },
    "identity_affidavit": {
        "tamil": "அடையாள சத்தியக்கடதாசி", "telugu": "గుర్తింపు అఫిడవిట్",
        "kannada": "ಗುರುತಿನ ಪ್ರಮಾಣಪತ್ರ", "bengali": "পরিচয় হলফনামা",
        "malayalam": "തിരിച്ചറിയൽ സത്യപ്രസ്താവന", "marathi": "ओळख प्रतिज्ञापत्र", "gujarati": "ઓળખ સોગંદનામું", "punjabi": "ਪਛਾਣ ਹਲਫ਼ਨਾਮਾ", "odia": "ପରିଚୟ ଶପଥପତ୍ର", "urdu": "شناخت حلف نامہ",
    },
    "income_certificate_application": {
        "tamil": "வருமானச் சான்று விண்ணப்பம்", "telugu": "ఆదాయ ధృవీకరణ పత్రం దరఖాస్తు",
        "kannada": "ಆದಾಯ ಪ್ರಮಾಣಪತ್ರ ಅರ್ಜಿ", "bengali": "আয়ের সনদ আবেদন",
        "malayalam": "വരുമാന സർട്ടിഫിക്കറ്റ് അപേക്ഷ", "marathi": "उत्पन्न दाखला अर्ज", "gujarati": "આવક પ્રમાણપત્ર અરજી", "punjabi": "ਆਮਦਨ ਸਰਟੀਫਿਕੇਟ ਬਿਨੈ", "odia": "ଆୟ ପ୍ରମାଣପତ୍ର ଆବେଦନ", "urdu": "آمدنی سرٹیفکیٹ درخواست",
    },
    "job_application": {
        "tamil": "வேலை விண்ணப்பம்", "telugu": "ఉద్యోగ దరఖాస్తు",
        "kannada": "ಉದ್ಯೋಗ ಅರ್ಜಿ", "bengali": "চাকরির আবেদন",
        "malayalam": "ജോലി അപേക്ഷ", "marathi": "नोकरी अर्ज", "gujarati": "નોકરી અરજી", "punjabi": "ਨੌਕਰੀ ਬਿਨੈ", "odia": "ଚାକିରି ଆବେଦନ", "urdu": "ملازمت درخواست",
    },
    "kyc_update_application": {
        "tamil": "KYC புதுப்பிப்பு விண்ணப்பம்", "telugu": "KYC అప్‌డేట్ దరఖాస్తు",
        "kannada": "KYC ನವೀಕರಣ ಅರ್ಜಿ", "bengali": "কেওয়াইসি আপডেট আবেদন",
        "malayalam": "KYC അപ്ഡേറ്റ് അപേക്ഷ", "marathi": "केवायसी अद्यतन अर्ज", "gujarati": "કેવાયસી અપડેટ અરજી", "punjabi": "ਕੇਵਾਈਸੀ ਅੱਪਡੇਟ ਬਿਨੈ", "odia": "KYC ଅଦ୍ୟତନ ଆବେଦନ", "urdu": "کے وائی سی اپڈیٹ درخواست",
    },
    "lost_document_affidavit": {
        "tamil": "தொலைந்த ஆவணச் சத்தியக்கடதாசி", "telugu": "పోగొట్టుకున్న పత్రం అఫిడవిట్",
        "kannada": "ಕಳೆದುಹೋದ ದಾಖಲೆ ಪ್ರಮಾಣಪತ್ರ", "bengali": "হারানো নথি হলফনামা",
        "malayalam": "നഷ്ടപ്പെട്ട രേഖ സത്യപ്രസ്താവന", "marathi": "हरवलेले कागदपत्र प्रतिज्ञापत्र", "gujarati": "ખોવાયેલ દસ્તાવેજ સોગંદનામું", "punjabi": "ਗੁੰਮ ਦਸਤਾਵੇਜ਼ ਹਲਫ਼ਨਾਮਾ", "odia": "ହଜିଯାଇଥିବା ଡକ୍ୟୁମେଣ୍ଟ ଶପଥପତ୍ର", "urdu": "گمشدہ دستاویز حلف نامہ",
    },
    "missing_person_report": {
        "tamil": "காணாமல் போனவர் அறிக்கை", "telugu": "తప్పిపోయిన వ్యక్తి నివేదిక",
        "kannada": "ಕಾಣೆಯಾದ ವ್ಯಕ್ತಿ ವರದಿ", "bengali": "নিখোঁজ ব্যক্তি রিপোর্ট",
        "malayalam": "കാണാതായ വ്യക്തി റിപ്പോർട്ട്", "marathi": "बेपत्ता व्यक्ती अहवाल", "gujarati": "ગુમ વ્યક્તિ અહેવાલ", "punjabi": "ਲਾਪਤਾ ਵਿਅਕਤੀ ਰਿਪੋਰਟ", "odia": "ନିଖୋଜ ବ୍ୟକ୍ତି ରିପୋର୍ଟ", "urdu": "لاپتہ شخص رپورٹ",
    },
    "mobile_theft_complaint": {
        "tamil": "மொபைல் திருட்டு புகார்", "telugu": "మొబైల్ దొంగతనం ఫిర్యాదు",
        "kannada": "ಮೊಬೈಲ್ ಕಳ್ಳತನ ದೂರು", "bengali": "মোবাইল চুরি অভিযোগ",
        "malayalam": "മൊബൈൽ മോഷണം പരാതി", "marathi": "मोबाईल चोरी तक्रार", "gujarati": "મોબાઈલ ચોરી ફરિયાદ", "punjabi": "ਮੋਬਾਈਲ ਚੋਰੀ ਸ਼ਿਕਾਇਤ", "odia": "ମୋବାଇଲ ଚୋରି ଅଭିଯୋଗ", "urdu": "موبائل چوری شکایت",
    },
    "mou_agreement": {
        "tamil": "புரிந்துணர்வு ஒப்பந்தம் (MOU)", "telugu": "అవగాహన ఒప్పందం (MOU)",
        "kannada": "ತಿಳುವಳಿಕಾ ಒಪ್ಪಂದ (MOU)", "bengali": "সমঝোতা স্মারক (MOU)",
        "malayalam": "ധാരണാപത്രം (MOU)", "marathi": "सामंजस्य करार (MOU)", "gujarati": "સમજૂતી કરાર (MOU)", "punjabi": "ਸਹਿਮਤੀ ਪੱਤਰ (MOU)", "odia": "ସମଝୋତା ପତ୍ର (MOU)", "urdu": "مفاہمتی یادداشت (MOU)",
    },
    "name_change_affidavit": {
        "tamil": "பெயர் மாற்றம் சத்தியக்கடதாசி", "telugu": "పేరు మార్పు అఫిడవిట్",
        "kannada": "ಹೆಸರು ಬದಲಾವಣೆ ಪ್ರಮಾಣಪತ್ರ", "bengali": "নাম পরিবর্তন হলফনামা",
        "malayalam": "പേര് മാറ്റം സത്യപ്രസ്താവന", "marathi": "नाव बदल प्रतिज्ञापत्र", "gujarati": "નામ ફેરફાર સોગંદનામું", "punjabi": "ਨਾਮ ਤਬਦੀਲੀ ਹਲਫ਼ਨਾਮਾ", "odia": "ନାମ ପରିବର୍ତ୍ତନ ଶପଥପତ୍ର", "urdu": "نام کی تبدیلی حلف نامہ",
    },
    "nda_agreement": {
        "tamil": "வெளிப்படுத்தாத ஒப்பந்தம் (NDA)", "telugu": "గోప్యతా ఒప్పందం (NDA)",
        "kannada": "ಗೌಪ್ಯತಾ ಒಪ್ಪಂದ (NDA)", "bengali": "গোপনীয়তা চুক্তি (NDA)",
        "malayalam": "രഹസ്യാത്മക കരാർ (NDA)", "marathi": "गोपनीयता करार (NDA)", "gujarati": "ગુપ્તતા કરાર (NDA)", "punjabi": "ਗੁਪਤਤਾ ਸਮਝੌਤਾ (NDA)", "odia": "ଗୋପନୀୟତା ଚୁକ୍ତି (NDA)", "urdu": "رازداری کا معاہدہ (NDA)",
    },
    "noc_application": {
        "tamil": "ஆட்சேபனையின்மைச் சான்று விண்ணப்பம் (வங்கி)", "telugu": "NOC దరఖాస్తు (బ్యాంకింగ్)",
        "kannada": "NOC ಅರ್ಜಿ (ಬ್ಯಾಂಕಿಂಗ್)", "bengali": "এনওসি আবেদন (ব্যাংকিং)",
        "malayalam": "NOC അപേക്ഷ (ബാങ്കിംഗ്)", "marathi": "एनओसी अर्ज (बँकिंग)", "gujarati": "એનઓસી અરજી (બેંકિંગ)", "punjabi": "ਐਨਓਸੀ ਬਿਨੈ (ਬੈਂਕਿੰਗ)", "odia": "NOC ଆବେଦନ (ବ୍ୟାଙ୍କିଂ)", "urdu": "این او سی درخواست (بینکنگ)",
    },
    "online_fraud_complaint": {
        "tamil": "ஆன்லைன் மோசடி புகார்", "telugu": "ఆన్‌లైన్ మోసం ఫిర్యాదు",
        "kannada": "ಆನ್‌ಲೈನ್ ವಂಚನೆ ದೂರು", "bengali": "অনলাইন জালিয়াতি অভিযোগ",
        "malayalam": "ഓൺലൈൻ തട്ടിപ്പ് പരാതി", "marathi": "ऑनलाइन फसवणूक तक्रार", "gujarati": "ઓનલાઈન છેતરપિંડી ફરિયાદ", "punjabi": "ਆਨਲਾਈਨ ਧੋਖਾਧੜੀ ਸ਼ਿਕਾਇਤ", "odia": "ଅନଲାଇନ ଠକେଇ ଅଭିଯୋଗ", "urdu": "آن لائن فراڈ شکایت",
    },
    "partnership_agreement": {
        "tamil": "கூட்டாண்மை ஒப்பந்தம்", "telugu": "భాగస్వామ్య ఒప్పందం",
        "kannada": "ಪಾಲುದಾರಿಕೆ ಒಪ್ಪಂದ", "bengali": "অংশীদারিত্ব চুক্তি",
        "malayalam": "പങ്കാളിത്ത കരാർ", "marathi": "भागीदारी करार", "gujarati": "ભાગીદારી કરાર", "punjabi": "ਭਾਈਵਾਲੀ ਸਮਝੌਤਾ", "odia": "ଭାଗିଦାରୀ ଚୁକ୍ତି", "urdu": "شراکت داری معاہدہ",
    },
    "product_return_request": {
        "tamil": "பொருள் திரும்பப் பெறும் கோரிக்கை", "telugu": "ఉత్పత్తి రిటర్న్ అభ్యర్థన",
        "kannada": "ಉತ್ಪನ್ನ ಹಿಂತಿರುಗಿಸುವ ವಿನಂತಿ", "bengali": "পণ্য ফেরত অনুরোধ",
        "malayalam": "ഉൽപ്പന്നം തിരികെ അഭ്യർത്ഥന", "marathi": "उत्पादन परतावा विनंती", "gujarati": "પ્રોડક્ટ રિટર્ન વિનંતી", "punjabi": "ਉਤਪਾਦ ਵਾਪਸੀ ਬੇਨਤੀ", "odia": "ଉତ୍ପାଦ ଫେରସ୍ତ ଅନୁରୋଧ", "urdu": "پروڈکٹ واپسی درخواست",
    },
    "property_dispute_notice": {
        "tamil": "சொத்து தகராறு அறிவிப்பு", "telugu": "ఆస్తి వివాదం నోటీసు",
        "kannada": "ಆಸ್ತಿ ವಿವಾದ ನೋಟಿಸ್", "bengali": "সম্পত্তি বিরোধ নোটিশ",
        "malayalam": "സ്വത്ത് തർക്ക അറിയിപ്പ്", "marathi": "मालमत्ता वाद सूचना", "gujarati": "મિલકત વિવાદ સૂચના", "punjabi": "ਜਾਇਦਾਦ ਵਿਵਾਦ ਨੋਟਿਸ", "odia": "ସମ୍ପତ୍ତି ବିବାଦ ବିଜ୍ଞପ୍ତି", "urdu": "جائیداد تنازعہ نوٹس",
    },
    "property_purchase_agreement": {
        "tamil": "சொத்து வாங்கும் ஒப்பந்தம்", "telugu": "ఆస్తి కొనుగోలు ఒప్పందం",
        "kannada": "ಆಸ್ತಿ ಖರೀದಿ ಒಪ್ಪಂದ", "bengali": "সম্পত্তি ক্রয় চুক্তি",
        "malayalam": "സ്വത്ത് വാങ്ങൽ കരാർ", "marathi": "मालमत्ता खरेदी करार", "gujarati": "મિલકત ખરીદી કરાર", "punjabi": "ਜਾਇਦਾਦ ਖਰੀਦ ਸਮਝੌਤਾ", "odia": "ସମ୍ପତ୍ତି କ୍ରୟ ଚୁକ୍ତି", "urdu": "جائیداد خریداری معاہدہ",
    },
    "property_sale_agreement": {
        "tamil": "சொத்து விற்பனை ஒப்பந்தம்", "telugu": "ఆస్తి అమ్మకం ఒప్పందం",
        "kannada": "ಆಸ್ತಿ ಮಾರಾಟ ಒಪ್ಪಂದ", "bengali": "সম্পত্তি বিক্রয় চুক্তি",
        "malayalam": "സ്വത്ത് വിൽപ്പന കരാർ", "marathi": "मालमत्ता विक्री करार", "gujarati": "મિલકત વેચાણ કરાર", "punjabi": "ਜਾਇਦਾਦ ਵਿਕਰੀ ਸਮਝੌਤਾ", "odia": "ସମ୍ପତ୍ତି ବିକ୍ରୟ ଚୁକ୍ତି", "urdu": "جائیداد فروخت معاہدہ",
    },
    "ration_card_application": {
        "tamil": "ரேஷன் அட்டை விண்ணப்பம்", "telugu": "రేషన్ కార్డు దరఖాస్తు",
        "kannada": "ಪಡಿತರ ಚೀಟಿ ಅರ್ಜಿ", "bengali": "রেশন কার্ড আবেদন",
        "malayalam": "റേഷൻ കാർഡ് അപേക്ഷ", "marathi": "रेशन कार्ड अर्ज", "gujarati": "રેશન કાર્ડ અરજી", "punjabi": "ਰਾਸ਼ਨ ਕਾਰਡ ਬਿਨੈ", "odia": "ରାସନ କାର୍ଡ ଆବେଦନ", "urdu": "راشن کارڈ درخواست",
    },
    "rent_agreement": {
        "tamil": "வாடகை ஒப்பந்தம்", "telugu": "అద్దె ఒప్పందం",
        "kannada": "ಬಾಡಿಗೆ ಒಪ್ಪಂದ", "bengali": "ভাড়া চুক্তি",
        "malayalam": "വാടക കരാർ", "marathi": "भाडे करार", "gujarati": "ભાડા કરાર", "punjabi": "ਕਿਰਾਇਆ ਸਮਝੌਤਾ", "odia": "ଭଡା ଚୁକ୍ତି", "urdu": "کرایہ نامہ",
    },
    "residence_certificate_application": {
        "tamil": "குடியிருப்புச் சான்று விண்ணப்பம்", "telugu": "నివాస ధృవీకరణ పత్రం దరఖాస్తు",
        "kannada": "ವಾಸಸ್ಥಳ ಪ್ರಮಾಣಪತ್ರ ಅರ್ಜಿ", "bengali": "বসবাসের সনদ আবেদন",
        "malayalam": "താമസ സർട്ടിഫിക്കറ്റ് അപേക്ഷ", "marathi": "रहिवासी दाखला अर्ज", "gujarati": "રહેઠાણ પ્રમાણપત્ર અરજી", "punjabi": "ਰਿਹਾਇਸ਼ ਸਰਟੀਫਿਕੇਟ ਬਿਨੈ", "odia": "ବାସସ୍ଥାନ ପ୍ରମାଣପତ୍ର ଆବେଦନ", "urdu": "رہائش سرٹیفکیٹ درخواست",
    },
    "resignation_letter": {
        "tamil": "ராஜினாமா கடிதம்", "telugu": "రాజీనామా లేఖ",
        "kannada": "ರಾಜೀನಾಮೆ ಪತ್ರ", "bengali": "পদত্যাগপত্র",
        "malayalam": "രാജി കത്ത്", "marathi": "राजीनामा पत्र", "gujarati": "રાજીનામું પત્ર", "punjabi": "ਅਸਤੀਫਾ ਪੱਤਰ", "odia": "ଇସ୍ତଫା ପତ୍ର", "urdu": "استعفیٰ خط",
    },
    "scholarship_application": {
        "tamil": "உதவித்தொகை விண்ணப்பம்", "telugu": "స్కాలర్‌షిప్ దరఖాస్తు",
        "kannada": "ವಿದ್ಯಾರ್ಥಿವೇತನ ಅರ್ಜಿ", "bengali": "বৃত্তি আবেদন",
        "malayalam": "സ്കോളർഷിപ്പ് അപേക്ഷ", "marathi": "शिष्यवृत्ती अर्ज", "gujarati": "શિષ્યવૃત્તિ અરજી", "punjabi": "ਵਜ਼ੀਫ਼ਾ ਬਿਨੈ", "odia": "ବୃତ୍ତି ଆବେଦନ", "urdu": "وظیفہ درخواست",
    },
    "service_agreement": {
        "tamil": "சேவை ஒப்பந்தம்", "telugu": "సేవా ఒప్పందం",
        "kannada": "ಸೇವಾ ಒಪ್ಪಂದ", "bengali": "পরিষেবা চুক্তি",
        "malayalam": "സേവന കരാർ", "marathi": "सेवा करार", "gujarati": "સેવા કરાર", "punjabi": "ਸੇਵਾ ਸਮਝੌਤਾ", "odia": "ସେବା ଚୁକ୍ତି", "urdu": "خدمت معاہدہ",
    },
    "service_complaint": {
        "tamil": "சேவை புகார்", "telugu": "సేవా ఫిర్యాదు",
        "kannada": "ಸೇವಾ ದೂರು", "bengali": "পরিষেবা অভিযোগ",
        "malayalam": "സേവന പരാതി", "marathi": "सेवा तक्रार", "gujarati": "સેવા ફરિયાદ", "punjabi": "ਸੇਵਾ ਸ਼ਿਕਾਇਤ", "odia": "ସେବା ଅଭିଯୋଗ", "urdu": "سروس شکایت",
    },
    "tc_application": {
        "tamil": "இடமாற்றுச் சான்று (TC) விண்ணப்பம்", "telugu": "బదిలీ ధృవీకరణ పత్రం (TC) దరఖాస్తు",
        "kannada": "ವರ್ಗಾವಣೆ ಪ್ರಮಾಣಪತ್ರ (TC) ಅರ್ಜಿ", "bengali": "ট্রান্সফার সার্টিফিকেট (TC) আবেদন",
        "malayalam": "ട്രാൻസ്ഫർ സർട്ടിഫിക്കറ്റ് (TC) അപേക്ഷ", "marathi": "बदली प्रमाणपत्र (TC) अर्ज", "gujarati": "ટ્રાન્સફર પ્રમાણપત્ર (TC) અરજી", "punjabi": "ਟ੍ਰਾਂਸਫਰ ਸਰਟੀਫਿਕੇਟ (TC) ਬਿਨੈ", "odia": "ଟ୍ରାନ୍ସଫର ପ୍ରମାଣପତ୍ର (TC) ଆବେଦନ", "urdu": "ٹرانسفر سرٹیفکیٹ (TC) درخواست",
    },
    "tenancy_termination_notice": {
        "tamil": "குடியிருப்பு முடிவு அறிவிப்பு", "telugu": "కిరాయిదారీ ముగింపు నోటీసు",
        "kannada": "ಬಾಡಿಗೆ ಅಂತ್ಯ ನೋಟಿಸ್", "bengali": "ভাড়া সমাপ্তি নোটিশ",
        "malayalam": "വാടക അവസാനിപ്പിക്കൽ അറിയിപ്പ്", "marathi": "भाडेकरार समाप्ती सूचना", "gujarati": "ભાડૂત સમાપ્તિ સૂચના", "punjabi": "ਕਿਰਾਏਦਾਰੀ ਸਮਾਪਤੀ ਨੋਟਿਸ", "odia": "ଭଡାଟିଆ ସମାପ୍ତି ବିଜ୍ଞପ୍ତି", "urdu": "کرایہ داری کے خاتمے کا نوٹس",
    },
    "threat_complaint": {
        "tamil": "மிரட்டல் புகார்", "telugu": "బెదిరింపు ఫిర్యాదు",
        "kannada": "ಬೆದರಿಕೆ ದೂರು", "bengali": "হুমকি অভিযোগ",
        "malayalam": "ഭീഷണി പരാതി", "marathi": "धमकी तक्रार", "gujarati": "ધમકી ફરિયાદ", "punjabi": "ਧਮਕੀ ਸ਼ਿਕਾਇਤ", "odia": "ଧମକ ଅଭିଯୋଗ", "urdu": "دھمکی شکایت",
    },
    "vehicle_theft_complaint": {
        "tamil": "வாகன திருட்டு புகார்", "telugu": "వాహన దొంగతనం ఫిర్యాదు",
        "kannada": "ವಾಹನ ಕಳ್ಳತನ ದೂರು", "bengali": "যানবাহন চুরি অভিযোগ",
        "malayalam": "വാഹന മോഷണം പരാതി", "marathi": "वाहन चोरी तक्रार", "gujarati": "વાહન ચોરી ફરિયાદ", "punjabi": "ਵਾਹਨ ਚੋਰੀ ਸ਼ਿਕਾਇਤ", "odia": "ଯାନ ଚୋରି ଅଭିଯୋଗ", "urdu": "گاڑی چوری شکایت",
    },
}


def localized_title(template: DraftTemplateDefinition, language: str) -> str:
    """The document title to display for `template` in `language`.

    "hindi" reuses the template's own `hindi_name` (already authored per
    template YAML, not duplicated here). Any other language, or any template
    not covered above, falls back to `template.name` (the official English
    title) -- per Part 53's own rule, a title is only translated "when a
    reliable translation exists," never fabricated.
    """
    if language == "hindi":
        return template.hindi_name
    if template.draft_id in _TITLE_TRANSLATIONS:
        translated = _TITLE_TRANSLATIONS[template.draft_id].get(language)
        if translated:
            return translated
    translated = _ADDITIONAL_TEMPLATE_NAME_TRANSLATIONS.get(template.draft_id, {}).get(language)
    if translated:
        return translated
    # Last resort before English: the ten Eighth Schedule languages none of the
    # tables above cover. Kept last (and in its own module) because those names
    # are unreviewed machine translations -- see
    # `app/drafting/eighth_schedule_names.py` for the provenance warning.
    return _EIGHTH_SCHEDULE_NAMES.get(template.draft_id, {}).get(language, template.name)


# Part 51 "Draft Generation Quality Pass" item 2: short, natural
# Tamil/Telugu/Kannada/Bengali document names a person would actually type
# ("சட்ட அறிவிப்பு" -- plain "legal notice") rather than the full official
# title above ("பொது சட்ட அறிவிப்பு" -- "GENERAL legal notice"). Needed as a
# SEPARATE table, not folded into `_TITLE_TRANSLATIONS`: `DraftIntentDetector`
# matches by checking whether a stored phrase appears as a substring of the
# user's message, so a full official title longer than what the user
# actually typed can never match -- the same reason the English YAML
# `trigger_phrases` list a short "legal notice" alongside the template's
# full English `name` ("General Legal Notice").
_SHORT_TRIGGER_TRANSLATIONS: dict[str, dict[str, str]] = {
    "legal_notice": {
        "tamil": "சட்ட அறிவிப்பு", "telugu": "న్యాయ నోటీసు",
        "kannada": "ಕಾನೂನು ನೋಟಿಸ್", "bengali": "আইনি নোটিশ",
        "malayalam": "നിയമ അറിയിപ്പ്", "marathi": "कायदेशीर सूचना", "gujarati": "કાનૂની સૂચના", "punjabi": "ਕਾਨੂੰਨੀ ਨੋਟਿਸ", "odia": "ଆଇନଗତ ବିଜ୍ଞପ୍ତି", "urdu": "قانونی نوٹس",
    },
    "cyber_crime_complaint": {
        "tamil": "சைபர் குற்ற புகார்", "telugu": "సైబర్ నేర ఫిర్యాదు",
        "kannada": "ಸೈಬರ್ ಅಪರಾಧ ದೂರು", "bengali": "সাইবার অপরাধ অভিযোগ",
        "malayalam": "സൈബർ കുറ്റകൃത്യ പരാതി", "marathi": "सायबर गुन्हा तक्रार", "gujarati": "સાયબર ગુનો ફરિયાદ", "punjabi": "ਸਾਈਬਰ ਅਪਰਾਧ ਸ਼ਿਕਾਇਤ", "odia": "ସାଇବର ଅପରାଧ ଅଭିଯୋଗ", "urdu": "سائبر جرم شکایت",
    },
    "police_complaint": {
        "tamil": "காவல் நிலைய புகார்", "telugu": "పోలీసు ఫిర్యాదు",
        "kannada": "ಪೊಲೀಸ್ ದೂರು", "bengali": "পুলিশ অভিযোগ",
        "malayalam": "പോലീസ് പരാതി", "marathi": "पोलीस तक्रार", "gujarati": "પોલીસ ફરિયાદ", "punjabi": "ਪੁਲਿਸ ਸ਼ਿਕਾਇਤ", "odia": "ପୋଲିସ୍ ଅଭିଯୋଗ", "urdu": "پولیس شکایت",
    },
    "rti_application": {
        "tamil": "தகவல் உரிமை விண்ணப்பம்", "telugu": "సమాచార హక్కు దరఖాస్తు",
        "kannada": "ಮಾಹಿತಿ ಹಕ್ಕು ಅರ್ಜಿ", "bengali": "তথ্য অধিকার আবেদন",
        "malayalam": "വിവരാവകാശ അപേക്ഷ", "marathi": "माहिती अधिकार अर्ज", "gujarati": "માહિતી અધિકાર અરજી", "punjabi": "ਸੂਚਨਾ ਦਾ ਅਧਿਕਾਰ ਬਿਨੈ", "odia": "ସୂଚନା ଅଧିକାର ଆବେଦନ", "urdu": "حق اطلاعات درخواست",
    },
    "consumer_complaint": {
        "tamil": "நுகர்வோர் புகார்", "telugu": "వినియోగదారు ఫిర్యాదు",
        "kannada": "ಗ್ರಾಹಕ ದೂರು", "bengali": "ভোক্তা অভিযোগ",
        "malayalam": "ഉപഭോക്തൃ പരാതി", "marathi": "ग्राहक तक्रार", "gujarati": "ગ્રાહક ફરિયાદ", "punjabi": "ਖਪਤਕਾਰ ਸ਼ਿਕਾਇਤ", "odia": "ଗ୍ରାହକ ଅଭିଯୋଗ", "urdu": "صارف شکایت",
    },
    "cheque_bounce_notice": {
        "tamil": "காசோலை மறுப்பு அறிவிப்பு", "telugu": "చెక్ బౌన్స్ నోటీసు",
        "kannada": "ಚೆಕ್ ಬೌನ್ಸ್ ನೋಟಿಸ್", "bengali": "চেক প্রত্যাখ্যান নোটিশ",
        "malayalam": "ചെക്ക് മടങ്ങൽ അറിയിപ്പ്", "marathi": "धनादेश परत सूचना", "gujarati": "ચેક બાઉન્સ સૂચના", "punjabi": "ਚੈੱਕ ਬਾਊਂਸ ਨੋਟਿਸ", "odia": "ଚେକ୍ ବାଉନ୍ସ ବିଜ୍ଞପ୍ତି", "urdu": "چیک باؤنس نوٹس",
    },
    "recovery_notice": {
        "tamil": "பணம் வசூல் அறிவிப்பு", "telugu": "డబ్బు వసూలు నోటీసు",
        "kannada": "ಹಣ ವಸೂಲಿ ನೋಟಿಸ್", "bengali": "টাকা আদায় নোটিশ",
        "malayalam": "പണം തിരിച്ചുപിടിക്കൽ അറിയിപ്പ്", "marathi": "रक्कम वसुली सूचना", "gujarati": "રકમ વસૂલાત સૂચના", "punjabi": "ਰਕਮ ਵਸੂਲੀ ਨੋਟਿਸ", "odia": "ଅର୍ଥ ଆଦାୟ ବିଜ୍ଞପ୍ତି", "urdu": "رقم وصولی نوٹس",
    },
    "rent_notice": {
        "tamil": "வாடகை அறிவிப்பு", "telugu": "అద్దె నోటీసు",
        "kannada": "ಬಾಡಿಗೆ ನೋಟಿಸ್", "bengali": "ভাড়া নোটিশ",
        "malayalam": "വാടക അറിയിപ്പ്", "marathi": "भाडे सूचना", "gujarati": "ભાડું સૂચના", "punjabi": "ਕਿਰਾਇਆ ਨੋਟਿਸ", "odia": "ଭଡା ବିଜ୍ଞପ୍ତି", "urdu": "کرایہ نوٹس",
    },
    "employment_notice": {
        "tamil": "வேலை நிலுவை அறிவிப்பு", "telugu": "ఉద్యోగ బకాయి నోటీసు",
        "kannada": "ಉದ್ಯೋಗ ಬಾಕಿ ನೋಟಿಸ್", "bengali": "চাকরির বকেয়া নোটিশ",
        "malayalam": "തൊഴിൽ കുടിശ്ശിക അറിയിപ്പ്", "marathi": "नोकरी थकबाकी सूचना", "gujarati": "રોજગાર બાકી સૂચના", "punjabi": "ਰੁਜ਼ਗਾਰ ਬਕਾਇਆ ਨੋਟਿਸ", "odia": "ନିଯୁକ୍ତି ବକେୟା ବିଜ୍ଞପ୍ତି", "urdu": "ملازمت واجبات نوٹس",
    },
    "affidavit": {
        "tamil": "சத்தியக்கடதாசி", "telugu": "అఫిడవిట్",
        "kannada": "ಪ್ರಮಾಣಪತ್ರ", "bengali": "হলফনামা",
        "malayalam": "സത്യപ്രസ്താവന", "marathi": "प्रतिज्ञापत्र", "gujarati": "સોગંદનામું", "punjabi": "ਹਲਫ਼ਨਾਮਾ", "odia": "ଶପଥପତ୍ର", "urdu": "حلف نامہ",
    },
    "sp_complaint": {
        "tamil": "கண்காணிப்பாளர் புகார்", "telugu": "సూపరింటెండెంట్ ఫిర్యాదు",
        "kannada": "ಅಧೀಕ್ಷಕರ ದೂರು", "bengali": "সুপার অভিযোগ",
        "malayalam": "സൂപ്രണ്ട് പരാതി", "marathi": "अधीक्षक तक्रार", "gujarati": "સુપ્રિન્ટેન્ડન્ટ ફરિયાદ", "punjabi": "ਸੁਪਰਡੈਂਟ ਸ਼ਿਕਾਇਤ", "odia": "ସୁପରିଣ୍ଟେଣ୍ଡେଣ୍ଟ ଅଭିଯୋଗ", "urdu": "سپرنٹنڈنٹ شکایت",
    },
}


def localized_document_name(template: DraftTemplateDefinition, language: str) -> str:
    """Short, natural document name for `template` in `language` -- meant for
    interpolating into a conversational sentence (e.g. "Let's draft your
    {name}."/"Here is your {name} draft:") where the FULL official title
    (`localized_title`, used for export headings/the template-selection
    list) would read overly long/formal for a chat aside.

    Part 55 "Draft Audit -- Language Propagation Fix": every wrapper
    sentence in `app/drafting/conversation.py` (`lets_draft`,
    `draft_ready_intro`, `draft_saved_with_count`, etc.) interpolated
    `template.name` -- the plain English title -- into its `{template}`
    placeholder regardless of the conversation's own language, so even a
    fully Tamil/Telugu/Kannada/Bengali reply still read "... your Police
    Complaint (Application to SHO) ..." with the one document-name phrase
    left in English. Reuses the SAME short-name table
    `DraftIntentDetector` already scores against (`all_localized_names`),
    so the name a user sees is also a name they could type back in to
    re-select the same template. Falls back to `template.name` (English)
    exactly like `localized_title`, for hindi (which already uses its own
    `hindi_name` elsewhere) or any language/template not covered.
    """
    if language == "hindi":
        return template.hindi_name
    if template.draft_id in _SHORT_TRIGGER_TRANSLATIONS:
        translated = _SHORT_TRIGGER_TRANSLATIONS[template.draft_id].get(language)
        if translated:
            return translated
    translated = _ADDITIONAL_TEMPLATE_NAME_TRANSLATIONS.get(template.draft_id, {}).get(language)
    if translated:
        return translated
    return _EIGHTH_SCHEDULE_NAMES.get(template.draft_id, {}).get(language, template.name)


def all_localized_names(draft_id: str) -> list[str]:
    """Every non-English name recorded for `draft_id` -- the full official
    title (Tamil/Telugu/Kannada/Bengali, from `_TITLE_TRANSLATIONS` above)
    plus the shorter natural trigger form (`_SHORT_TRIGGER_TRANSLATIONS`),
    plus (for the 43 templates covered by neither of those two, originally
    "flagship-only" tables) `_ADDITIONAL_TEMPLATE_NAME_TRANSLATIONS`. Used by
    `DraftIntentDetector` so a user typing the document name in one of these
    languages, in either its full or short form, is recognized -- the same
    way `template.hindi_name` already is. `hindi_name` itself isn't
    duplicated here since callers already score it separately (it lives on
    `DraftTemplateDefinition`, not in this table).
    """
    return [
        *_TITLE_TRANSLATIONS.get(draft_id, {}).values(),
        *_SHORT_TRIGGER_TRANSLATIONS.get(draft_id, {}).values(),
        *_ADDITIONAL_TEMPLATE_NAME_TRANSLATIONS.get(draft_id, {}).values(),
        *_EIGHTH_SCHEDULE_NAMES.get(draft_id, {}).values(),
    ]
