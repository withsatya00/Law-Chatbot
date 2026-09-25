import json
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8000/chat"

TESTS = [
    ("English", "What is FIR?"),
    ("Hindi", "एफआईआर क्या होती है?"),
    ("Hinglish", "FIR kya hota hai?"),
    ("English", "What is Bail?"),
    ("Hindi", "जमानत क्या होती है?"),
    ("Hinglish", "Bail kya hoti hai?"),
    ("English", "Difference between FIR and NCR."),
    ("Hindi", "एफआईआर और एनसीआर में क्या अंतर है?"),
    ("Hinglish", "FIR aur NCR me difference kya hai?"),
    ("English", "My bike has been stolen.\nWhat should I do?"),
    ("Hindi", "मेरी बाइक चोरी हो गई है।\nमुझे क्या करना चाहिए?"),
    ("Hinglish", "Meri bike chori ho gayi hai.\nMain kya karu?"),
    ("English", "Draft a Police Complaint for bike theft."),
    ("Hindi", "मेरी बाइक चोरी के लिए पुलिस शिकायत तैयार करें।"),
    ("Hinglish", "Bike chori ki complaint draft kar do.")
]

def post(question, language):
    data = json.dumps({"question": question, "language": language}).encode("utf-8")
    req = urllib.request.Request(BASE_URL, data=data, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read()
            elapsed = (time.perf_counter() - start) * 1000.0
            return body.decode("utf-8"), elapsed
    except Exception as e:  # noqa: BLE001 - smoke-test harness: a failed request is recorded as a result row, not a crash
        elapsed = (time.perf_counter() - start) * 1000.0
        return json.dumps({"error": str(e)}), elapsed

if __name__ == '__main__':
    for i, (lang, q) in enumerate(TESTS, start=1):
        print('-' * 40)
        print(f'TEST #{i:02d}')
        print('Language:')
        print(lang)
        print('\nUser Query:')
        print(q)
        print('\nAssistant Response:')
        body, t = post(q, lang)
        print(body)
        print('\nEvaluation')
        print('Accuracy:')
        print()
        print('Language Quality:')
        print()
        print('Naturalness:')
        print()
        print('Legal Accuracy:')
        print()
        print(f'Response Time: {t:.1f} ms')
        print('\nPASS / FAIL')
        print('\nReason (if failed)')
    print('\nAll tests completed.')
