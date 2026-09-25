import json
import os
import time
import urllib.request

BASE_URL = "http://127.0.0.1:8000/chat"
OUT_DIR = os.path.join(os.path.dirname(__file__), 'test_results')
os.makedirs(OUT_DIR, exist_ok=True)

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
        body, t = post(q, lang)
        fname = os.path.join(OUT_DIR, f'test_{i:02d}.json')
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump({
                'test': i,
                'language': lang,
                'question': q,
                'response': body,
                'time_ms': t
            }, f, ensure_ascii=False, indent=2)
        print(f'Wrote {fname}')
    print('Done')
