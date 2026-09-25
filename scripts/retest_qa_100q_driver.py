"""100-test end-to-end QA driver, live against the running backend.
No mocks. Writes requests/responses/exports + results.json into OUT.
Organized in phases; independent/stateless batches run concurrently
(bounded semaphore), stateful sequences run in order.
"""
import asyncio
import io
import json
import time
import wave
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
OUT = Path("C:/Law Chatbot/Law Chatbot/qa-100q-retest-20260924")
REQ = OUT / "requests"
RESP = OUT / "responses"
EXP = OUT / "exports"

results: dict[str, dict] = {}
state: dict = {}
LOCK = asyncio.Lock()


def _save_results():
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def _save_state():
    with open(OUT / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


async def call(client, tid, method, path, meta=None, **kw):
    started = time.time()
    url = f"{BASE_URL}{path}"
    entry = {"tid": tid, "method": method, "url": path, "meta": meta or {}}
    try:
        resp = await client.request(method, url, timeout=kw.pop("timeout", 180.0), **kw)
        entry["http_status"] = resp.status_code
        entry["latency_s"] = round(time.time() - started, 2)
        ctype = resp.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                entry["body"] = resp.json()
            except Exception:
                entry["body_text"] = resp.text[:2000]
        else:
            entry["body_bytes"] = len(resp.content)
            entry["_raw_content"] = resp.content
    except Exception as exc:
        entry["http_status"] = None
        entry["latency_s"] = round(time.time() - started, 2)
        entry["error"] = f"{type(exc).__name__}: {exc}"

    raw = entry.pop("_raw_content", None)
    async with LOCK:
        try:
            req_payload = kw.get("json")
            if req_payload is None and "data" in kw:
                req_payload = {k: v for k, v in kw["data"].items()} if isinstance(kw.get("data"), dict) else str(kw.get("data"))
            with open(REQ / f"{tid}.json", "w", encoding="utf-8") as f:
                json.dump(req_payload or {}, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        if "body" in entry:
            with open(RESP / f"{tid}.json", "w", encoding="utf-8") as f:
                json.dump(entry["body"], f, ensure_ascii=False, indent=2)
        elif raw is not None:
            ext = {"pdf": "pdf", "docx": "docx", "txt": "txt"}.get((meta or {}).get("export_fmt"), "bin")
            path_out = EXP / f"{tid}.{ext}"
            path_out.write_bytes(raw)
            entry["saved_to"] = str(path_out)
        results[tid] = entry
        _save_results()
    print(f"[{tid}] {method} {path} -> {entry.get('http_status')} ({entry.get('latency_s')}s) {entry.get('error','')}", flush=True)
    return entry


def hdr(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def silent_wav(seconds=2, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x00" * rate * seconds)
    return buf.getvalue()


def ocr_test_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (900, 200), color="white")
    d = ImageDraw.Draw(img)
    d.text((20, 60), "AGREEMENT FOR SALE OF PROPERTY VALUE RS 500000", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def minimal_pdf(text_lines):
    stream = "\n".join(f"BT /F1 12 Tf 20 {250 - i*20} Td ({line}) Tj ET" for i, line in enumerate(text_lines))
    content = (
        f"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        f"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        f"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 400 300]/Contents 4 0 R"
        f"/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        f"4 0 obj<</Length {len(stream)}>>stream\n{stream}\nendstream\nendobj\n"
        f"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        f"xref\n0 6\n0000000000 65535 f \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n0\n%%EOF"
    )
    return content.encode("utf-8")


# --- 23 supported languages: (code, name, question) ---
# Same underlying legal fact (NI Act s.138 cheque-bounce punishment, confirmed
# well-indexed in KB) asked in each language, so citation accuracy is
# directly comparable across the language matrix.
LANG_QUESTIONS = [
    ("en", "English", "What is the punishment for a bounced cheque in India?"),
    ("hi", "Hindi", "चेक बाउंस होने पर भारत में क्या सजा है?"),
    ("as", "Assamese", "চেক বাউন্স হ'লে ভাৰতত কি শাস্তি আছে?"),
    ("bn", "Bengali", "চেক বাউন্স হলে ভারতে কী শাস্তি হয়?"),
    ("brx", "Bodo", "चेक बाउंस जायखौ भारतनि सजा मा?"),
    ("doi", "Dogri", "चेक बाउंस होने पर भारत च की सजा ऐ?"),
    ("gu", "Gujarati", "ચેક બાઉન્સ થવા પર ભારતમાં શું સજા છે?"),
    ("kn", "Kannada", "ಚೆಕ್ ಬೌನ್ಸ್ ಆದರೆ ಭಾರತದಲ್ಲಿ ಏನು ಶಿಕ್ಷೆ?"),
    ("ks", "Kashmiri", "चेक बाउंस गछिथ भारत मंज़ क्या सजा छे?"),
    ("kok", "Konkani", "चेक बाउंस जाल्यार भारतांत कसली शिक्षा आसा?"),
    ("ml", "Malayalam", "ചെക്ക് ബൗൺസ് ആയാൽ ഇന്ത്യയിൽ ശിക്ഷ എന്താണ്?"),
    ("mni", "Manipuri", "check bounce oirabadi India da kari punishment oi?"),
    ("mr", "Marathi", "चेक बाउन्स झाल्यास भारतात काय शिक्षा आहे?"),
    ("mai", "Maithili", "चेक बाउंस भेला पर भारत मे की सजा अछि?"),
    ("ne", "Nepali", "चेक बाउन्स भएमा भारतमा के सजाय हुन्छ?"),
    ("or", "Odia", "ଚେକ୍ ବାଉନ୍ସ ହେଲେ ଭାରତରେ କଣ ଶାସ୍ତି?"),
    ("pa", "Punjabi", "ਚੈੱਕ ਬਾਊਂਸ ਹੋਣ 'ਤੇ ਭਾਰਤ ਵਿੱਚ ਕੀ ਸਜ਼ਾ ਹੈ?"),
    ("sa", "Sanskrit", "चेक-बाउंस-प्रसंगे भारते का शिक्षा भवति?"),
    ("sat", "Santali", "check bounce lekhan India re cet sajai kana?"),
    ("sd", "Sindhi", "چيڪ باؤنس ٿيڻ تي ڀارت ۾ ڪهڙي سزا آهي؟"),
    ("ta", "Tamil", "காசோலை மறுக்கப்பட்டால் இந்தியாவில் என்ன தண்டனை?"),
    ("te", "Telugu", "చెక్ బౌన్స్ అయితే భారతదేశంలో శిక్ష ఏమిటి?"),
    ("ur", "Urdu", "چیک باؤنس ہونے پر بھارت میں کیا سزا ہے؟"),
]

SEARCH_MODES = ["semantic", "keyword", "hybrid", "metadata", "section", "act"]


async def main():
    ts = int(time.time())
    limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
    async with httpx.AsyncClient(limits=limits) as client:
        # ---------- Setup: primary test users (reuse if already registered) ----------
        if state.get("token_a") and state.get("token_b"):
            print("Reusing existing token_a/token_b from state.json", flush=True)
        else:
            email_a = f"qa100.a.{ts}@qalegalai-testdomain.com"
            email_b = f"qa100.b.{ts}@qalegalai-testdomain.com"
            pw = "QaTest!Pass1234"
            r = await call(client, "SETUP_register_A", "POST", "/register",
                            json={"email": email_a, "password": pw, "full_name": "QA100 User A"})
            state["token_a"] = r.get("body", {}).get("access_token")
            state["user_a_id"] = r.get("body", {}).get("user_id")
            r = await call(client, "SETUP_register_B", "POST", "/register",
                            json={"email": email_b, "password": pw, "full_name": "QA100 User B"})
            state["token_b"] = r.get("body", {}).get("access_token")
            state["user_b_id"] = r.get("body", {}).get("user_id")
            _save_state()
        hdr_a = hdr(state["token_a"])
        hdr_b = hdr(state["token_b"])

        # NOTE: originally 6 -- reduced after live discovery (this run) that
        # 6 concurrent /chat calls drove the local (non-Atlas) vector scan
        # past 100-190s each, timed out Redis reads, and caused Postgres
        # long-term-memory writes to fail; see CONCURRENCY_BUG_EVIDENCE_6way.log.
        # Kept low here so the rest of the suite can actually complete; the
        # concurrency behavior itself is tested deliberately and narrowly at
        # T097/T098 instead of accidentally here.
        sem = asyncio.Semaphore(2)

        # ================= T001-T023: language coverage =================
        async def lang_task(idx, code, name, question):
            tid = f"T{idx:03d}_lang_{code}"
            async with sem:
                r = await call(client, tid, "POST", "/chat", meta={"feature": "chat_language_coverage", "language": name},
                                json={"question": question}, headers=hdr_a)
                return tid, r

        lang_tasks = [lang_task(i + 1, code, name, q) for i, (code, name, q) in enumerate(LANG_QUESTIONS)]
        await asyncio.gather(*lang_tasks)

        # ================= T024-T032: chat quality group =================
        r = await call(client, "T024_typo_english", "POST", "/chat", meta={"feature": "chat_typo", "language": "English"},
                        json={"question": "wat is teh punishment for chek bonuce in indai"}, headers=hdr_a)

        r = await call(client, "T025_romanized_hindi_typos", "POST", "/chat", meta={"feature": "chat_romanized_typo", "language": "Hinglish"},
                        json={"question": "chek bounc hone pe kya saja milti h india m"}, headers=hdr_a)

        # multi-turn thread (3 turns, same session)
        r = await call(client, "T026_multiturn_turn1", "POST", "/chat", meta={"feature": "multi_turn_context", "language": "Hindi"},
                        json={"question": "mera landlord mera security deposit wapas nahi de raha hai"}, headers=hdr_a)
        mt_session = r.get("body", {}).get("session_id")
        state["multiturn_session"] = mt_session
        r = await call(client, "T027_multiturn_turn2", "POST", "/chat", meta={"feature": "multi_turn_context", "language": "Hindi"},
                        json={"question": "iske liye mujhe kis court mein jana hoga?", "session_id": mt_session}, headers=hdr_a)
        r = await call(client, "T028_multiturn_turn3_pronoun", "POST", "/chat", meta={"feature": "multi_turn_context", "language": "Hindi"},
                        json={"question": "uska time limit kitna hai?", "session_id": mt_session}, headers=hdr_a)

        # new session isolation: same user, brand-new session, ask about "what did I just ask"
        r = await call(client, "T029_new_session_isolation", "POST", "/chat", meta={"feature": "session_isolation", "language": "English"},
                        json={"question": "What did I just ask you about in my previous message?"}, headers=hdr_a)

        r = await call(client, "T030_ambiguous_question", "POST", "/chat", meta={"feature": "ambiguous_handling", "language": "Hinglish"},
                        json={"question": "kya karu ab"}, headers=hdr_a)

        r = await call(client, "T031_gk_fallback", "POST", "/chat", meta={"feature": "gk_fallback", "language": "English"},
                        json={"question": "Who is the current Chief Justice of India as of this year?"}, headers=hdr_a)

        r = await call(client, "T032_out_of_scope_refusal", "POST", "/chat", meta={"feature": "honest_refusal", "language": "English"},
                        json={"question": "What is the punishment for littering under the criminal code of Bhutan?"}, headers=hdr_a)

        _save_state()
        print("PHASE 1 (T001-T032) DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
