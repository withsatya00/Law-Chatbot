"""Phase 2 (T033-T100) of the 100-test QA driver. Loads state.json written
by phase 1 for the primary user tokens, runs drafting lifecycle, search,
upload/OCR, voice, auth mechanics, mongo/redis verification, privacy
deletion, cross-user isolation, validation, security, rate limiting,
errors, concurrency, and API-compatibility checks.
"""
import asyncio
import io
import json
import time
import wave
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
OUT = Path("C:/Law Chatbot/Law Chatbot/qa-100q-full-20260924")
REQ = OUT / "requests"
RESP = OUT / "responses"
EXP = OUT / "exports"

results: dict = json.load(open(OUT / "results.json", encoding="utf-8")) if (OUT / "results.json").exists() else {}
state: dict = json.load(open(OUT / "state.json", encoding="utf-8"))
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
            entry["_headers"] = dict(resp.headers)
    except Exception as exc:
        entry["http_status"] = None
        entry["latency_s"] = round(time.time() - started, 2)
        entry["error"] = f"{type(exc).__name__}: {exc}"

    raw = entry.pop("_raw_content", None)
    hdrs = entry.pop("_headers", None)
    if hdrs is not None:
        entry["response_headers"] = hdrs
    async with LOCK:
        try:
            req_payload = kw.get("json")
            if req_payload is None and "data" in kw and isinstance(kw.get("data"), dict):
                req_payload = dict(kw["data"])
            with open(REQ / f"{tid}.json", "w", encoding="utf-8") as f:
                json.dump(req_payload or {}, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        if "body" in entry:
            with open(RESP / f"{tid}.json", "w", encoding="utf-8") as f:
                json.dump(entry["body"], f, ensure_ascii=False, indent=2)
        elif raw is not None:
            ext = (meta or {}).get("export_fmt", "bin")
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


async def main():
    ts = int(time.time())
    hdr_a = hdr(state["token_a"])
    hdr_b = hdr(state["token_b"])

    async with httpx.AsyncClient() as client:
        # ================= T033-T038: drafting triggers, multi-language =================
        r = await call(client, "T033_draft_trigger_en", "POST", "/chat", meta={"feature": "drafting_trigger", "language": "English"},
                        json={"question": "I want to file a consumer complaint"}, headers=hdr_a)
        session_en = r.get("body", {}).get("session_id")
        draft_en_ok = bool((r.get("body", {}).get("draft") or {}).get("stage"))

        r = await call(client, "T034_draft_trigger_hi_terse", "POST", "/chat", meta={"feature": "drafting_trigger", "language": "Hindi"},
                        json={"question": "पुलिस शिकायत"}, headers=hdr_a)
        session_hi = r.get("body", {}).get("session_id")

        r = await call(client, "T035_draft_trigger_hinglish", "POST", "/chat", meta={"feature": "drafting_trigger", "language": "Hinglish"},
                        json={"question": "cheque bounce ho gaya hai, notice draft karna hai"}, headers=hdr_a)
        session_hinglish = r.get("body", {}).get("session_id")
        draft_hinglish_ok = bool((r.get("body", {}).get("draft") or {}).get("stage"))

        r = await call(client, "T036_draft_trigger_mr", "POST", "/chat", meta={"feature": "drafting_trigger", "language": "Marathi"},
                        json={"question": "पोलीस तक्रार नोंदवायची आहे"}, headers=hdr_a)

        r = await call(client, "T037_draft_trigger_ta", "POST", "/chat", meta={"feature": "drafting_trigger", "language": "Tamil"},
                        json={"question": "நுகர்வோர் புகார் பதிவு செய்ய வேண்டும்"}, headers=hdr_a)

        r = await call(client, "T038_draft_trigger_bn", "POST", "/chat", meta={"feature": "drafting_trigger", "language": "Bengali"},
                        json={"question": "ভোক্তা অভিযোগ দায়ের করতে চাই"}, headers=hdr_a)

        # Pick whichever draft actually reached 'collecting' to run the rest
        # of the lifecycle against (English consumer_complaint preferred).
        lifecycle_session = session_en if draft_en_ok else (session_hinglish if draft_hinglish_ok else session_en)
        state["lifecycle_session"] = lifecycle_session

        fill_msg = (
            "applicant name is Ravi Kumar. applicant address is 10 Anna Salai Chennai. "
            "mobile number is 9988776655. respondent name is XYZ Electronics. "
            "respondent address is 20 MG Road Chennai. product or service is a refrigerator. "
            "facts is that I purchased a refrigerator which stopped working within a week and "
            "the seller refused replacement or refund. expected relief is a full refund. "
            "place is Chennai. purchase date is 01-08-2026. amount paid is Rs 25000. "
            "defect or deficiency is compressor failure within a week of purchase."
        )
        r = await call(client, "T039_draft_field_collection", "POST", "/chat", meta={"feature": "drafting_field_collection", "language": "English"},
                        json={"question": fill_msg, "session_id": lifecycle_session}, headers=hdr_a, timeout=180)
        draft = r.get("body", {}).get("draft") or {}
        draft_id = draft.get("draft_id")
        if not draft_id and draft.get("missing_fields"):
            r2 = await call(client, "T039b_draft_field_collection_followup", "POST", "/chat",
                             meta={"feature": "drafting_field_collection", "language": "English"},
                             json={"question": "Please finalize the draft with the information already provided.",
                                   "session_id": lifecycle_session}, headers=hdr_a, timeout=180)
            draft = r2.get("body", {}).get("draft") or draft
            draft_id = draft.get("draft_id") or draft_id
        state["lifecycle_draft_id"] = draft_id
        _save_state()

        if draft_id:
            r = await call(client, "T040_draft_correction", "POST", "/chat", meta={"feature": "drafting_correction", "language": "English"},
                            json={"question": "change the amount paid to Rs 30000", "session_id": lifecycle_session},
                            headers=hdr_a, timeout=180)

            r = await call(client, "T041_draft_regeneration", "POST", "/chat", meta={"feature": "drafting_regeneration", "language": "English"},
                            json={"question": "please regenerate the full draft with these updates", "session_id": lifecycle_session},
                            headers=hdr_a, timeout=180)

            r = await call(client, "T042_draft_version_history", "GET", f"/draft/{draft_id}/versions",
                            meta={"feature": "drafting_version_history", "language": "English"},
                            params={"session_id": lifecycle_session}, headers=hdr_a)

            r = await call(client, "T043_draft_export_pdf", "POST", "/draft/export",
                            meta={"feature": "drafting_export_pdf", "language": "English", "export_fmt": "pdf"},
                            json={"draft_id": draft_id, "session_id": lifecycle_session, "format": "pdf"}, headers=hdr_a)

            r = await call(client, "T044_draft_export_docx", "POST", "/draft/export",
                            meta={"feature": "drafting_export_docx", "language": "English", "export_fmt": "docx"},
                            json={"draft_id": draft_id, "session_id": lifecycle_session, "format": "docx"}, headers=hdr_a)

            r = await call(client, "T045_draft_translate", "POST", "/draft/translate",
                            meta={"feature": "drafting_translate", "language": "Tamil"},
                            json={"draft_id": draft_id, "session_id": lifecycle_session,
                                  "target_language": "tamil", "source_language": "english"}, headers=hdr_a, timeout=120)
        else:
            for tid in ["T040_draft_correction", "T041_draft_regeneration", "T042_draft_version_history",
                        "T043_draft_export_pdf", "T044_draft_export_docx", "T045_draft_translate"]:
                results[tid] = {"tid": tid, "http_status": None, "error": "SKIPPED: no draft_id produced by T039 field collection"}
            _save_results()

        # ================= T046-T051: search modes =================
        search_specs = [
            ("T046_search_semantic", "semantic", "cheque dishonour punishment", {}),
            ("T047_search_keyword", "keyword", "Section 138 Negotiable Instruments Act", {}),
            ("T048_search_hybrid", "hybrid", "RTI application process", {}),
            ("T049_search_metadata", "metadata", "Right to Information Act", {"act_name": "Right to Information Act"}),
            ("T050_search_section", "section", "173", {}),
            ("T051_search_act", "act", "Negotiable Instruments Act", {}),
        ]
        for tid, mode, query, filters in search_specs:
            await call(client, tid, "POST", "/search", meta={"feature": "search_modes", "mode": mode},
                       json={"query": query, "mode": mode, "top_k": 5, "filters": filters}, headers=hdr_a, timeout=90)

        # ================= T052-T057: upload/OCR/doc-analysis =================
        txt_content = ("This agreement is made between Landlord Mr. Sharma and Tenant Mr. Verma for a "
                       "monthly rent of Rs. 18000, security deposit Rs. 54000, dated 01-02-2026.").encode("utf-8")
        r = await call(client, "T052_upload_txt", "POST", "/upload", meta={"feature": "upload"},
                        files={"file": ("rent_test.txt", txt_content, "text/plain")},
                        data={"session_id": state.get("lifecycle_session") or ""}, headers=hdr_a, timeout=60)
        state["uploaded_txt_id"] = (r.get("body") or {}).get("document_id")

        pdf_bytes = minimal_pdf(["SALE DEED FOR PROPERTY", "Consideration Rs 1200000 dated 05-03-2026"])
        r = await call(client, "T053_upload_pdf", "POST", "/upload", meta={"feature": "upload"},
                        files={"file": ("sale_deed_test.pdf", pdf_bytes, "application/pdf")},
                        data={"session_id": state.get("lifecycle_session") or ""}, headers=hdr_a, timeout=60)
        state["uploaded_pdf_id"] = (r.get("body") or {}).get("document_id")

        img_bytes = ocr_test_image()
        r = await call(client, "T054_upload_image_ocr", "POST", "/upload", meta={"feature": "upload_ocr"},
                        files={"file": ("agreement_scan_test.png", img_bytes, "image/png")},
                        data={"session_id": state.get("lifecycle_session") or ""}, headers=hdr_a, timeout=90)
        state["uploaded_img_id"] = (r.get("body") or {}).get("document_id")

        r = await call(client, "T055_upload_unsupported_ext", "POST", "/upload", meta={"feature": "upload_validation"},
                        files={"file": ("malware_test.exe", b"MZ\x90\x00fake", "application/octet-stream")},
                        data={"session_id": state.get("lifecycle_session") or ""}, headers=hdr_a, timeout=30)

        r = await call(client, "T056_upload_oversized", "POST", "/upload", meta={"feature": "upload_validation"},
                        files={"file": ("huge_test.txt", b"A" * (26 * 1024 * 1024), "text/plain")},
                        data={"session_id": state.get("lifecycle_session") or ""}, headers=hdr_a, timeout=90)

        r = await call(client, "T057_document_analysis", "POST", "/document-analysis", meta={"feature": "document_analysis"},
                        json={"text": txt_content.decode("utf-8"), "analysis_type": "general",
                              "session_id": state.get("lifecycle_session")}, headers=hdr_a, timeout=90)

        # ================= T058-T061: voice =================
        r = await call(client, "T058_voice_speak", "POST", "/voice/speak", meta={"feature": "voice_tts"},
                        json={"text": "Namaste, yeh ek test hai cheque bounce ke baare mein."}, timeout=60)
        audio_b64 = (r.get("body") or {}).get("audio_base64")

        wav_bytes = silent_wav(2)
        r = await call(client, "T059_voice_chat_vad_silence", "POST", "/voice/chat", meta={"feature": "voice_vad"},
                        files={"audio_file": ("silence_test.wav", wav_bytes, "audio/wav")},
                        data={"new_conversation": "true"}, headers=hdr_a, timeout=90)

        if audio_b64:
            import base64
            real_audio = base64.b64decode(audio_b64)
            r = await call(client, "T060_voice_chat_roundtrip", "POST", "/voice/chat", meta={"feature": "voice_stt_roundtrip"},
                            files={"audio_file": ("speech_test.wav", real_audio, "audio/wav")},
                            data={"new_conversation": "true"}, headers=hdr_a, timeout=90)
        else:
            results["T060_voice_chat_roundtrip"] = {"tid": "T060_voice_chat_roundtrip", "error": "SKIPPED: T058 produced no audio"}
            _save_results()

        r = await call(client, "T061_voice_draft_confirm_wrong_text", "POST", "/voice/drafts/nonexistent-draft-id/confirm",
                        meta={"feature": "voice_draft_confirm"},
                        json={"session_id": state.get("lifecycle_session") or "x", "confirmation_text": "yes okay"},
                        headers=hdr_a, timeout=30)

        _save_state()
        print("PHASE 2A (T033-T061) DONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
