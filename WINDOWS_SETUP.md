# Windows Host Setup

Everything in this file is **host-environment configuration only**. No application
logic, API, schema, prompt, or route was changed to make the project run here.

Verified on: Windows 11 Pro 26200 · Python 3.12.12 · NVIDIA Quadro RTX 3000 (6 GB,
driver 580.92, CUDA 13.0) · MongoDB 8.2.5 · Memurai/Redis 6379.

---

## 1. Start the app

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_api.ps1          # API      :8000
powershell -ExecutionPolicy Bypass -File scripts\run_streamlit.ps1    # console  :8501
powershell -ExecutionPolicy Bypass -File scripts\run_tests.ps1        # pytest
```

Use the launcher scripts rather than a bare `uvicorn app.main:app` / `pytest`.
They are thin wrappers around the stock README commands; all they add is
`scripts\_win_env.ps1`, which clears one variable that otherwise crashes the
interpreter (§2a) and fixes one PATH ordering problem that otherwise breaks PDF
export (§4). Without it `import app.main` dies and 14 test modules fail at
collection. The equivalent manual command is:

```powershell
Remove-Item Env:\SSLKEYLOGFILE -ErrorAction SilentlyContinue
$env:PATH = "C:\msys64\ucrt64\bin;C:\Program Files\Tesseract-OCR;$env:PATH"
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`_win_env.ps1` resolves Poppler's `bin` directory by pattern rather than by a
hard-coded version, so a `winget upgrade` no longer silently moves it out from
under the launcher. Its path probes are deliberately quiet: a directory the
current user cannot enumerate is one the script will not add, which is the same
outcome as it not existing, and emitting an "Access is denied" record for that
only trains you to ignore the launcher's output. **Nothing is hidden by that** —
a genuinely missing or broken dependency is reported, by name and with the
install command, by the validator below.

### Check the environment before starting

```powershell
.venv\Scripts\python.exe scripts\validate_environment.py
.venv\Scripts\python.exe scripts\validate_environment.py --json     # machine-readable
.venv\Scripts\python.exe scripts\validate_environment.py --strict   # fail on optional gaps too
powershell -ExecutionPolicy Bypass -File scripts\run_api.ps1 -Validate   # validate, then start
```

It reports one line per FEATURE, and probes the real capability rather than
inferring it: every required Python package is imported; Tesseract and Poppler
are located as executables and asked for a version; WeasyPrint's native stack is
exercised by actually rendering a PDF (importing it is not enough — see §3);
MongoDB, Redis and the configured LLM provider are pinged; every storage
directory is written to and read back; and a notarization QR plus one file per
export format are really generated.

Exit code `0` means every required check passed, `1` that at least one did not.
Optional features (OCR without Tesseract, say) are reported as `[warn]` and do
not fail the run unless `--strict` is given.

Secrets are never printed. API keys appear only as `configured` / `missing`, and
connection strings only as `scheme://host:port` with any embedded credentials
dropped — the output is safe to paste into a bug report.

---

## 2. Avast Antivirus — two separate breakages

Both are caused by Avast, both are worked around without touching `app/`.
**If you turn off Avast's Web Shield HTTPS scanning, both problems disappear and
neither workaround is needed.**

### 2a. `SSLKEYLOGFILE` crashes the interpreter

Avast injects `SSLKEYLOGFILE=\.\aswMonFltProxy\<id>` into every process it
monitors. Python's `ssl.create_default_context()` passes that to OpenSSL's
`BIO_new_file()`; this CPython build has no applink stub for a device path, so
OpenSSL calls `abort()`:

```
OPENSSL_Uplink(0x00007FF892B59C60,08): no OPENSSL_Applink
```

The process dies with no traceback. It is reached on `import app.main` (WeasyPrint
constructs a `URLFetcher` at import time) and on any outbound HTTPS call.

The variable is **not** in the User or Machine registry — Avast injects it at
process creation — so it can only be cleared per-process. That is exactly what
`scripts\run_api.ps1` and `scripts\run_streamlit.ps1` do. To confirm the fix:

```powershell
.venv\Scripts\python.exe -c "import ssl; ssl.create_default_context()"   # crashes
$env:SSLKEYLOGFILE=$null
.venv\Scripts\python.exe -c "import ssl; ssl.create_default_context()"   # fine
```

### 2b. Avast re-signs every TLS certificate

Web Shield terminates outbound HTTPS and re-issues certificates under
`CN=Avast Web/Mail Shield Root`. That root is trusted in the Windows store, so
`urllib` and browsers are happy — but `httpx` (every provider in `app/llm/`) and
`requests` (`huggingface_hub` model downloads) verify against **certifi's bundled
PEM**, which cannot contain a locally generated root:

```
httpx.ConnectError: [SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate
```

Fix — copies the already-OS-trusted root into this venv's certifi bundle:

```powershell
.venv\Scripts\python.exe scripts\windows_trust_avast_ca.py
```

Idempotent. **Re-run it after any `pip install -U certifi`**, which replaces
`cacert.pem` wholesale.

---

## 3. WeasyPrint / GTK3 (PDF export)

`app/drafting/export.py` needs Pango/Cairo/HarfBuzz/fontconfig. It probes three
locations; on this machine the third one hits:

```
C:\msys64\ucrt64\bin        <-- present and complete here
```

No action required, and no `WEASYPRINT_DLL_DIRECTORIES` needs setting. If MSYS2 is
ever removed, either reinstall the GTK stack there
(`pacman -S mingw-w64-ucrt-x86_64-pango`) or install the *GTK3 Runtime for
Windows* to `C:\Program Files\GTK3-Runtime Win64\bin`, which the same code probes
first. Hindi/Tamil/Telugu/Kannada/Bengali shaping comes from Windows' own
Nirmala UI via CSS `local()` — already installed.

**Import order matters, and PATH alone is not enough.** Python 3.8+ stopped
searching `PATH` when resolving a loaded DLL's own transitive dependencies, so
`import weasyprint` fails with

```
OSError: cannot load library 'C:\msys64\ucrt64\bin\libgobject-2.0-0.dll': error 0x7e
```

even with `C:\msys64\ucrt64\bin` first on `PATH`. `app/drafting/export.py`
registers that directory with `os.add_dll_directory()` at import time, which is
what makes it work. Anything reaching WeasyPrint outside the app — a script, a
notebook, a diagnostic — must therefore `import app.drafting.export` first:

```powershell
.venv\Scripts\python.exe -c "import app.drafting.export; from weasyprint import HTML; print(len(HTML(string='<p>ok</p>').write_pdf()))"
```

`scripts\validate_environment.py` does exactly this before probing, so its
verdict matches what the API will actually do.

---

## 4. OCR and PDF rasterisation

| Tool | Location | Why |
|---|---|---|
| Tesseract 5.4.0 | `C:\Program Files\Tesseract-OCR` (**deliberately NOT on the global User PATH** — see below) | `OCR_ENABLED=true` |
| `hin.traineddata` | `...\Tesseract-OCR\tessdata\hin.traineddata` | `OCR_LANGUAGE=eng+hin` |
| Poppler 25.07.0 | WinGet package dir (already on User PATH) | `pdf2image` |

Reinstall with:

```powershell
winget install --id UB-Mannheim.TesseractOCR --exact --source winget
# then drop hin.traineddata from https://github.com/tesseract-ocr/tessdata into tessdata\
```

Verify (via a shell that has sourced `scripts\_win_env.ps1`, so PATH is ordered
correctly):

```powershell
.venv\Scripts\python.exe -c "import pytesseract; print(pytesseract.get_languages())"
# -> ['eng', 'hin', 'osd']
```

### Do not add Tesseract to the global PATH

UB-Mannheim's Tesseract build ships its **own** copy of the GTK/Pango stack —
`libgobject-2.0-0.dll`, `libpango-1.0-0.dll`, `libharfbuzz-0.dll`,
`libfontconfig-1.dll` and friends — built against **mingw64**.
`C:\msys64\ucrt64\bin` (§3) ships those same library *names* built against
**ucrt64**.

WeasyPrint resolves each of those libraries independently. If Tesseract's
directory is searched first, the process ends up with Tesseract's `libgobject`
and MSYS2's `libpango` loaded side by side, and the import fails:

```
OSError: cannot load library 'C:\Program Files\Tesseract-OCR\libpango-1.0-0.dll': error 0x7f
```

`0x7f` is ERROR_PROC_NOT_FOUND — the two builds do not export a mutually
compatible ABI. This takes down `import app.drafting.export`, and therefore
`import app.main` and 14 test modules.

`scripts\_win_env.ps1` therefore places `C:\msys64\ucrt64\bin` **before**
`C:\Program Files\Tesseract-OCR` on PATH, so the whole GTK stack resolves from
one consistent build. Tesseract is unharmed by coming later: Windows searches an
`.exe`'s own directory before PATH when resolving its DLLs, so `tesseract.exe`
still loads its own copies.

---

## 5. Services

`.env` shipped pointing at the Docker Compose service names (`mongo:27017`,
`redis:6379`). For a native Windows run they are now `localhost`. `.env.example`
and `docker-compose.yml` are untouched, so `docker compose up` still works.

| Service | How it runs here |
|---|---|
| MongoDB 8.2.5 | Windows service `MongoDB`, `localhost:27017` |
| Redis | Memurai Developer (Redis-compatible Windows service), `localhost:6379` |

`VECTOR_SEARCH_BACKEND=local` — `scripts/create_indexes.py` logs a warning that it
cannot create the Atlas `$vectorSearch` index. That is expected on a local server
and the code falls back to cosine search, as documented in `README.md`.

### Restoring the database snapshot

`storage/backups/mongo_backup/` holds a `mongodump` of the project database. It can
be restored without installing the MongoDB Database Tools:

```powershell
.venv\Scripts\python.exe scripts\restore_mongo_backup.py --dry-run
.venv\Scripts\python.exe scripts\restore_mongo_backup.py
.venv\Scripts\python.exe scripts\create_indexes.py
```

The snapshot's `users` collection is **empty** — create a login with
`scripts\create_privileged_user.py`.

---

## 6. GPU

`EMBEDDING_DEVICE` is deliberately blank in `.env`; `EmbeddingProvider._detect_device()`
returns `"cuda"` whenever `torch.cuda.is_available()`. Do not set it — blank is the
auto-detect path and it resolves correctly here.

```
torch 2.12.1+cu130 · CUDA 13.0 · cuda.is_available() True · Quadro RTX 3000
arch_list includes sm_75 (this GPU's compute capability)
```

Only 6 GB of VRAM. `BAAI/bge-m3` at `INDEXING_BATCH_SIZE=32` is comfortable, but
avoid holding the embedding model and the cross-encoder reranker resident
alongside a local LLM.

---

## 7. Models

Downloaded lazily on first use into `%USERPROFILE%\.cache\huggingface`; not
pre-cached on this machine:

| Model | Size | Used by |
|---|---|---|
| `BAAI/bge-m3` | ~2.3 GB | `EMBEDDING_MODEL` → `app/rag/embeddings.py` |

This table previously also listed `cross-encoder/ms-marco-MiniLM-L-6-v2`,
pointing at `app/rag/hybrid_legal_retriever.py`. That module was never imported
anywhere and has been removed (Phase 1); no model download exists on the real
retrieval path today. `app/rag/reranker.py`, which is what actually reranks, is a
deterministic lexical/metadata scorer with no neural model and no download.

The first `/chat`, `/search`, or `/upload` request therefore stalls on a ~2.3 GB
download. To pre-fetch instead:

```powershell
$env:SSLKEYLOGFILE=$null
.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

Requires §2b to have been applied first.

---

## 8. Dependency reproducibility

`pyproject.toml` stays the source of truth for ranges. `requirements.txt` is a
pinned snapshot of the verified working environment — see its header for how to
regenerate it and for the CPU-vs-CUDA torch index.
