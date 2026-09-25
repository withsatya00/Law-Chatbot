FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime OS packages, grouped by the feature each one enables. Every one of
# these is exercised by `scripts/validate_environment.py`, which is the way to
# confirm the image is complete:
#
#   docker compose run --rm api python scripts/validate_environment.py
#
#   build-essential  - source builds for any wheel without a manylinux binary.
#   tesseract-ocr    - the OCR engine `pytesseract` shells out to.
#   clamav           - fail-closed malware scanning for autonomous KB downloads.
#   poppler-utils    - pdftoppm/pdfinfo, which `pdf2image` shells out to when
#                      OCR-ing a scanned PDF. Without it OCR returns EMPTY TEXT
#                      rather than an error, so its absence is silent.
#   libpango-1.0-0,  - WeasyPrint's native rendering stack. `import weasyprint`
#   libpangoft2-     succeeds without these and fails only at render time, so a
#   1.0-0,           missing one presents as "PDF export broken in production"
#   libharfbuzz0b,   long after the build. This image previously shipped none
#   libcairo2,       of them: PDF export -- the notarized-document download
#   libgdk-pixbuf-   included -- could not work in any container built from it.
#   2.0-0,
#   libffi8,
#   shared-mime-info,
#   fontconfig
#   fonts-noto-core, - glyph coverage for the scripts this app drafts in.
#   fonts-noto-      Without Devanagari/Bengali/Tamil/... faces, a Hindi PDF
#   core + Indic     renders as tofu boxes: the export "succeeds" and the
#                    document is unreadable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        tesseract-ocr \
        tesseract-ocr-hin \
        clamav \
        poppler-utils \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
        libharfbuzz0b \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi8 \
        shared-mime-info \
        fontconfig \
        fonts-noto-core \
        fonts-deva \
        fonts-beng \
        fonts-taml \
        fonts-telu \
        fonts-knda \
        fonts-mlym \
        fonts-guru \
        fonts-gujr \
        fonts-orya \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# Install CPU-only torch first so the project resolve does not pull the CUDA
# wheels (nvidia-cublas/cudnn/nccl/...), which are several GB and unusable here:
# no GPU is passed into these containers.
#
# Dependencies are installed from pyproject.toml alone, BEFORE the source is
# copied, so editing application code does not invalidate this layer.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install ".[dev]"

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY conftest.py ./
COPY config ./config

# Now that the source is present, install the project itself. `--no-deps`
# because the resolve above already placed every dependency; this step exists
# so `app` is a real installed package rather than merely importable because
# WORKDIR happens to be its parent.
RUN pip install --no-deps .

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir -p /app/storage /home/app/.cache/huggingface \
    && chown -R app:app /app/storage /home/app/.cache
ENV HF_HOME=/home/app/.cache/huggingface
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
