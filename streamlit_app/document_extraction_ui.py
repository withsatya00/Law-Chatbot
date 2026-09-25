"""Private transcription preview and downloads, separate from shared KB ingestion."""

import base64
from pathlib import Path

import auth_client
import httpx
import streamlit as st


def render(state, api_base_url: str, session_id: str) -> None:
    owner = state.get(auth_client.USER_ID_KEY) or "anonymous"
    scope = (owner, session_id)
    if state.get("extraction_scope") != scope:
        state.pop("extraction_result", None)
        state["extraction_scope"] = scope
    with st.expander("Extract handwriting / document text"):
        st.caption("Upload a scan, photo or Word file. Get a structured preview, editable Word and text downloads.")
        st.caption("Handwriting is processed by the configured vision service. Unclear words are marked for review.")
        uploaded = st.file_uploader(
            "Document to transcribe", type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
            key=f"extract_file_{owner}_{session_id}",
        )
        if st.button("Extract text", disabled=uploaded is None, key="extract_document_button"):
            state.pop("extraction_result", None)
            try:
                with st.spinner("Reading the document and preserving its structure..."):
                    response = auth_client.request(
                        state, api_base_url, "POST", "/documents/extract", timeout=630,
                        files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")},
                    )
                if response.is_success:
                    state["extraction_result"] = response.json()
                else:
                    body = response.json()
                    error = body.get("error", {})
                    st.error(error.get("message", "Extraction failed. Try a clearer or smaller document.") if isinstance(error, dict) else "Extraction failed.")
            except (httpx.HTTPError, ValueError):
                st.error("The extraction service could not complete the request. Please retry.")
        result = state.get("extraction_result")
        if not result:
            return
        st.caption(f"Extracted file: {result['filename']}")
        for warning in result.get("warnings", []):
            st.caption(warning)
        for index, page in enumerate(result["pages"], 1):
            st.subheader(f"Page {page['page_number']}" if page.get("page_number") else f"Document section {index}")
            for warning in page.get("warnings", []):
                st.warning(warning)
            for block in page["blocks"]:
                if block["kind"] == "table":
                    st.table(block["rows"])
                elif block["kind"] == "heading":
                    # Display source text as text, never execute HTML or document instructions.
                    st.text(block["text"])
                    st.divider()
                else:
                    st.text(block["text"])
        stem = Path(result["filename"]).stem
        word_col, text_col = st.columns(2)
        with word_col:
            st.download_button(
                "Download editable Word", data=base64.b64decode(result["docx_base64"]),
                file_name=f"{stem}_extracted.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        with text_col:
            st.download_button(
                "Download text", data=result["text"].encode("utf-8"),
                file_name=f"{stem}_extracted.txt", mime="text/plain; charset=utf-8",
            )
