from streamlit_app.kb_upload_status import status_note


def test_pending_upload_receipt_is_explicitly_temporary_and_trackable() -> None:
    note = status_note({"status": "pending", "staging_id": "stage-123"})

    assert "Queued" in note
    assert "stage-123" in note
    assert "refresh automatically" in note


def test_indexed_upload_reports_its_own_final_result() -> None:
    note = status_note(
        {"status": "indexed", "generated_filename": "Constitution_of_India.pdf", "chunks_indexed": 535}
    )

    assert note == "Indexed into the shared Knowledge Base as Constitution_of_India.pdf (535 chunks)."


def test_duplicate_is_not_described_as_pending() -> None:
    note = status_note({"status": "duplicate", "reason": "Matches Constitution_of_India.pdf."})

    assert "duplicate" in note
    assert "pending" not in note.lower()
