import pytest

from streamlit_app.upload_routing import upload_target


@pytest.mark.parametrize(
    "message",
    [
        "KB me file upload karo",
        "add this PDF to the knowledge base",
        "knowledgebase mein document dalo",
        "इसे KB में अपलोड करो",
    ],
)
def test_explicit_kb_upload_routes_admin_to_the_guarded_endpoint(message: str) -> None:
    assert upload_target(message, "admin") == "admin_kb"
    assert upload_target(message, "super_admin") == "admin_kb"


def test_non_admin_explicit_kb_upload_is_blocked_instead_of_becoming_private() -> None:
    assert upload_target("KB me file upload karo", "user") == "admin_required"
    assert upload_target("KB me file upload karo", "") == "admin_required"


@pytest.mark.parametrize(
    "message",
    [
        "review this contract",
        "summarise my PDF",
        "upload this for my case",
        "what is the knowledge base status?",
    ],
)
def test_ordinary_attachments_remain_private(message: str) -> None:
    assert upload_target(message, "admin") == "private"
