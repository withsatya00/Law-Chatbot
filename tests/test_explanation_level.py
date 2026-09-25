import pytest

from app.language.explanation_level import detect_explanation_level


@pytest.mark.parametrize(
    "message,expected_level",
    [
        ("Explain like I am 15", "student"),
        ("explain like I'm 8", "child"),
        ("explain this for a kid", "child"),
        ("explain professionally", "professional"),
        ("in legal jargon please", "professional"),
        ("explain simply", "student"),
        ("in simple terms please", "student"),
        ("What is FIR?", None),
        ("My landlord won't return my deposit", None),
    ],
)
def test_detect_explanation_level(message: str, expected_level: str | None) -> None:
    assert detect_explanation_level(message) == expected_level
