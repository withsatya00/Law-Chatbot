from scripts.kb_exact_deduplicate import unreferenced_canonical


def test_unreferenced_canonical_prefers_unsuffixed_name() -> None:
    assert unreferenced_canonical(["Act_3.pdf", "Act.pdf", "Act_2.pdf"]) == "Act.pdf"


def test_unreferenced_canonical_is_deterministic_for_descriptive_names() -> None:
    assert unreferenced_canonical(["Long_Misleading_Name.pdf", "Act.pdf"]) == "Act.pdf"
