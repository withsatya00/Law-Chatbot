"""Storage isolation, asserted rather than assumed.

A full suite run once left 17 extra files in the repository's real
`storage/uploads`. The redirect that prevents that lives in the root
`conftest.py`; these tests fail if it is removed, narrowed, or silently
stops applying.
"""

from pathlib import Path

import pytest

from app.core.config import settings
from conftest import REAL_STORAGE, REPO_ROOT, session_storage_root

_ISOLATED_SETTINGS = (
    "upload_storage_dir",
    "knowledge_base_dir",
    "kb_staging_dir",
    "kb_review_dir",
    "archive_dir",
    "operations_output_dir",
    "backup_dir",
    "draft_output_dir",
)


@pytest.mark.parametrize("setting_name", _ISOLATED_SETTINGS)
def test_every_storage_setting_points_outside_the_repository(setting_name: str) -> None:
    configured = Path(getattr(settings, setting_name)).resolve()
    assert not configured.is_relative_to(REPO_ROOT), (
        f"settings.{setting_name} resolves inside the repository at {configured}; "
        "tests must never write there."
    )
    # Under the conftest defaults this is the session temp root; a developer
    # who exported their own scratch path keeps it, so only the
    # "outside the repository" half of the invariant is universal.


@pytest.mark.parametrize("setting_name", _ISOLATED_SETTINGS)
def test_the_conftest_defaults_point_at_the_session_temp_root(setting_name: str) -> None:
    configured = Path(getattr(settings, setting_name)).resolve()
    root = session_storage_root().resolve()
    if not configured.is_relative_to(root):
        pytest.skip(f"{setting_name} overridden in the environment; the outside-the-repo guard still applies")
    assert configured.is_relative_to(root)


def test_a_write_through_the_settings_path_lands_in_the_temp_tree() -> None:
    """The redirect is real, not merely declared: a file written the way the
    upload service writes one must appear under the temp root."""
    target = Path(settings.upload_storage_dir) / "isolation-probe.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("probe", encoding="utf-8")
    try:
        assert target.exists()
        assert not target.resolve().is_relative_to(REAL_STORAGE)
    finally:
        target.unlink(missing_ok=True)


def test_the_real_storage_tree_is_not_the_configured_tree() -> None:
    """Guard against a future default that quietly points back at `storage/`."""
    assert REAL_STORAGE.exists(), "the real storage tree should still exist; only writes are redirected"
    assert Path(settings.upload_storage_dir).resolve() != (REAL_STORAGE / "uploads").resolve()
