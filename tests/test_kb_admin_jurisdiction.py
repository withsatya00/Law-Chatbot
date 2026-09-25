"""Phase 1 "Jurisdiction-Aware Knowledge Base": the admin upload route's
jurisdiction-metadata parsing/validation (`app.api.admin._parse_jurisdiction_metadata`).

Exercised directly rather than through a FastAPI `TestClient` -- the function
under test is the exact seam the route calls before ever touching the
ingestion service, and testing it directly avoids standing up the full app
(auth deps, Mongo-backed routers) for what is a pure validation question.
"""

import json

import pytest

from app.api.admin import _parse_jurisdiction_metadata
from app.core.exceptions import BadRequestError
from app.rag import kb_jurisdiction as kj


def test_valid_metadata_json_is_normalized() -> None:
    fields = _parse_jurisdiction_metadata(
        json.dumps(
            {
                "issuing_level": "state",
                "applicability": "specific_states",
                "applicable_state_codes": ["MH"],
                "jurisdiction_source_type": "bare_act",
                "source_url": "https://example.gov.in/act",
                "verification_status": "verified",
                "verified_by": "admin@example.com",
            }
        )
    )
    assert fields["issuing_level"] == "state"
    assert fields["applicable_state_codes"] == ["MH"]
    assert fields["review_status"] == kj.REVIEW_APPROVED


def test_no_metadata_supplied_defaults_to_needs_review() -> None:
    """A new admin upload with NO `jurisdiction_metadata` field must still be
    gated -- omission is not a way to reach shared retrieval unreviewed."""
    fields = _parse_jurisdiction_metadata(None)
    assert fields["issuing_level"] == kj.ISSUING_LEVEL_UNKNOWN
    assert fields["review_status"] == kj.REVIEW_NEEDS_REVIEW


def test_invalid_state_code_is_rejected_as_bad_request() -> None:
    with pytest.raises(BadRequestError) as excinfo:
        _parse_jurisdiction_metadata(
            json.dumps({"applicability": "specific_states", "applicable_state_codes": ["XX"]})
        )
    assert any("XX" in issue for issue in excinfo.value.details["issues"])


def test_invalid_date_range_is_rejected_as_bad_request() -> None:
    with pytest.raises(BadRequestError) as excinfo:
        _parse_jurisdiction_metadata(
            json.dumps({"effective_from": "2022-06-01", "effective_to": "2021-01-01"})
        )
    assert any("effective_to" in issue for issue in excinfo.value.details["issues"])


def test_malformed_json_is_rejected() -> None:
    with pytest.raises(BadRequestError):
        _parse_jurisdiction_metadata("{not json")


def test_non_object_json_is_rejected() -> None:
    with pytest.raises(BadRequestError):
        _parse_jurisdiction_metadata("[1, 2, 3]")
