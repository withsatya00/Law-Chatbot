"""Regression test for BUG-104 (QA session 2026-09-24): the field-fill
candidate set only included a field's key with underscores replaced by
spaces ("dishonour date"), not the raw key ("dishonour_date") the API's own
`missing_fields` response actually shows the caller. A user or API
integrator who reasonably answers using the exact string the API told them
was missing could not fill the field at all.
"""

import asyncio
from unittest.mock import AsyncMock

from app.drafting.field_extraction import DraftFieldExtractor
from app.drafting.templates import get_template
from app.llm.base import LLMResponse


def _extractor_with_no_llm() -> DraftFieldExtractor:
    extractor = DraftFieldExtractor()
    extractor.llm.chat = AsyncMock(return_value=LLMResponse(content="", model="test", provider="test"))
    return extractor


def test_raw_underscored_field_key_fills_the_field() -> None:
    # Live-reproduced: `missing_fields` on a cheque_bounce_notice draft
    # reported "dishonour_date"; answering with that exact string (matching
    # the API's own output) previously matched nothing -- only the
    # differently-spelled human label ("date of return memo") worked.
    template = get_template("cheque_bounce_notice")
    assert template is not None
    extracted = asyncio.run(_extractor_with_no_llm().extract("dishonour_date is 10-08-2026", template))
    assert extracted.get("dishonour_date") == "10-08-2026"


def test_human_label_still_fills_the_field_no_regression() -> None:
    template = get_template("cheque_bounce_notice")
    assert template is not None
    extracted = asyncio.run(_extractor_with_no_llm().extract("the date of return memo is 10 August 2026", template))
    assert extracted.get("dishonour_date") == "10 August 2026"


def test_space_joined_key_form_still_fills_the_field_no_regression() -> None:
    template = get_template("cheque_bounce_notice")
    assert template is not None
    extracted = asyncio.run(_extractor_with_no_llm().extract("dishonour date is 10-08-2026", template))
    assert extracted.get("dishonour_date") == "10-08-2026"
