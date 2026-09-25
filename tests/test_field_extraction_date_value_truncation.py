"""Regression tests for two chained bugs found while re-verifying BUG-104
(QA session 2026-09-24), both in `DraftFieldExtractor._extract_labelled_
fields`'s value-boundary detection (`app/drafting/field_extraction.py`):

1. The numbered-list-prefix allowance in `next_label` (`\\d+[.)]\\s*`, meant
   for pasted forms like "1. Purpose", "2. Address") had no lower bound and
   no lookbehind, so it matched a 4-digit YEAR immediately followed by a
   sentence-ending period as if it were list numbering. "date of return
   memo is 10-08-2026. reason for dishonour is ..." matched `next_label`
   starting at "2026. reason for dishonour is", truncating the captured
   value to "10-08" (then, after a first narrowing to `\\d{1,2}`, to
   "10-08-20" -- the LAST two digits of the year still matched the same
   way). A `(?<!\\d)` lookbehind is the actual fix: it can never match the
   tail of a longer digit run, only a genuine, digit-unpreceded list marker.

2. Even once the full date text was captured ("10-08-2026."), the sentence's
   own trailing period stayed attached (deliberately, for the general case
   -- "ABC Electronics Pvt. Ltd." must keep its internal/trailing periods)
   and then failed date-format validation, so the field was reported as
   still missing despite having been correctly captured. Fixed narrowly for
   `field_type == "date"` fields only.

Live impact: a cheque-bounce draft's `dishonour_date` field (and, in the
same message, `cheque_number`, truncated to "4" instead of "445566" by the
same year-as-list-number misparse) could never be filled via a natural,
multi-field message -- only a message containing this one field in
isolation worked.
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


def test_date_value_is_not_truncated_by_a_following_labelled_field():
    template = get_template("cheque_bounce_notice")
    assert template is not None
    message = "date of return memo is 10-08-2026. reason for dishonour is insufficient funds."
    extracted = asyncio.run(_extractor_with_no_llm().extract(message, template))
    assert extracted.get("dishonour_date") == "10-08-2026"


def test_a_non_date_field_immediately_before_a_year_is_also_not_truncated():
    # The same "4-digit year misread as a list number" bug also truncated
    # cheque_number ("445566" -> "4") when cheque_date (ending in a year)
    # preceded it in the same message. A trailing "." is expected here (only
    # `date`-type fields strip it, same deliberate choice that keeps "ABC
    # Electronics Pvt. Ltd." intact) -- what matters is the DIGITS are no
    # longer truncated down to a single leading "4".
    template = get_template("cheque_bounce_notice")
    assert template is not None
    message = "cheque date is 01-08-2026. cheque number is 445566. place is Chennai."
    extracted = asyncio.run(_extractor_with_no_llm().extract(message, template))
    assert extracted.get("cheque_date") == "01-08-2026"
    assert extracted.get("cheque_number", "").rstrip(".") == "445566"


def test_genuine_numbered_form_labels_still_work_no_regression():
    template = get_template("cheque_bounce_notice")
    assert template is not None
    message = "1. applicant name is Ravi Kumar. 2. place is Chennai."
    extracted = asyncio.run(_extractor_with_no_llm().extract(message, template))
    # Trailing "." expected on the non-date field, same as above.
    assert extracted.get("applicant_name", "").rstrip(".") == "Ravi Kumar"
    assert extracted.get("place", "").rstrip(".") == "Chennai"


def test_textarea_value_still_keeps_its_own_internal_period_no_regression():
    # The reason `field_type == "date"` stripping is narrow, not general:
    # a company/entity name legitimately keeps a trailing "Ltd." period.
    template = get_template("cheque_bounce_notice")
    assert template is not None
    message = "respondent name is ABC Electronics Pvt. Ltd. respondent address is Chennai."
    extracted = asyncio.run(_extractor_with_no_llm().extract(message, template))
    assert extracted.get("respondent_name") == "ABC Electronics Pvt. Ltd."
