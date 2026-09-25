"""Regression test for the local-vector-scan coverage gap (QA session
2026-09-24, from `QA_REPORT_100Q_RETEST_20260924.md` section 7 items 2-3 and
6, T026/T072/T074): `MongoVectorStore._local_cosine_leg` runs
`collection.find(mongo_filter).limit(self.local_scan_limit)` with no sort --
per that field's own long-standing docstring in `app/core/config.py`, once
the corpus exceeds the limit, whichever documents were indexed LAST become
silently unreachable by this leg, not just slower to reach.

The corpus grew past the old default (20,000) to ~55,000 chunks, so roughly
the newest third of the corpus was unreachable by every local-scan query.
Live-reproduced against the real dev KB (55,070 chunks at the time): a
Hindi/Hinglish security-deposit question ("mera landlord mera security
deposit wapas nahi de raha hai") correctly expanded to the right vocabulary
and correctly passed `is_relevant_chunk`'s lexical gate on a genuinely
on-topic Tenancy Act chunk -- but only when that chunk happened to fall
inside the 20,000-document scanned window, which varied non-deterministically
between otherwise-identical calls (`.find().limit()` with no sort has no
ordering guarantee). Raising the limit to comfortably exceed the corpus size
makes the local fallback deterministically cover all of it, matching what
Atlas `$vectorSearch` already does for a deployed cluster.

This is a plain value-floor test (no DB access, mirrors the style of
`test_bug101_concurrency_fixes.py`'s config assertions) -- it only catches
someone silently lowering the constant back down, not corpus growth past
it. There is no live-KB-size assertion here on purpose: the corpus size is
real production/dev data that changes over time and isn't something this
test suite should depend on or gate CI on.
"""

from app.core.config import settings

# The corpus was 55,070 chunks when this fix landed (2026-09-24) -- the
# floor is set with headroom above that, not pinned to the exact count, so
# ordinary KB growth doesn't require touching this test.
_CORPUS_SIZE_AT_FIX_TIME = 55_070


def test_local_vector_scan_limit_exceeds_corpus_size_at_fix_time() -> None:
    assert settings.local_vector_scan_limit > _CORPUS_SIZE_AT_FIX_TIME


def test_local_vector_scan_limit_has_meaningful_headroom_above_the_old_cap() -> None:
    # The old cap (20,000) is exactly the value that was silently excluding
    # roughly a third of the corpus -- the new default must be a real fix,
    # not a token bump that would still leave most of the corpus unreachable.
    _OLD_CAP = 20_000
    assert settings.local_vector_scan_limit >= _OLD_CAP * 2
