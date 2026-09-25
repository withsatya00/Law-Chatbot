"""The application's civil clock.

`date.today()` and `datetime.now()` return the *server process's* local date,
which is a bug for this product rather than a style issue: the assistant drafts
Indian legal documents, and the date printed on an affidavit, a legal notice or
a contract is the date in the jurisdiction the document is executed in, never
the date wherever the container happens to run. A UTC-hosted deployment serving
users in India is five and a half hours behind local civil time, so every draft
produced between 00:00 and 05:30 IST would carry the *previous* day's date --
on documents where the date is legally operative (limitation periods, notice
periods, the "within 30 days" windows the cyber-fraud workflow keys on).

`LEGAL_TIMEZONE` is configurable so a deployment serving another jurisdiction
can set it, but it is deliberately a single application-wide value rather than
a per-request one: a document has one execution date, and inferring it from a
browser locale would make the same draft render differently for two people
looking at the same record.
"""

import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import structlog

log = structlog.get_logger(__name__)

_DEFAULT_TIMEZONE = "Asia/Kolkata"
# Windows and slim Linux images ship no system tz database, so `tzdata` is a
# declared dependency. This fixed offset is the last-resort fallback for a host
# where it is somehow still unavailable: IST has no DST and has not changed
# since 1945, so a fixed +05:30 is exactly correct for the default zone -- but
# it is NOT correct for any other configured zone, hence the error log.
_IST_FALLBACK = timezone(timedelta(hours=5, minutes=30), "IST")


def _resolve_zone() -> ZoneInfo | timezone:
    name = os.getenv("LEGAL_TIMEZONE", _DEFAULT_TIMEZONE)
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        pass
    if name != _DEFAULT_TIMEZONE:
        try:
            # A typo'd or unsupported zone must not take the process down at
            # import time -- fall back to the jurisdiction default and say so.
            zone = ZoneInfo(_DEFAULT_TIMEZONE)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            pass
        else:
            log.warning("legal_timezone_unavailable", configured=name, using=_DEFAULT_TIMEZONE)
            return zone
    log.error("tz_database_unavailable", configured=name, using="UTC+05:30", hint="pip install tzdata")
    return _IST_FALLBACK


LEGAL_TIMEZONE = _resolve_zone()


def now() -> datetime:
    """The current instant, timezone-aware, in the jurisdiction's local time."""
    return datetime.now(LEGAL_TIMEZONE)


def today() -> date:
    """The current civil date in the jurisdiction -- what a document is dated."""
    return now().date()
