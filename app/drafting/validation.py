import re
from datetime import date, datetime

from app.core import clock
from app.drafting.localized_dates import parse_localized_date, resolve_relative_date
from app.drafting.templates.base import DraftTemplateDefinition
from app.drafting.wrapper_messages import msg

_PHONE_PATTERN = re.compile(r"^[6-9]\d{9}$")
_DATE_FORMATS = ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y", "%d %B %Y", "%d %b %Y", "%Y-%m-%d"]
_MIN_YEAR = 1900

# Field keys that hold a bare monetary figure. Every subject line and body
# phrase that renders one supplies its OWN currency mark -- the Hindi recovery
# notice subject is literally "₹{principal_amount} की धनवापसी/वसूली हेतु मांग"
# (see `fallback_phrases.localized_subject_template`), and the English one is
# "Demand for refund/recovery of Rs. {principal_amount}".
#
# So the value stored for these keys must be the NUMBER ALONE. It wasn't:
# `_extract_by_label_prefix` copies whatever followed the label verbatim, so a
# user typing "Principal Amount: ₹85,000" stored "₹85,000" and the rendered
# subject came out "₹₹85,000 की धनवापसी/वसूली हेतु मांग" -- on the first line
# of a legal demand notice. (The regex extractor already stripped the symbol,
# which is why this only showed up for labelled input.) Normalised here, in
# the one place every path passes through before generation.
_AMOUNT_FIELD_KEYS = frozenset({
    "principal_amount", "fraud_amount", "cheque_amount", "amount_paid",
    "security_deposit_amount", "dues_amount", "claim_amount", "compensation_amount",
})
# A leading currency mark, in the forms people actually type, plus any
# trailing separator before the digits begin.
_CURRENCY_PREFIX_PATTERN = re.compile(
    r"^\s*(?:₹|rs\.?|inr|rupees?|रु\.?|रुपये|रुपए|₨)\s*[.:\-]?\s*", re.IGNORECASE
)


def normalize_amount(value: str) -> str:
    """`value` with any leading currency mark removed, e.g. "₹85,000" ->
    "85,000". Repeated so "Rs. ₹85,000" also reduces to the bare figure.
    Returns the input unchanged when it carries no currency mark, and never
    strips the digits themselves.
    """
    cleaned = (value or "").strip()
    while True:
        stripped = _CURRENCY_PREFIX_PATTERN.sub("", cleaned, count=1).strip()
        if stripped == cleaned or not stripped:
            break
        cleaned = stripped
    return cleaned or (value or "").strip()


class DraftFieldValidator:
    """Best-effort sanity checks run before generation: impossible dates,
    malformed phone numbers, empty/digit-only names. Not a substitute for
    legal review -- it just catches obvious data-entry problems before
    they're baked into a generated draft.
    """

    def validate(
        self,
        template: DraftTemplateDefinition,
        fields: dict[str, str],
        language: str = "english",
        resolved_dates: list[tuple[str, str, str]] | None = None,
    ) -> list[tuple[str, str]]:
        """Validates `fields` in place and returns `(field_key, message)` issues.

        `fields` is MUTATED for one specific case: a relative date expression
        is replaced by the absolute date it resolves to. Callers that want to
        tell the user what was assumed pass a list as `resolved_dates` and get
        `(field_key, original_text, resolved_text)` appended for each one.
        """
        issues: list[tuple[str, str]] = []
        resolved_dates = resolved_dates if resolved_dates is not None else []
        for draft_field in template.all_fields():
            value = fields.get(draft_field.key, "").strip()
            if not value:
                continue
            if draft_field.key in _AMOUNT_FIELD_KEYS:
                # Rewritten in place, exactly like a resolved relative date
                # below: the stored value must be the bare figure so the one
                # currency mark in the subject/body template is the only one
                # that renders.
                normalized = normalize_amount(value)
                if normalized != value:
                    fields[draft_field.key] = normalized
                continue
            if draft_field.field_type == "tel":
                digits_only = value.replace(" ", "").replace("-", "")
                if not _PHONE_PATTERN.match(digits_only):
                    issues.append((
                        draft_field.key,
                        msg(
                            "invalid_mobile", language,
                            "'{value}' doesn't look like a valid 10-digit Indian mobile number.", value=value,
                        ),
                    ))
            elif draft_field.field_type == "date":
                # A relative expression ("kal", "yesterday", "3 din pehle",
                # "நேற்று") is how people actually report when something
                # happened to them. It used to be rejected outright, which
                # cost the user their incident date and sent a complaint out
                # with none -- so it is RESOLVED to an absolute date here and
                # rewritten in place, and the caller is told what was assumed
                # (see `resolved_dates` below) rather than the user being
                # blocked and asked again.
                relative = resolve_relative_date(value)
                if relative is not None:
                    resolved_value = relative.strftime("%d/%m/%Y")
                    fields[draft_field.key] = resolved_value
                    resolved_dates.append((draft_field.key, value, resolved_value))
                    continue
                parsed = self._parse_date(value)
                if parsed is None:
                    issues.append((
                        draft_field.key,
                        msg("invalid_date", language, "'{value}' doesn't look like a valid date.", value=value),
                    ))
                elif parsed.year < _MIN_YEAR or parsed.year > clock.today().year + 1:
                    issues.append((
                        draft_field.key,
                        msg(
                            "impossible_date", language,
                            "'{value}' looks like an impossible date -- please double-check it.", value=value,
                        ),
                    ))
            elif draft_field.key == "applicant_name" and (value.isdigit() or len(value) < 2):
                issues.append((draft_field.key, msg("invalid_name", language, "That doesn't look like a valid name.")))
        return issues

    def _parse_date(self, value: str) -> date | None:
        for fmt in _DATE_FORMATS:
            try:
                # DTZ007 (no %z) is correct-by-construction here and not
                # suppressible by adding a timezone: the input is a bare
                # calendar date the user typed into a form field ("15/07/2026"),
                # which carries no instant and no offset. Attaching one would
                # invent information the user never supplied; `.date()` on the
                # same line discards the placeholder midnight immediately, so no
                # naive datetime ever escapes this function.
                return datetime.strptime(value, fmt).date()  # noqa: DTZ007
            except ValueError:
                continue
        # `%B`/`%b` above only recognize English month names (locale-bound,
        # process-global) -- "15 जुलाई 2026" and similar localized dates are
        # recognized separately here, never rejected as an "invalid date"
        # just because the user wrote it in their own language.
        return parse_localized_date(value)
