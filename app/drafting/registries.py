"""Versioned language, formatting and jurisdiction configuration registries."""

from __future__ import annotations

import builtins
from dataclasses import asdict, dataclass, field
from typing import Any, cast


@dataclass(frozen=True)
class LanguageProfile:
    language_code: str
    language_name: str
    native_name: str
    default_script: str
    script_variants: tuple[str, ...]
    direction: str = "ltr"
    locale: str = ""
    ocr_support: str = "provider_dependent"
    handwritten_ocr_support: str = "experimental"
    legal_drafting_support: str = "partial"
    translation_support: str = "provider_dependent"
    transliteration_support: str = "provider_dependent"
    font_profile: str = "indic_default"


@dataclass(frozen=True)
class FormattingProfile:
    profile_id: str
    version: str
    paper_size: str = "A4"
    orientation: str = "portrait"
    margin_left_cm: float = 2.54
    margin_right_cm: float = 2.54
    margin_top_cm: float = 2.54
    margin_bottom_cm: float = 2.54
    font_family: str = "Noto Serif"
    font_size_pt: float = 12
    heading_size_pt: float = 14
    line_spacing: float = 1.5
    paragraph_spacing_pt: float = 6
    alignment: str = "justify"
    page_numbering: str = "page_x_of_y"
    duplex: bool = False
    effective_from: str | None = None
    effective_to: str | None = None
    source_reference: str = ""
    verification_status: str = "review_required"
    script_font_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JurisdictionProfile:
    profile_id: str
    country: str
    state: str = ""
    district: str = ""
    forum_type: str = ""
    formatting_profile: str = "generic_legal_document"
    require_verification: bool = False
    require_affidavit: bool = False
    version: str = "1.0"
    verification_status: str = "review_required"


class VersionedRegistry[T]:
    def __init__(self, key_name: str) -> None:
        self._key_name = key_name
        self._items: dict[str, T] = {}

    def register(self, item: T) -> None:
        key = str(getattr(item, self._key_name))
        if key in self._items:
            raise ValueError(f"Duplicate registry entry: {key}")
        self._items[key] = item

    def get(self, key: str) -> T | None:
        return self._items.get(key)

    def list(self) -> builtins.list[T]:
        return list(self._items.values())

    def serialized(self) -> builtins.list[dict[str, object]]:
        return [asdict(cast(Any, item)) for item in self.list()]


language_registry: VersionedRegistry[LanguageProfile] = VersionedRegistry("language_code")
formatting_profile_registry: VersionedRegistry[FormattingProfile] = VersionedRegistry("profile_id")
jurisdiction_registry: VersionedRegistry[JurisdictionProfile] = VersionedRegistry("profile_id")


def _seed() -> None:
    languages = (
        ("en", "English", "English", "Latin", ("Latin",), "ltr", "en-IN", "full"),
        ("as", "Assamese", "অসমীয়া", "Bengali", ("Bengali",), "ltr", "as-IN", "partial"),
        ("bn", "Bengali", "বাংলা", "Bengali", ("Bengali",), "ltr", "bn-IN", "good"),
        ("brx", "Bodo", "बड़ो", "Devanagari", ("Devanagari",), "ltr", "brx-IN", "partial"),
        ("doi", "Dogri", "डोगरी", "Devanagari", ("Devanagari",), "ltr", "doi-IN", "partial"),
        ("gu", "Gujarati", "ગુજરાતી", "Gujarati", ("Gujarati",), "ltr", "gu-IN", "good"),
        ("hi", "Hindi", "हिन्दी", "Devanagari", ("Devanagari", "Latin"), "ltr", "hi-IN", "full"),
        ("kn", "Kannada", "ಕನ್ನಡ", "Kannada", ("Kannada",), "ltr", "kn-IN", "good"),
        ("ks", "Kashmiri", "کٲشُر", "Perso-Arabic", ("Perso-Arabic", "Devanagari"), "rtl", "ks-IN", "experimental"),
        ("kok", "Konkani", "कोंकणी", "Devanagari", ("Devanagari", "Latin", "Kannada"), "ltr", "kok-IN", "partial"),
        ("ml", "Malayalam", "മലയാളം", "Malayalam", ("Malayalam",), "ltr", "ml-IN", "good"),
        ("mni", "Manipuri", "ꯃꯤꯇꯩ ꯂꯣꯟ", "Meetei Mayek", ("Meetei Mayek", "Bengali"), "ltr", "mni-IN", "experimental"),
        ("mr", "Marathi", "मराठी", "Devanagari", ("Devanagari",), "ltr", "mr-IN", "good"),
        ("mai", "Maithili", "मैथिली", "Devanagari", ("Devanagari", "Tirhuta"), "ltr", "mai-IN", "partial"),
        ("ne", "Nepali", "नेपाली", "Devanagari", ("Devanagari",), "ltr", "ne-IN", "partial"),
        ("or", "Odia", "ଓଡ଼ିଆ", "Odia", ("Odia",), "ltr", "or-IN", "good"),
        ("pa", "Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi", ("Gurmukhi",), "ltr", "pa-IN", "good"),
        ("sa", "Sanskrit", "संस्कृतम्", "Devanagari", ("Devanagari",), "ltr", "sa-IN", "partial"),
        ("sat", "Santali", "ᱥᱟᱱᱛᱟᱲᱤ", "Ol Chiki", ("Ol Chiki", "Devanagari", "Bengali"), "ltr", "sat-IN", "experimental"),
        ("sd", "Sindhi", "سنڌي", "Perso-Arabic", ("Perso-Arabic", "Devanagari"), "rtl", "sd-IN", "experimental"),
        ("ta", "Tamil", "தமிழ்", "Tamil", ("Tamil",), "ltr", "ta-IN", "good"),
        ("te", "Telugu", "తెలుగు", "Telugu", ("Telugu",), "ltr", "te-IN", "good"),
        ("ur", "Urdu", "اردو", "Perso-Arabic", ("Perso-Arabic",), "rtl", "ur-IN", "good"),
    )
    for code, name, native, script, variants, direction, locale, drafting in languages:
        language_registry.register(LanguageProfile(
            code, name, native, script, variants, direction, locale,
            legal_drafting_support=drafting,
            font_profile="rtl_default" if direction == "rtl" else "indic_default",
        ))

    formatting_profile_registry.register(FormattingProfile(
        profile_id="generic_legal_document", version="1.0",
        script_font_map={"Devanagari": "Noto Serif Devanagari", "Perso-Arabic": "Noto Nastaliq Urdu"},
    ))
    formatting_profile_registry.register(FormattingProfile(
        profile_id="delhi_courts_2022", version="1.0", margin_left_cm=4, margin_right_cm=4,
        margin_top_cm=2, margin_bottom_cm=2, font_family="Times New Roman",
        font_size_pt=14, heading_size_pt=14, line_spacing=1.5, duplex=True,
        effective_from="2022-11-01", verification_status="source_review_required",
    ))
    jurisdiction_registry.register(JurisdictionProfile(
        profile_id="generic_india", country="IN", verification_status="review_required",
    ))
    jurisdiction_registry.register(JurisdictionProfile(
        profile_id="delhi_district_courts", country="IN", state="Delhi",
        forum_type="district_court", formatting_profile="delhi_courts_2022",
        require_verification=True, require_affidavit=True,
        verification_status="source_review_required",
    ))


_seed()
