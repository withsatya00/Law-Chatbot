"""Detects the confirmed WeasyPrint PDF text-extraction corruption (D2) for
the Indic-script PDFs this app generates.

Root cause (confirmed live, 2026-09-23, against WeasyPrint 70.0 -- reproduced
by generating a real Hindi draft through `app.drafting.export.PdfDraftExporter`
and extracting it back with both `pypdf` and `pymupdf`):

  `weasyprint/draw/text.py`, the glyph-drawing loop inside `draw_first_line`
  (around the `# Create mapping between glyphs and Unicode codepoints.`
  comment), builds each font's `/ToUnicode` CMap entry like this:

      if glyph_id not in font.to_unicode:
          t1 = clusters[i]
          t2 = clusters[i+1] if ... else glyph_item.item.length
          font.to_unicode[glyph_id] = utf8_text[offset+t1:offset+t2].decode()

  This assumes each Pango glyph maps 1:1 onto a contiguous slice of the
  ORIGINAL (logical, storage-order) text. That assumption fails for exactly
  the class of script this app must support: Devanagari/Bengali/Gujarati/
  Gurmukhi/Odia/Tamil/Telugu/Kannada/Malayalam all have PRE-BASE dependent
  vowel signs (e.g. Devanagari ि, U+093F) that Unicode stores AFTER their
  consonant but HarfBuzz/Pango correctly SHAPE and draw BEFORE it -- so one
  Unicode "cluster" of 2+ codepoints becomes 2+ glyphs in visual (not
  logical) order sharing the SAME `log_cluster` value. For the first glyph of
  such a cluster, `t1 == t2` (or an otherwise wrong slice), so it gets an
  empty or truncated `to_unicode` entry; combined with `if glyph_id not in
  font.to_unicode` caching that entry FOREVER (even when the very same glyph
  ID is legitimately reused elsewhere in the document for different text),
  the wrong mapping can also leak into unrelated words. This is a WeasyPrint
  bug, not a font, CSS, or `app.drafting.export` bug -- it reproduces
  identically with the current single available system font (Nirmala UI) and
  is unrelated to `_font_face_css`/`_SCRIPT_FONTS` (see that module's own
  Part 53/57 history for the SEPARATE, already-fixed rendering-shaping bug
  this is often confused with: WeasyPrint/Pango render these PDFs correctly
  on screen; only extracting text back OUT of the PDF is affected).

Confirmed symptoms, from that live reproduction:
  * `pypdf`: a pre-base vowel sign silently DROPPED ("मोटरसाइकिल" -> "मोटरसाइकल").
  * `pymupdf`: a stray, unrelated character (observed: U+012E "Į", a Latin
    Extended-A letter that has no business anywhere in Hindi text) inserted
    in the same position instead -- the poisoned/reused glyph mapping.

Neither corrupts the PDF's VISUAL rendering (Pango still draws the correct
shaped glyphs) or this app's own re-ingestion of a PDF it generated
(`DocumentExtractionService.extract_path` rasterizes every PDF page and runs
OCR on it -- see its own docstring -- never `pypdf`/text-layer extraction).
It DOES affect anyone extracting text from the PDF outside this app: screen
readers, copy-paste, third-party search indexing.

CURRENT STATE (2026-09-23): MOSTLY FIXED for `pypdf`, PARTIALLY MITIGATED
for `pymupdf`. See `app.drafting._weasyprint_indic_cmap_patch` for the
shipped fix and its full history, including an earlier, superseded
strategy. Short version of what ships now: `_weasyprint_indic_cmap_patch.
apply_patch()` monkeypatches `weasyprint.draw.text.draw_first_line`
(version-guarded, fails safe) so that, within a multi-glyph Unicode
cluster, exactly one glyph (the CARRIER) gets the cluster's real text and
every other glyph gets a zero-width space (U+200B), never an empty string
and never a duplicate of the real text. Confirmed live, on the SAME
reproduction that first found this bug:

* FIXED, both libraries: the "Į"-class stray-out-of-script-character
  symptom is gone completely, including in the specific glyph-ID-collision
  case that caused it. `detect_corruption` below now finds nothing on a
  real document, pinned as a real, passing regression test (no longer
  `xfail`).
* FIXED, `pypdf` only: exact text recovery, for every word EXCEPT two
  separate, documented, unfixed collision classes:
    1. A glyph-ID collision specific to the pre-base-vowel CARRIER glyph
       (affects "मोटरसाइकिल", "प्राथमिकी" in the pinned test document) --
       needs the font-subsetting-level unique-glyph-ID fix described
       below, not attempted here.
    2. A separate, previously-undocumented stray-prefix bug (affects
       "दर्ज", "भवदीय" in the same document), confirmed unrelated to
       vowel-reordering or this patch's group logic at all -- reproduces
       identically whether or not this patch's carrier/ZWS logic even
       applies to that glyph. Root cause not investigated; out of this D2
       fix's scope.
  Neither case is silently missing (both were silent drops or stray
  characters before any fix shipped) or duplicated (this fix's predecessor
  duplicated ALL multi-glyph clusters; these two narrower cases are what
  remains after that predecessor was superseded) -- see
  `_weasyprint_indic_cmap_patch`'s "CURRENT STATE" section for the exact,
  current word-level detail.
* NOT FIXED, `pymupdf`: exact text recovery is deliberately NOT attempted
  or asserted. Confirmed live: `pymupdf` still duplicates a handful of
  otherwise-uncontested words under the shipped ZWS-based fix (a DIFFERENT
  set of words than `pypdf`'s two remaining gaps above) -- see
  `_weasyprint_indic_cmap_patch`'s "WHY ZWS INSTEAD OF DUPLICATION" section
  for why this is a deliberate, confirmed-safe trade (no stray characters
  either way) rather than an oversight.

Two further, more complete fixes were actually attempted, not just scoped
on paper -- both are recorded in full in `_weasyprint_indic_cmap_patch`'s
module docstring:
    1. `/ActualText` marked-content spans (the PDF spec's own mechanism for
       exactly this problem) -- shipped (zero rendering risk, confirmed
       correctly written into the content stream) but confirmed NOT read by
       either `pypdf` or `pymupdf` in any extraction mode, so it does not by
       itself change what this app's own tooling recovers.
    2. Giving the "swallowed" glyph a zero-width space instead of an empty
       string, to dodge `pymupdf`'s empty-destination bug without the
       duplicated-glyph-ID/font-subsetting work -- this IS what ships now
       (see above); it worked exactly as designed for `pypdf`, but
       `pymupdf` turned out to NOT simply concatenate `/ToUnicode` CMap
       entries in content-stream order the way this fix (and `pypdf`'s own
       behavior) assumes: several words with no collision at all came back
       duplicated in a different pattern anyway. This means `pymupdf`'s
       extraction has its own additional, undocumented fallback heuristics
       beyond the CMap data itself, which was not chased further.
  That second finding is the real reason the remaining, more invasive idea
  -- giving each glyph in a multi-glyph cluster its OWN glyph ID (a
  duplicated outline in the subsetted font, the same mechanism WeasyPrint's
  own `Font.get_unused_glyph_id` already uses for `.notdef` substitutes,
  requiring ALSO patching `weasyprint/pdf/fonts.py`'s font-subsetting step)
  -- was investigated in real depth (2026-09-23) and set aside rather than
  built, for TWO separate, now-confirmed reasons:

  ROOT-CAUSE UNIFICATION (the good news first): case 1 (glyph-ID collision)
  and case 2 (stray-prefix bug) are NOT two separate bugs -- live
  instrumentation of `_weasyprint_indic_cmap_patch`'s `font.to_unicode`
  cache (temporarily logging every write and every collision) proved both
  are the SAME mechanism: a CARRIER glyph's shape is reused elsewhere in
  the document for an ordinary, un-clustered occurrence of that same base
  shape (e.g. plain "द"), and whichever occurrence's write reaches
  `font.to_unicode` first wins -- confirmed directly: glyph ID 248 ("क"
  shape) collided with "मोटरसाइकिल"'s "कि" carrier; glyph ID 265 ("द"
  shape) was cached as "दि" by "दिनांक"'s EARLIER carrier, then wrongly
  inherited by "दर्ज"'s plain, un-clustered "द" LATER in the same
  document. One real fix (giving every carrier a glyph ID no plain
  occurrence can ever collide with) would therefore close BOTH gaps at
  once, not just case 1.

  WHY IT WASN'T BUILT ANYWAY (the hard blocker, confirmed empirically, not
  just estimated as risky on paper): giving a glyph a fresh ID requires a
  REAL duplicate outline at that ID in the EMBEDDED font, because this
  PDF's `CIDToGIDMap` is `/Identity` (CID = GID, direct lookup, no
  remapping) -- so the new ID's `/ToUnicode` entry alone is not enough,
  the font itself needs new, valid glyph data there. `Font.get_unused_
  glyph_id` (the ONLY existing allocator for IDs beyond the font's real
  glyph count) is built for `.notdef` PLACEHOLDERS specifically -- it
  relies on `HB_SUBSET_FLAGS_NOTDEF_OUTLINE` at subset time to give those
  IDs a blank/notdef box, not a copy of a real character. Getting a REAL
  duplicate outline into the final font was tested directly two ways:
    * Editing `Font.file_content` (raw font bytes, manipulable with
      `fontTools`) works in isolation -- confirmed by injecting a
      duplicate glyph into a standalone copy of Nirmala.ttc successfully.
    * But `Font._harfbuzz_subset` -- the active subsetting path in this
      environment (`harfbuzz_subset` installed, HarfBuzz >= 4.1.0) --
      subsets from `self.hb_face`, a HarfBuzz-native C object created
      once in `Font.__init__`, NOT from `self.file_content`. Confirmed
      directly: modifying `file_content` before calling `Font.subset()`
      had ZERO effect on the result -- `_harfbuzz_subset` rebuilds
      `file_content` entirely from `hb_face`, discarding any prior edit.
  This means there is no safe, verify-before-commit way to do this from
  the draw-time patch: the content stream has to be told which glyph ID
  to reference WHILE drawing, before the font is finalized, and there is
  no way to confirm at that point whether a real duplicate outline will
  actually exist for that ID later. Reaching in via HarfBuzz's own C-level
  face-builder API instead of `fontTools` was the only remaining path and
  was not attempted -- deeper, undocumented-for-this-purpose territory
  with the same fundamental problem: a bug or edge case in that step would
  not raise an exception or fail a test, it would silently draw the wrong
  or a blank character on a real, generated legal document, which
  `tests/test_pdf_text_extraction_integrity.py` (a text-extraction check,
  not a rendered-pixel check) would not catch. That failure mode, not
  merely "this is a bigger change," is why this was not built. Documented
  here in this much detail specifically so a future attempt does not have
  to rediscover the `hb_face`-vs-`file_content` dead end from scratch.

`detect_corruption`/`detect_missing_words` below are what make all of these
claims verifiable rather than assumed: automated, ongoing detection, so a
regression (or a future WeasyPrint upgrade / a properly-scoped complete fix)
can be noticed immediately (see `tests/test_pdf_text_extraction_integrity.py`
-- the out-of-script check is now a real assertion for both libraries, the
`pypdf` exact-text-recovery check stays `xfail(strict=True)` for the two
remaining word-level gaps above) instead of a claim either way going
unverified.
"""

import re
import unicodedata
from dataclasses import dataclass

# Primary Unicode block for each script this app renders PDFs in
# (`app.drafting.export._SCRIPT_FONTS`'s keys), used to flag any character
# that has no business appearing in that script's text at all.
_SCRIPT_BLOCKS: dict[str, tuple[int, int]] = {
    "devanagari": (0x0900, 0x097F),
    "bengali": (0x0980, 0x09FF),
    "gurmukhi": (0x0A00, 0x0A7F),
    "gujarati": (0x0A80, 0x0AFF),
    "odia": (0x0B00, 0x0B7F),
    "tamil": (0x0B80, 0x0BFF),
    "telugu": (0x0C00, 0x0C7F),
    "kannada": (0x0C80, 0x0CFF),
    "malayalam": (0x0D00, 0x0D7F),
    "perso_arabic": (0x0600, 0x06FF),
}

# Characters legitimately expected alongside script text regardless of
# script: ASCII (headings/labels fall back to English, e.g. "Details";
# digits; page-number footers), common punctuation/whitespace, the
# Devanagari danda "।"/"॥" used as sentence punctuation by several of the
# scripts above even where it isn't in their own block, and U+200B
# (zero-width space) -- deliberately written into a multi-glyph cluster's
# swallowed glyph by `_weasyprint_indic_cmap_patch`'s exact-recovery fix
# (see that module's docstring), invisible and semantically empty, not a
# corruption symptom.
_NEUTRAL_RE = re.compile(r"[\x00-\x7F।॥\u200b\s]")


@dataclass(frozen=True)
class CorruptionFinding:
    character: str
    codepoint: str
    context: str


def detect_corruption(text: str, script: str) -> list[CorruptionFinding]:
    """Flags characters in `text` that do not belong in `script`'s block and
    are not otherwise-expected neutral text (ASCII/punctuation/whitespace).

    Empty for genuinely clean text. Each finding names the offending
    character and a short window of surrounding text so a report is
    actionable without re-opening the PDF. Deliberately conservative (only
    flags characters clearly OUTSIDE where they could belong) -- this is a
    corruption DETECTOR, not a validator of correct shaping/ordering, since
    only character-set membership can be checked generically across scripts
    without re-implementing each script's own reordering rules.
    """
    block = _SCRIPT_BLOCKS.get(script)
    if block is None or not text:
        return []
    low, high = block
    findings: list[CorruptionFinding] = []
    for index, char in enumerate(text):
        if _NEUTRAL_RE.match(char):
            continue
        codepoint = ord(char)
        if low <= codepoint <= high:
            continue
        if unicodedata.category(char) in {"Zs", "Cc", "Po", "Pd", "Ps", "Pe"}:
            # Punctuation/format categories not already covered by the ASCII
            # fast path above (e.g. a right double quotation mark) --
            # legitimate regardless of script.
            continue
        context = text[max(0, index - 15):index + 16].replace("\n", " ")
        findings.append(CorruptionFinding(
            character=char, codepoint=f"U+{codepoint:04X}", context=context,
        ))
    return findings


_WORD_RE = re.compile(r"\S+", re.UNICODE)


@dataclass(frozen=True)
class MissingWordFinding:
    word: str


def detect_missing_words(extracted: str, source: str) -> list[MissingWordFinding]:
    """Flags a whitespace-delimited word from the known-correct `source` text
    that does not appear, verbatim, anywhere in `extracted` -- the OTHER
    confirmed symptom of the same bug (see module docstring): a pre-base
    vowel sign changes the WORD it was part of into something else --
    historically a silent drop ("मोटरसाइकिल" -> "मोटरसाइकल"); after
    `_weasyprint_indic_cmap_patch`'s fix, a duplicated syllable instead
    ("मोटरसाइकिल" -> "मोटरसाइकिकिल"). Both trip this check, deliberately --
    it asserts EXACT text recovery, which the shipped patch does not attempt
    (see that module's docstring for why), so this stays a documented,
    `xfail`-pinned gap rather than a claim of a complete fix.

    Deliberately a substring/word check, not a character-frequency count: the
    same bug's glyph-ID-reuse half can just as easily INSERT the poisoned
    mapping into some unrelated word elsewhere in the document (see
    `detect_corruption`'s docstring), which cancels out in a naive aggregate
    count of how many times each mark appears in the whole document -- a word
    dropped in one place and a stray mark gained in another can leave the
    document-wide total unchanged even though the text is corrupted in two
    places. Comparing whether each exact source word survives, wherever it
    ends up, does not have this blind spot.

    Only flags words containing at least one non-ASCII character -- an
    English/Hinglish label or heading (e.g. "Details") reflowing differently
    (justification, line wrap) is not this bug and would otherwise be a
    constant source of false positives unrelated to script shaping.
    """
    expected_words = {
        word for word in _WORD_RE.findall(source)
        if any(ord(char) > 127 for char in word)
    }
    # U+200B (zero-width space): deliberately written into a multi-glyph
    # cluster's swallowed glyph (see `_weasyprint_indic_cmap_patch`) so it
    # never carries real text -- invisible and semantically empty, so an
    # exact-word check must not let its mere PRESENCE, interrupting an
    # otherwise-exact word, register as a missing word.
    normalized_extracted = extracted.replace("\u200b", "")
    return [
        MissingWordFinding(word=word) for word in sorted(expected_words)
        if word not in normalized_extracted
    ]
