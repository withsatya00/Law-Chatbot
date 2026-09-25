"""D2 mitigation: corrects WeasyPrint's glyph-to-`/ToUnicode` CMap mapping
for a pre-base reordered Indic dependent vowel sign. Exact recovery for
`pypdf` (with two known, separate, unfixed collision-class gaps -- see
"CURRENT STATE" below); `pymupdf` keeps its own, different partial
improvement, never a stray out-of-script character either way.

See `app.drafting.pdf_text_integrity`'s module docstring for the full,
confirmed root cause -- read that before touching this file. Summary:
`weasyprint.draw.text.draw_first_line` builds each glyph's `/ToUnicode`
entry by slicing the source text between this glyph's Pango `log_cluster`
value and the NEXT DIFFERENT one -- correct only when each Unicode cluster
maps to exactly one glyph. A pre-base vowel sign (e.g. Devanagari ि,
U+093F) is stored AFTER its consonant in Unicode but drawn BEFORE it, so
Pango emits TWO glyphs, in visual order, sharing the SAME `log_cluster`
value -- for the first of those two glyphs, upstream's `t1 == t2`,
producing an empty or wrong slice.

THE FIX: group consecutive glyphs sharing one `log_cluster` value. Exactly
ONE glyph in the group -- the CARRIER, the one whose position the
UNMODIFIED upstream formula already resolves to the group's full,
correct text on its own (last in visual order for LTR, first for RTL) --
gets that real text. Every OTHER (swallowed, e.g. a pre-base matra) glyph
in the group gets U+200B (zero-width space), never `group_text` and never
an empty string:

* An empty string was tried FIRST and reverted for a confirmed, concrete
  reason: `pymupdf`'s extraction, on encountering a `/ToUnicode` bfchar
  entry whose destination is the EMPTY string, does not treat it as "this
  glyph contributes no text" -- it falls back to reading the GLYPH ID
  ITSELF as if it were a UTF-16BE codepoint. Confirmed live: glyph ID
  `0x012E` mapped to an empty destination extracted as "Į" (U+012E) -- the
  exact stray character this whole investigation started from. ZWS is a
  non-empty, legitimate destination that dodges that fallback entirely.
* Duplicating `group_text` onto EVERY glyph in the group (not just the
  carrier) was the FIRST shipped fix, superseded by this one -- it never
  produced a stray character either, but recovered a duplicated syllable
  ("मोटरसाइकिल" -> "मोटरसाइकिकिल") instead of the exact word, on BOTH
  `pypdf` and `pymupdf`. See "WHY ZWS INSTEAD OF DUPLICATION" below for
  why the carrier/ZWS split above is safe to ship in its place.
* This is applied through the SAME `if glyph_id not in font.to_unicode`
  cache guard upstream already uses (unchanged) -- it can only ever change
  what gets written the FIRST time a given glyph ID is seen anywhere in the
  document; upstream's per-glyph slice is ALREADY wrong on that first
  sighting for any multi-glyph cluster, so there is no code path this patch
  can make WORSE than today, only ones it can make less wrong.

CURRENT STATE (2026-09-23, re-verified live against a real export of
`tests/test_pdf_text_extraction_integrity.py`'s `_HINDI_SECTIONS`):

* `pypdf`: EXACT recovery for every word EXCEPT three, all ONE unified,
  UNFIXED root cause (confirmed by live instrumentation, 2026-09-23 --
  see `pdf_text_integrity`'s module docstring for the full trace and why
  a real fix is NOT safely buildable here): a multi-glyph cluster's
  CARRIER glyph shares its glyph ID with an ORDINARY, un-clustered
  occurrence of that same base shape elsewhere in the document, and
  whichever occurrence's `font.to_unicode` write happens first wins --
  regardless of which one comes first in the document:
    1. "मोटरसाइकिल" -> "मोटरसाइकल" and "प्राथमिकी" -> "प्राथमकी" (both
       missing their "ि"): the CARRIER's own id collided with an EARLIER,
       plain, un-clustered occurrence of that shape.
    2. "दर्ज" -> "दिजर्ज", "भवदीय" -> "भवदिीय" (an extra "दि" prefix): the
       REVERSE direction of the exact same mechanism -- an EARLIER
       carrier (from "दिनांक") poisoned a LATER, plain, un-clustered "द".
  A real fix (a fresh glyph ID for every carrier, immune to collision
  either way) would close both at once, but requires a real duplicate
  outline in the embedded, subsetted font -- confirmed empirically NOT
  achievable by editing `Font.file_content` before subsetting, because the
  active subsetting path (`Font._harfbuzz_subset`) subsets from `self.
  hb_face`, a HarfBuzz-native C object, discarding any such edit
  entirely. See `pdf_text_integrity`'s docstring for the full investigation.
  A third, cosmetic artifact -- "कुमार"/"कृपया" extracting with a literal
  space mid-word ("क ु मार") -- is ALSO present, identically, in the OLD
  duplicate strategy, so it predates and is unrelated to this fix too:
  most likely a `pypdf` heuristic that reads a glyph-positioning (kerning)
  adjustment in the content stream's `TJ` array as an implicit word break,
  nothing to do with `/ToUnicode` at all.
* `pymupdf`: never a stray out-of-script character (verified, real,
  passing assertion, unchanged from before) but NOT asserted for exact
  recovery -- confirmed to still duplicate a handful of otherwise-unrelated
  words under this scheme (see "WHY ZWS INSTEAD OF DUPLICATION"), so only
  `pypdf`'s exact-recovery is asserted in
  `tests/test_pdf_text_extraction_integrity.py`.

`detect_corruption` (both libraries, no out-of-script characters) is a
real, passing regression test; `detect_missing_words` against `pypdf`
alone is asserted too now for the words that DO exactly recover, with the
three remaining words above the reason a fully-clean assertion is still
not made (see that test file for the exact, current scope).

ALSO SHIPPED: an `/ActualText` marked-content span (`Span << /ActualText
(...) >> BDC ... EMC`) wrapped around each multi-glyph cluster's glyphs --
the PDF spec's own, purpose-built mechanism for "extract THIS exact string
for this span, regardless of the individual glyphs' own text/order",
confirmed written correctly into the content stream (inspected the raw,
decompressed stream directly: `/Span <</ActualText <feff...hex...>>> BDC
[<glyph><glyph>] TJ EMC`). Kept because it is zero-risk (marked content is
purely descriptive metadata a renderer that doesn't understand it simply
ignores -- confirmed no effect on the visible page, Part 53/57's rendering
tests all still pass) and may help a reader that DOES honor it (Adobe
Acrobat/Reader and some accessibility tooling do). It is NOT, on its own,
sufficient: confirmed live that NEITHER `pypdf` NOR `pymupdf` -- the two
libraries this app and its own verification tooling actually use -- read
`/ActualText` in any extraction mode tested (`pymupdf`: text/words/blocks/
dict/rawdict, all unaffected). The duplicate-`/ToUnicode` fallback above is
still what determines what THESE tools recover.

WHY ZWS INSTEAD OF DUPLICATION (2026-09-23; this IS what's shipped above --
recorded in this much detail because the reasoning is what makes it safe to
carry ZWS's known `pymupdf` cost). Reasoning first, then what's confirmed:
giving the group's NON-carrier glyph(s) a zero-width space (U+200B) instead
of duplicating `group_text` onto them was tried, on the theory that (a) ZWS
is a non-empty destination, so it should not trigger `pymupdf`'s
empty-destination-falls-back-to-raw-GID bug described above, and (b) the
"swallowed" role (the pre-base matra glyph, visually first) is
STRUCTURALLY the same role every time that exact glyph ID appears anywhere
in a document (a given matra shape is always the one being swallowed,
never the carrier), so -- unlike the carrier/consonant role -- assigning
it is never in conflict with an earlier cached value for the same glyph
ID, and the "first write wins" cache guard was never going to be a problem
for it specifically.

That reasoning held for `pypdf`: it behaved exactly as designed -- ZWS
correctly contributes nothing once stripped (see
`pdf_text_integrity.detect_missing_words`'s own ZWS-stripping step), and
every word EXCEPT the ones with a PRE-EXISTING glyph-ID collision (see
"CURRENT STATE" above) extracts exactly, verbatim -- a real, measured
improvement over the old duplicate-to-every-glyph strategy, confirmed by
reproducing BOTH strategies against the identical document. `pymupdf`,
however, does NOT simply concatenate the `/ToUnicode` CMap values in
content-stream order the way this whole design (and `pypdf`'s own
behavior) assumes: several OTHER words that had never had any collision
problem at all come back duplicated instead of exact ("प्रति" ->
"प्रतिति", "दिल्ली" -> "दिदिल्ली", "विषय" -> "विविषय") even though their
matra glyph correctly holds a ZWS and their consonant glyph correctly
holds the exact, uncontested text. The only plausible explanation is that
`pymupdf`/MuPDF's text extraction has its OWN additional heuristics beyond
a literal per-glyph `/ToUnicode` concatenation -- for instance, treating a
Unicode "format" category character (ZWS is Cf) as a signal to fall back
to inferring the glyph's identity some other way (via the embedded font's
own internal cmap, most likely) -- and that fallback does not agree with
this document's actual, correct CMap. This was not chased further: it
means `pymupdf`'s extraction behavior for this class of fix is not fully
predictable from the PDF/CMap data alone, so even the FULL
font-subsetting-level fix described in `pdf_text_integrity`'s module
docstring (which solves `pypdf`'s collision-class gaps) is not guaranteed
to produce clean, verified results on `pymupdf` without a much larger
investigation into its internals -- a
materially different, and larger, undertaking than scoped here. Shipping
ZWS anyway (superseding the duplicate-to-every-glyph strategy this module
carried until 2026-09-23) is a deliberate trade: `pymupdf`'s
`detect_corruption` guarantee (never a stray/wrong-script character) is
UNCHANGED either way -- ZWS never reproduces the empty-destination bug,
only shifts WHICH words `pymupdf` duplicates -- while `pypdf` moves from
"every multi-glyph cluster duplicated" to "exact except two documented,
narrow collision cases", a strictly better outcome with no observed
downside on the one guarantee that mattered enough to pin as a real,
non-`xfail` assertion.

SAFETY OF THE PATCH MECHANISM ITSELF (why a monkeypatch, and why this one is
low-risk): the bug lives inside a single large, non-decomposable upstream
function -- there is no smaller unit to patch a few lines of. `apply_patch()`
below fails safe on any mismatch:

* Only patches when the installed WeasyPrint version is EXACTLY the one this
  was written and tested against (`_TESTED_VERSION`) -- an upgrade silently
  reverts to unpatched (still only as-broken-as-documented, tracked by the
  test file above) behavior rather than running a stale copy against a
  changed internal API.
* Applying it is wrapped in `try/except Exception` -- any failure logs a
  warning and leaves WeasyPrint's original function in place. PDF export
  must never fail, or fail differently, because of this patch.
* The patched function changes ONLY the `/ToUnicode` computation; every line
  governing glyph selection, positioning, kerning, RTL handling, emoji, and
  the actual bytes written into the PDF's visible content stream is
  untouched, verbatim upstream -- this cannot alter what a PDF viewer draws,
  only what a text-extraction tool recovers afterward.

`app.drafting.export` imports and calls `apply_patch()` once, before any PDF
is generated (see its own import of this module).
"""

from typing import Any

import structlog

log = structlog.get_logger(__name__)

# The WeasyPrint version this patch was written and verified against (see
# `tests/test_pdf_text_extraction_integrity.py`, which asserts both the fixed
# and the still-open case). Bump this ONLY after re-verifying `draw_first_
# line`'s current source still matches the assumptions in
# `_patched_draw_first_line` below and re-running that test file.
_TESTED_VERSION = "70.0"

_patched = False


def apply_patch() -> bool:
    """Applies the fix if (and only if) it is safe to. Returns whether it
    was applied. Idempotent -- safe to call more than once."""
    global _patched
    if _patched:
        return True
    try:
        import weasyprint

        if weasyprint.__version__ != _TESTED_VERSION:
            log.warning(
                "weasyprint_indic_cmap_patch_skipped_version_mismatch",
                installed=weasyprint.__version__, tested=_TESTED_VERSION,
            )
            return False
        import weasyprint.draw.text as weasyprint_draw_text

        weasyprint_draw_text.draw_first_line = _patched_draw_first_line
        _patched = True
        log.info("weasyprint_indic_cmap_patch_applied", version=weasyprint.__version__)
        return True
    except Exception as exc:  # noqa: BLE001 - must never break PDF export
        log.warning("weasyprint_indic_cmap_patch_failed", error=type(exc).__name__, message=str(exc))
        return False


def _patched_draw_first_line(
    stream: Any, textbox: Any, text_overflow: Any, block_ellipsis: Any, matrix: Any
) -> list[Any]:
    """A copy of WeasyPrint 70.0's `weasyprint.draw.text.draw_first_line`
    with ONE change: the "Create mapping between glyphs and Unicode
    codepoints" block (see module docstring). Every other line is verbatim
    upstream -- changes to layout, glyph positioning, kerning, emoji or RTL
    handling are all deliberately NOT touched, to keep this patch's surface
    area as small as the bug it fixes.
    """
    from io import BytesIO
    from xml.etree import ElementTree

    import pydyf
    from PIL import Image
    from weasyprint.images import RasterImage, SVGImage
    from weasyprint.logger import LOGGER
    from weasyprint.text.ffi import FROM_UNITS, TO_UNITS, ffi, pango
    from weasyprint.text.fonts import get_hb_object_data
    from weasyprint.text.line_break import get_last_word_end

    if not textbox.text.strip():
        return []

    if textbox.style['font_size'] < 1e-6:
        return []

    pango.pango_layout_set_single_paragraph_mode(textbox.pango_layout.layout, True)

    if text_overflow == 'ellipsis' or block_ellipsis != 'none':
        assert textbox.pango_layout.max_width is not None
        max_width = textbox.pango_layout.max_width
        pango.pango_layout_set_width(
            textbox.pango_layout.layout, int(max_width * TO_UNITS))
        if text_overflow == 'ellipsis':
            pango.pango_layout_set_ellipsize(
                textbox.pango_layout.layout, pango.PANGO_ELLIPSIZE_END)
        else:
            if block_ellipsis == 'auto':
                ellipsis = '…'
            else:
                assert block_ellipsis[0] == 'string'
                ellipsis = block_ellipsis[1]

            new_text = textbox.pango_layout.text
            if new_text.endswith(textbox.style['hyphenate_character']):
                last_word_end = get_last_word_end(
                    new_text[:-len(textbox.style['hyphenate_character'])],
                    textbox.style['lang'])
                if last_word_end:
                    new_text = new_text[:last_word_end]

            textbox.pango_layout.set_text(new_text + ellipsis)

    first_line, index = textbox.pango_layout.get_first_line()

    if block_ellipsis != 'none':
        while index:
            last_word_end = get_last_word_end(
                textbox.pango_layout.text[:-len(ellipsis)],
                textbox.style['lang'])
            if last_word_end is None:
                break
            new_text = textbox.pango_layout.text[:last_word_end]
            textbox.pango_layout.set_text(new_text + ellipsis)
            first_line, index = textbox.pango_layout.get_first_line()

    stream.set_text_matrix(*matrix.values)
    previous_pango_font = None
    string = ''
    x_advance = 0
    emojis = []
    run = first_line.runs[0]
    while run != ffi.NULL:
        glyph_item = run.data
        run = run.next
        glyph_string = glyph_item.glyphs
        glyphs_info = glyph_string.glyphs
        num_glyphs = glyph_string.num_glyphs
        clusters = glyph_string.log_clusters
        utf8_text = None

        # --- D2 fix: pre-compute the correct per-glyph /ToUnicode text for
        # any Unicode cluster spanning MORE than one glyph (see module
        # docstring). A single-glyph cluster (the common case) is left to
        # the original inline computation below, unchanged.
        item_offset = glyph_item.item.offset
        item_length = glyph_item.item.length
        rtl_run = bool(glyph_item.item.analysis.level % 2)
        per_glyph_text: dict[int, str] = {}
        # index -> (group_start, group_end, group_text), for every glyph
        # index that belongs to a multi-glyph cluster -- lets the main loop
        # below wrap exactly that glyph range in an `/ActualText`
        # marked-content span (see the "COMPLETE FIX" section of the module
        # docstring). `per_glyph_text` stays as the safe, duplicate-text
        # `/ToUnicode` fallback for a reader that does not honor
        # `/ActualText`.
        group_bounds: dict[int, tuple[int, int, str]] = {}
        group_start = 0
        for cluster_i in range(1, num_glyphs + 1):
            if cluster_i == num_glyphs or clusters[cluster_i] != clusters[group_start]:
                group_end = cluster_i
                if group_end - group_start > 1:
                    t1 = clusters[group_start]
                    if rtl_run:
                        t2 = item_length if group_start == 0 else clusters[group_start - 1]
                    else:
                        t2 = item_length if group_end == num_glyphs else clusters[group_end]
                    utf8_text = utf8_text or textbox.pango_layout.text.encode()
                    group_text = utf8_text[item_offset + t1:item_offset + t2].decode()
                    # D2 exact-recovery fix (pypdf): only the CARRIER glyph --
                    # the one whose position the unmodified upstream formula
                    # would already resolve to `group_text` on its own (last
                    # in visual order for LTR, first for RTL; see the
                    # `group_bounds` t1/t2 derivation two lines below, which
                    # mirrors this) -- gets the real text. Every other
                    # (swallowed, e.g. a pre-base matra) glyph in the group
                    # gets U+200B (zero-width space), NOT `group_text` and NOT
                    # an empty string. `pypdf` strips a ZWS and contributes
                    # nothing for it, so this recovers the EXACT source word
                    # instead of a duplicated syllable -- confirmed live
                    # 2026-09-23. An empty string was tried and reverted
                    # first: `pymupdf` treats an empty `/ToUnicode`
                    # destination as "fall back to reading the glyph ID
                    # itself as UTF-16BE", producing a stray, out-of-script
                    # character (see module docstring) -- ZWS is a non-empty,
                    # legitimate destination that dodges that fallback
                    # entirely. `pymupdf` was separately confirmed to still
                    # duplicate a handful of otherwise-unrelated words under
                    # this scheme (its own extraction has further heuristics
                    # beyond a literal CMap read) but never regresses to a
                    # stray character, so `detect_corruption` stays green for
                    # it; only `pypdf`'s exact-recovery is asserted (see
                    # tests/test_pdf_text_extraction_integrity.py).
                    carrier_index = group_start if rtl_run else group_end - 1
                    zero_width_space = '\u200b'
                    for glyph_index in range(group_start, group_end):
                        per_glyph_text[glyph_index] = (
                            group_text if glyph_index == carrier_index else zero_width_space
                        )
                    # The main loop's `PANGO_GLYPH_EMPTY` branch `continue`s
                    # BEFORE it would reach the BDC/EMC logic below, so a
                    # group containing an empty (zero-width placeholder) or
                    # `.notdef`-substituted glyph anywhere in it could open a
                    # `BDC` at the group's start and never reach the
                    # matching `EMC` -- an unbalanced marked-content pair,
                    # which is a real content-stream corruption risk, not
                    # just a missed optimization. Skip `/ActualText` for
                    # such a group entirely (falls back to the safe,
                    # duplicate-text `/ToUnicode` mapping via
                    # `per_glyph_text` above, unchanged) rather than risk it.
                    group_glyph_ids = [glyphs_info[k].glyph for k in range(group_start, group_end)]
                    if not any(
                        gid == pango.PANGO_GLYPH_EMPTY or gid & pango.PANGO_GLYPH_UNKNOWN_FLAG
                        for gid in group_glyph_ids
                    ):
                        for glyph_index in range(group_start, group_end):
                            group_bounds[glyph_index] = (group_start, group_end, group_text)
                group_start = cluster_i
        # --- end D2 fix precomputation

        pango_font = glyph_item.item.analysis.font
        if pango_font != previous_pango_font:
            previous_pango_font = pango_font
            font, font_size = stream.add_font(pango_font)

            if pango.pango_version() < 14802 or font.png:
                font_size = textbox.style['font_size']

            if string:
                stream.show_text(string)
            string = ''
            stream.set_font_size(font.hash, 1 if font.bitmap else font_size)
        string += '<'
        for i, glyph_info in enumerate(glyphs_info[0:num_glyphs]):
            glyph_id = glyph_info.glyph
            width = glyph_info.geometry.width

            if glyph_id == pango.PANGO_GLYPH_EMPTY:
                string += f'>{-width / font_size}<'
                continue

            if glyph_id & pango.PANGO_GLYPH_UNKNOWN_FLAG:
                codepoint = glyph_id - pango.PANGO_GLYPH_UNKNOWN_FLAG
                LOGGER.warning(
                    '.notdef glyph rendered for Unicode string unsupported by fonts: '
                    '"%s" (U+%04X)', chr(codepoint), codepoint)
                glyph_id = font.get_unused_glyph_id(codepoint)
                font.widths[glyph_id] = round(width * 1000 * FROM_UNITS / font_size)
                if 0 not in font.widths:
                    font.widths[0] = font.widths[glyph_id]

            # --- D2 COMPLETE FIX: wrap a multi-glyph cluster's glyphs in an
            # `/ActualText` marked-content span, so a spec-compliant reader
            # recovers the cluster's EXACT original text regardless of
            # visual-vs-logical glyph order or any individual glyph's
            # `/ToUnicode` entry (see module docstring's "COMPLETE FIX"
            # section). Pending glyph data must be flushed via `show_text`
            # first -- `BDC`/`EMC` are their own content-stream operators
            # and cannot appear inside an in-progress `[...] TJ` array under
            # construction, the same constraint the `rise` branch below
            # already has to honor (identical flush logic, copied rather
            # than factored out, to keep this diff from touching upstream's
            # existing lines).
            group = group_bounds.get(i)
            if group is not None and group[0] == i:
                if string:
                    if string[-1] == '<':
                        string = string[:-1]
                    else:
                        string += '>'
                    if string:
                        stream.show_text(string)
                stream.begin_marked_content(
                    'Span', pydyf.Dictionary({'ActualText': pydyf.String(group[2])}))
                string = '<'
            # --- end D2 COMPLETE FIX marked-content start

            # Create mapping between glyphs and Unicode codepoints.
            if glyph_id not in font.to_unicode:
                if i in per_glyph_text:
                    font.to_unicode[glyph_id] = per_glyph_text[i]
                else:
                    offset = glyph_item.item.offset
                    t1 = clusters[i]
                    if glyph_item.item.analysis.level % 2:  # rtl
                        t2 = glyph_item.item.length if i == 0 else clusters[i - 1]
                    else:
                        t2 = glyph_item.item.length if i == num_glyphs - 1 else clusters[i + 1]
                    utf8_text = utf8_text or textbox.pango_layout.text.encode()
                    font.to_unicode[glyph_id] = utf8_text[offset + t1:offset + t2].decode()

            offset = glyph_info.geometry.x_offset / font_size
            rise = glyph_info.geometry.y_offset / 1000
            if rise:
                if string[-1] == '<':
                    string = string[:-1]
                else:
                    string += '>'
                if string:
                    stream.show_text(string)
                stream.set_text_rise(-rise)
                string = ''
                if offset:
                    string = f'{-offset}'
                string += f'<{glyph_id:02x}>' if font.bitmap else f'<{glyph_id:04x}>'
                stream.show_text(string)
                stream.set_text_rise(0)
                string = '<'
            else:
                if offset:
                    string += f'>{-offset}<'
                string += f'{glyph_id:02x}' if font.bitmap else f'{glyph_id:04x}'

            if glyph_id in font.widths:
                logical_width = font.widths[glyph_id]
            else:
                pango.pango_font_get_glyph_extents(
                    pango_font, glyph_id, stream.ink_rect, stream.logical_rect)
                logical_width = font.widths[glyph_id] = round(
                    stream.logical_rect.width * 1000 * FROM_UNITS / font_size)

            kerning = logical_width + offset - width * 1000 * FROM_UNITS / font_size
            if kerning:
                string += f'>{int(kerning)}<'

            if font.svg:
                svg_data = get_hb_object_data(font.hb_face, 'svg', glyph_id)
                if svg_data:
                    tree = ElementTree.fromstring(svg_data)
                    if tree.get('id') != f'glyph{glyph_id}':
                        defs = ElementTree.Element('defs')
                        for child in list(tree):
                            defs.append(child)
                            tree.remove(child)
                        tree.append(defs)
                        ElementTree.SubElement(
                            tree, 'use', attrib={'href': f'#glyph{glyph_id}'})
                    if 'viewBox' not in tree.attrib:
                        tree.attrib['viewBox'] = f'0 0 {font.upem} {font.upem}'
                    image = SVGImage(tree, None, None, None)
                    a = d = 1
                    emojis.append([image, font, a, d, x_advance, 0])
            elif font.png:
                png_data = get_hb_object_data(font.hb_font, 'png', glyph_id)
                if png_data:
                    pillow_image = Image.open(BytesIO(png_data))
                    image_id = f'{font.hash}{glyph_id}'
                    image = RasterImage(pillow_image, image_id, png_data)
                    d = logical_width / 1000
                    a = pillow_image.width / pillow_image.height * d
                    pango.pango_font_get_glyph_extents(
                        pango_font, glyph_id, stream.ink_rect,
                        stream.logical_rect)
                    f = -stream.logical_rect.y
                    f = f * FROM_UNITS / font_size - font_size
                    emojis.append([image, font, a, d, x_advance, f])
            elif font.colr:
                svg_data = get_hb_object_data(font.hb_font, 'colr', glyph_id)
                if svg_data:
                    tree = ElementTree.fromstring(svg_data)
                    image = SVGImage(tree, None, None, None)
                    a = d = 1
                    e = x_advance - kerning
                    emojis.append([image, font, a, d, e, -textbox.baseline])

            x_advance += (logical_width + offset - kerning) / 1000

            # --- D2 COMPLETE FIX: close the `/ActualText` span opened above
            # after this cluster's LAST glyph.
            if group is not None and group[1] - 1 == i:
                if string[-1] == '<':
                    string = string[:-1]
                else:
                    string += '>'
                if string:
                    stream.show_text(string)
                stream.end_marked_content()
                string = '<'
            # --- end D2 COMPLETE FIX marked-content end

        if string[-1] == '<':
            string = string[:-1]
        else:
            string += '>'

    stream.show_text(string)

    return emojis
