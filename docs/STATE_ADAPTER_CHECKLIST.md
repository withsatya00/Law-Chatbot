# State/UT automation adapter checklist

Tracks two SEPARATE things per State/UT, because they are not the same fact:

- **KB content** -- documents/chunks already indexed for that jurisdiction
  (`KnowledgeBaseCoverageService.coverage()`, live-queried; most of this came
  from one-time bulk ingestion, not from any adapter below).
- **Live adapter** -- a registered `OfficialCatalogueAdapter` in
  `app/services/kb_state_adapters.py`/`kb_high_court_adapters.py` that the
  ALREADY-RUNNING `kb_automation_scheduler` checks automatically, on its own
  schedule, with no manual action required once registered.

An adapter never auto-approves anything: every candidate it finds lands
`needs_review`, same as every other path in this codebase (see
`SOURCE_VERIFICATION_CHECKLIST.md`). "Verified" column here means machine- or
human-verified per the existing review workflow, not something this checklist
grants.

Rule this file exists to enforce: **no portal URL is added on a guess.** Every
row's URL was fetched for real (same network path the live automation runtime
uses) before being marked anything other than "not attempted" or "blocked".

Last updated: 2026-09-22.

## Summary

| | Count |
|---|---|
| Total States/UTs | 36 |
| Have a live adapter | 17 states (UP: `up_acts`+`up_ordinances`; MH: `mh_bombay_hc_acts_library`; AS: `assam_acts`; JK: `jk_law_acts`; JH: `jharkhand_acts_rules_policies`; ML: `meghalaya_acts`; PY: `py_law_publications`; TR: `tripura_hc_acts_library`; UT: `uttarakhand_hc_acts`; MN: `manipur_assembly_acts`; NL: `nagaland_act_rules`; LA: `ladakh_acts_rules`; OR: `odisha_law_dept_acts`; CT: `chhattisgarh_law_acts`; MP: `mp_code_state_acts`; DL: `delhi_law_dept_centralized_cos`; DH: `ddd_acts_rules`) |
| Checked and blocked on a specific reason | 19 (KA, RJ, TN, WB, GJ, HR, KL, BR, PB, TG, AN, AP, AR, CH, HP, LD, MZ, SK, GA -- all with real, unfixable-from-here reasons: DNS/TLS failure, no official portal found, JS-rendered SPA / JS-POST-form listing, admin-only site, login-gated CMS, or (GA specifically) an unscoped page mixing hundreds of unrelated documents with no safe way to filter them. None of these is a bare parser bug like CT's was.) |
| Not yet attempted | 0 -- every State/UT has now been checked at least once |
| Have existing KB content (any source) | 4 (UP, MH, OR, MP) |

## Status by State/UT

| Code | State/UT | KB content (docs / approved chunks) | Live adapter | Status |
|---|---|---|---|---|
| UP | Uttar Pradesh | 88 / 1,412 | `up_acts`, `up_ordinances` (upvidhai.gov.in) | ✅ Adapter live |
| MH | Maharashtra | 514 / 21,783 | `mh_bombay_hc_acts_library` (bombayhighcourt.gov.in) | ✅ Adapter live (added 2026-09-22; needed a scoped TLS fix, see below) |
| OR | Odisha | 1,478 / 8,363 | `odisha_law_dept_acts` (law.odisha.gov.in) | ✅ Adapter live (added 2026-09-22; almost certainly the existing corpus's own source, now with an ongoing adapter; paginated `?page=0..84`, 85 pages; needed the generic-anchor-label fix, see below) |
| MP | Madhya Pradesh | 23 / 152 | `mp_code_state_acts` (code.mp.gov.in) | ✅ Adapter live (added 2026-09-22; MP Code site was rebuilt since the disabled `config/law_monitors.json` entry's stale 404; known ceiling: only page 1 of 889 records reachable, later pages are JS-postback) |
| DL | Delhi | 0 | `delhi_high_court_judgments` (judgments) + `delhi_law_dept_centralized_cos` (Acts, added 2026-09-22) | ✅ Adapter live (56 real candidates; cross-department LAW-domain feed, Delhi's own dedicated Acts page has zero real hyperlinks) |
| GA | Goa | 0 | `bombay_high_court_judgments` (shared HC, case law only) | ⛔ Blocked: `www.goa.gov.in/government/acts-and-rules/` has real Act PDFs and the 2026-09-22 `<div>`-ancestor fix DOES extract correct titles, but the same page template mixes in hundreds of unrelated non-Act documents with no reliable DOM signal to separate them; `goaprintingpress.gov.in/state-acts/`'s per-Act PDFs are `pdf.js`-viewer-only (no real `<a href>`), and its index is JS-paginated |
| DH | Dadra and Nagar Haveli and Daman and Diu | 0 | `bombay_high_court_judgments` (judgments) + `ddd_acts_rules` (Acts, added 2026-09-22) | ✅ Adapter live (same NIC template as Ladakh; only 1 concrete document currently published, thinner even than Ladakh's 3) |
| AS | Assam | 0 | `assam_acts` (legislative.assam.gov.in) | ✅ Adapter live (added 2026-09-22; two-hop listing -> detail page -> PDF, paginated) |
| KA | Karnataka | 0 | none | ⛔ Blocked: `dpal.karnataka.gov.in`'s Acts index is reachable, but its per-year PDF listings are not in the static HTML (likely JS-rendered/POST) |
| RJ | Rajasthan | 0 | none | ⛔ Blocked: both candidate hosts (`law-justice.rajasthan.gov.in`, `rajassembly.nic.in`) fail DNS resolution from the automation runtime |
| TN | Tamil Nadu | 0 | none | ⛔ Blocked: `lawdept.tn.gov.in` fails DNS resolution; `tn.gov.in/acts` returns HTTP 404 |
| WB | West Bengal | 0 | none | ⛔ Blocked (reason updated 2026-09-22): the TLS chain issue is now confirmed FIXABLE (server sends the wrong intermediate cert; the correct one is fetchable from the CA's own AIA URL) but wasn't worth fixing on its own -- the real, still-current blocker is that the site's "WB Acts & Rules" tab is a `method="post"` search form with an empty results container in the static response (JS/POST-driven, same class as Karnataka) |
| AN | Andaman and Nicobar Islands | 0 | none | ⛔ Blocked: no reachable official Acts/Gazette listing -- gazette-search host `andssw1.and.nic.in` fails the TLS handshake itself (connection reset); Forest Dept's Acts page lists only central environmental Acts, not a law/legislative source; `law.and.nic.in`/`legal.and.nic.in` fail DNS; `and.nic.in`/`andaman.gov.in` time out |
| AP | Andhra Pradesh | 0 | none | ⛔ Blocked: six candidate Law Dept/legislature hostnames all fail DNS (`apleg`, `lawdept`, `law`, `apld`, `legislature`, `aplegislature` + `.ap.gov.in`); the reachable NeVA Council portal's Legislation path (`apc.neva.gov.in/Bill`) returns HTTP 302 into a login-gated CMS |
| AR | Arunachal Pradesh | 0 | none | ⛔ Blocked: `law.arunachal.gov.in/acts` is a JS-rendered React SPA shell (1,102-byte static response, empty `#root`); NeVA Assembly portal's Legislation path (`arla.neva.gov.in/Bill`) returns HTTP 302 into the same login-gated CMS pattern as AP |
| CH | Chandigarh | 0 | none | ⛔ Blocked: `chandigarh.gov.in/act` and its Law & Prosecution dept page are administrative-template boilerplate with placeholder (`inner.html`) links, not a real listing (same pattern as HR/KL); `egazette.chd.gov.in` is reachable but its anchor shape is unverified -- worth a follow-up |
| CT | Chhattisgarh | 0 | `chhattisgarh_law_acts` (law.cgstate.gov.in) | ✅ Adapter live (added 2026-09-22; needed TWO fixes: a new bounded `<div>`-ancestor parser fallback (see below) for its `<tr>`-less markup, AND a scoped TLS legacy-renegotiation fix, same error class as Bombay HC; uses the page's own `?notice_type[]=act` filter to exclude unrelated notifications) |
| GJ | Gujarat | 0 | none | ⛔ Blocked: `lpd.gujarat.gov.in`'s "State Enactments" page returns HTTP 200 but 0 PDF links in the static response -- a search/filter UI, not a static listing |
| HP | Himachal Pradesh | 0 | none | ⛔ Blocked: `himachal.nic.in/law`'s Acts menu resolves through `LinkPageContent`/`GetLinkPageContent` wrapper endpoints, 3 levels deep, ending in a `BindFileFolderGrid()` JS-populated table shell with no real rows in the static response (same class as KA) |
| HR | Haryana | 0 | none | ⛔ Blocked: both `haryana.gov.in/acts/` and `lawandlegislativehry.gov.in` are reachable but are administrative department sites (forms, circulars, results), not Acts listings |
| JH | Jharkhand | 0 | `jharkhand_acts_rules_policies` (jharkhand.gov.in) | ✅ Adapter live (added 2026-09-22; cross-department, 132 items over 14 explicitly-enumerated pages) |
| JK | Jammu and Kashmir | 0 | `jk_law_acts` (law.jk.gov.in) | ✅ Adapter live (added 2026-09-22; 131 items, single unpaginated page) |
| KL | Kerala | 0 | none | ⛔ Blocked: `lawsect.kerala.gov.in` is reachable but is an administrative department site (RTI forms, internship applications), not an Acts listing |
| LA | Ladakh | 0 | `ladakh_acts_rules` (ladakh.gov.in) | ✅ Adapter live (added 2026-09-22; 3 real documents, inherited J&K Reorganisation Act material; PDFs resolve to the shared s3waas.gov.in NIC CDN; needed the generic-anchor-label fix, see below) |
| LD | Lakshadweep | 0 | none | ⛔ Blocked: the UT's own dedicated "Acts & Rules" category (`lakshadweep.gov.in/document-category/acts-rules/`) is reachable but returns zero documents live (a real server-rendered empty state, not a fetch failure); the broader Gazette Notifications feed is populated but mixes non-Act content, not a dedicated Acts listing |
| MN | Manipur | 0 | `manipur_assembly_acts` (assembly.mn.gov.in) | ✅ Adapter live (added 2026-09-22; 42 real rows, single unpaginated page; needed the generic-anchor-label fix, see below) |
| ML | Meghalaya | 0 | `meghalaya_acts` (meghalaya.gov.in) | ✅ Adapter live (added 2026-09-22; two-hop listing -> detail page -> PDF; per-department PDF host varies, partial coverage where a department still serves plain HTTP) |
| MZ | Mizoram | 0 | none | ⛔ Blocked: `mizoram.gov.in` is a Quasar/Vue JS SPA shell (997-byte static response, empty mount point); `mizoramassembly.in` (has real Acts PDFs) fails `official_url()`'s `.gov.in`/`.nic.in` domain rule; legacy `mizoram.nic.in` times out |
| NL | Nagaland | 0 | `nagaland_act_rules` (nagaland.gov.in) | ✅ Adapter live (added 2026-09-22; 349 real rows, single unpaginated page; needed the generic-anchor-label fix, see below) |
| PY | Puducherry | 0 | `py_law_publications` (law.py.gov.in) | ✅ Adapter live (added 2026-09-22; 8 real "Puducherry Code" volumes; needed a scoped TLS 1.2 cap, see below) |
| PB | Punjab | 0 | none | ⛔ Blocked: no direct official Acts-listing URL identified yet (Department of Legal and Legislative Affairs has no dedicated public Acts page found; revenue.punjab.gov.in only covers Revenue-specific Acts) |
| SK | Sikkim | 0 | none | ⛔ Blocked: both real candidate pages on `sikkim.gov.in` (including one literally titled "List of Acts Passed by the Sikkim Legislative Assembly") return HTTP 200 but with a genuinely empty content region in the raw response (0 `<table>` tags, no article/body node) -- likely JS/AJAX-populated on this page template |
| TG | Telangana | 0 | none | ⛔ Blocked: no direct official Acts-listing URL identified yet (Law Department is described as advisory-only; no dedicated public Acts page found) |
| TR | Tripura | 0 | `tripura_hc_acts_library` (thc.nic.in) | ✅ Adapter live (added 2026-09-22; 164 real PDF anchors, Tripura's own Law Dept page 404s so this reuses the Tripura High Court's library) |
| UT | Uttarakhand | 0 | `uttarakhand_hc_acts` (highcourtofuttarakhand.gov.in) | ✅ Adapter live (added 2026-09-22; 195 real rows; Uttarakhand's own portals unusable (soft-404 / HTTP 503) so this reuses the High Court's library; needed a scoped TLS 1.2 cap, see below) |
| BR | Bihar | 0 | none | ⛔ Blocked: official Law Department page (`state.bihar.gov.in/law/`) links to e-Gazette search, not a bare-Acts listing; needs a different candidate URL |

## Known systemic issue: legacy TLS on `bombayhighcourt.gov.in`/`.nic.in` and `law.cgstate.gov.in`

These servers' TLS stacks require unsafe legacy renegotiation that Python's
OpenSSL 3.x refuses by default. Fixed 2026-09-22 with a scoped allowance
(`app/schemas/law_monitoring.py::ssl_context_for_host`) used by both fetchers
(`fetch_official_snapshot`, `OfficialDownloader`) -- applies ONLY to these
three hostnames, every other fetch keeps the ordinary secure default.
Confirmed live against `mh_bombay_hc_acts_library` after the fix (100 real
candidates discovered), and independently against `chhattisgarh_law_acts`
(same `[SSL: UNSAFE_LEGACY_RENEGOTIATION_DISABLED]` error, different host,
confirmed 2026-09-22). The separate `bombay_high_court_judgments` adapter
(`bombayhighcourt.nic.in/recentorderjudgment.php`) still fails, but with a
DIFFERENT error (`ConnectTimeout`, not a TLS error) -- a connectivity issue
this fix does not address and that needs its own investigation.

## Known systemic issue: TLS 1.3 handshake hang on `law.py.gov.in`/`highcourtofuttarakhand.gov.in`

These two hosts' TLS 1.3 handshake hangs indefinitely against Python's
OpenSSL client, even though `curl`/Windows Schannel on the same machine
connects in well under a second -- a different TLS negotiation path, and a
DIFFERENT failure mode from the Bombay High Court's legacy-renegotiation
issue above (this one is a silent hang, not an explicit rejection). Fixed
2026-09-22 with the same scoped mechanism
(`app/schemas/law_monitoring.py::ssl_context_for_host`), capping the client
to TLS 1.2 for exactly these two hostnames -- confirmed directly (handshake
completes in under 0.3s once capped) and confirmed live against both
`py_law_publications` (8 real candidates) and `uttarakhand_hc_acts` (100 real
candidates, capped by `max_items`) after the fix.

## Known systemic issue: generic download/view button text overriding the real title

Five real state portals (Chhattisgarh, Ladakh, Manipur, Nagaland, Odisha)
were found where the listing's real title lives in a sibling table cell, but
`OfficialCatalogueAdapter._parse` (`kb_central_adapters.py`) picked the
anchor's own visible text first whenever it was non-empty -- and for these
five, that text was a generic download/view button label ("View",
"Download", "Download 926.24 KB", "ENG"/"HIN", "Download(495.23 KB)"), not
the title. Fixed 2026-09-22 with `_GENERIC_ANCHOR_LABEL`, a narrow regex of
exactly the label shapes confirmed live across those five sites -- treats a
matching label the same as an empty/icon-only anchor (falls back to the row
text), while leaving every other adapter's behavior unchanged (regression-
verified: all pre-existing adapter tests still pass unmodified).

This unblocked **four** of the five -- Manipur, Nagaland, Ladakh, Odisha --
confirmed live after the fix (42, 349, 3, and 20-per-page/85-page real
candidates respectively, all with correct titles). Chhattisgarh needed a
SECOND, separate fix -- see below -- because its anchor has no
`<tr>`/`<li>`/`<article>` ancestor at all (the title sits in a sibling
`<div>`), so this fix alone didn't reach it.

## Known systemic issue: `<div>`-only listing markup with no `<tr>`/`<li>`/`<article>` ancestor

Chhattisgarh's listing (`law.cgstate.gov.in/act-details`) lays out each item
as a `div.act-card` whose title (`div.act-content > .act-title-box >
p.act-title`) and download button (`div.act-btn-grp > a`) are SIBLING
`<div>`s -- there is no `<tr>`/`<li>`/`<article>` ancestor anywhere for
`_parse`'s title fallback to walk up to, so even after the generic-label fix
above, the extracted title still collapsed to the button's own "ENG"/"HIN"
label. Fixed 2026-09-22 with `kb_central_adapters.py::_nearby_record_text`:
when no `<tr>`/`<li>`/`<article>` ancestor exists, it walks up a SMALL,
bounded number (4) of `<div>`/`<span>` ancestors and uses the first one
whose text is both meaningfully longer than the anchor's own (so it actually
adds new content) and not implausibly large (capped at 600 characters, so a
big outer container spanning multiple listing items is never mistaken for
one item's own text). This path is used ONLY as a fallback when no semantic
row ancestor exists at all, so every one of the 13 already-live adapters
(which all find a real `<tr>`/`<li>`/`<article>`) is unaffected -- confirmed
by the full existing test suite passing unmodified. Confirmed live against
`chhattisgarh_law_acts` after both this fix and the TLS fix below (8 real
candidates, correct titles, e.g. "Chattisgarh Land Holdings (Validation)
Act, 2013").

## How a row moves from "Not yet attempted" to a live adapter

1. Search for the State/UT's official law/legislative/gazette department
   website.
2. Fetch the candidate URL for real, from this same automation runtime's
   network path (not just a browser) -- confirm HTTP 200, a genuine
   `.gov.in`/`.nic.in` host, and real Act/PDF links in the response actually
   received (not assumed from a screenshot or a search snippet).
3. Write a `CatalogueConfig` in `kb_state_adapters.py` (or
   `kb_high_court_adapters.py` for a court), matching the real page's anchor
   text / link shape.
4. Add a unit test with a byte-for-byte real snippet of that page (see
   `tests/test_kb_state_adapters.py::test_maharashtra_acts_adapter_extracts_the_real_page_shape`).
5. Run `adapter.discover(None)` once against the live URL to confirm it
   actually parses real candidates (see the diagnostic scripts used for MH).
6. Register it in `state_source_adapters()`/`high_court_source_adapters()`.
   The already-running `kb_automation_scheduler` picks it up automatically
   from then on -- no separate registration step, no manual download.
7. Update this checklist's row and the Summary table.
