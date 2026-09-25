"""Official State/UT Gazette and legislature catalogue adapters.

Same shared engine as `kb_central_adapters.py` (`OfficialCatalogueAdapter`),
carrying a real State/UT identity via `CatalogueConfig.jurisdiction_code` so a
discovered candidate's `applicable_state_codes` is correct from the moment it
is discovered (see `KnowledgeBaseAutomationService._process_claimed`) instead
of landing as `applicability: unknown` for every automated candidate.

Per `docs/SOURCE_GOVERNANCE.md`, this module does NOT invent portal URLs for
States/UTs this repository has not verified. It registers:
  - the two Uttar Pradesh listings already live in `config/law_monitors.json`
    (`upvidhai.gov.in`'s Acts and Ordinances listings);
  - the Bombay High Court's own Maharashtra Acts library index
    (`bombayhighcourt.gov.in/bhc/libweb/legislation/acts/listofmahacts.html`),
    added 2026-09-22. This is the SAME page whose `YYYY.NN.pdf` files already
    make up the ~514 `mh_acts_*.pdf` documents already indexed in this KB (a
    prior one-time ingestion, not run through this adapter engine) -- reusing
    it, rather than a second Maharashtra source
    (`lj.maharashtra.gov.in`'s "Act List", also confirmed reachable but with a
    different filename scheme and 16-page pagination this engine would need
    extra logic to walk), keeps ongoing MH discovery checking the exact
    source the existing corpus came from.
  - Assam's Legislative Department Acts listing
    (`legislative.assam.gov.in/documents/assam-acts`), added 2026-09-22.
    Confirmed live: real per-Act detail-page links, paginated (`?page=1..7+`,
    enumerated explicitly in `urls`), each detail page linking to the actual
    PDF -- `resolve_pdf_links=True` walks that second hop.
  - Jammu and Kashmir's Law Department Acts table
    (`law.jk.gov.in/LnJActNRulStaActs.php`), added 2026-09-22. 131 real rows
    on one unpaginated page; `include_pattern` is scoped to the specific
    `backendportal/uploads/acts/` path since this host's PDFs are not all
    Acts.
  - Jharkhand's cross-department "Acts, Rules & Policies" document list
    (`jharkhand.gov.in/Home/DocumentList?doctype=...&subdoctype=...`), added
    2026-09-22. 132 items over 14 explicitly-enumerated pages (confirmed the
    filter query params survive pagination, unlike the bare `page=N` links
    rendered in the page body).
  - Meghalaya's Acts listing (`meghalaya.gov.in/acts`), added 2026-09-22.
    Two-hop like Assam, but each detail page's PDF lives on a different
    CONCERNED DEPARTMENT domain (`megurban.gov.in`, `megcooperation.gov.in`,
    ...) -- coverage is partial by construction where a department still
    serves plain HTTP (`official_url()` is HTTPS-only).
  - Puducherry's Law Department publications list
    (`law.py.gov.in/publications.html`), added 2026-09-22. 8 real
    "Puducherry Code" compiled-Acts volumes (English + Tamil); icon-only
    anchors, `include_pattern` scoped to the specific `docs/[T]codeN.pdf`
    filenames since the same page links many unrelated admin PDFs.
  - Tripura's Acts library, hosted by the Tripura High Court
    (`thc.nic.in/tsl_acts.html`), added 2026-09-22, because Tripura's own Law
    Dept page (`law.tripura.gov.in/acts_of_tripura`) returns HTTP 404. 164
    real PDF anchors, anchor text is the real title, no ancestor needed.
  - Uttarakhand's Acts library, hosted by the High Court of Uttarakhand
    (`highcourtofuttarakhand.gov.in/uttarakhand-act/`), added 2026-09-22,
    because Uttarakhand's own portals were unusable (`uk.gov.in`: soft-404
    template; `slsa.uk.gov.in`: HTTP 503). 195 real rows, anchor text is the
    real title.

  Added 2026-09-22 (second pass), after `kb_central_adapters.py::_parse`
  gained `_GENERIC_ANCHOR_LABEL` -- a fix for a title-extraction bug found
  in FIVE real state portals (Chhattisgarh, Ladakh, Manipur, Nagaland,
  Odisha): a `<tr>`/`<li>`/`<article>` ancestor was found correctly, but the
  anchor's own non-empty visible text was a generic download/view button
  label ("View", "Download", "Download 926.24 KB", "ENG"/"HIN"), which used
  to win outright over the real title sitting in a sibling cell. The fix
  treats a small, closed set of confirmed generic labels the same as an
  empty/icon-only anchor (falls back to the row text). This unblocked FOUR
  of the five -- Manipur, Nagaland, Ladakh, Odisha -- whose anchors sit
  inside a real `<tr>`/`<li>` the parser could already walk to. Chhattisgarh
  needed a SECOND, separate fix (below) because its anchor sits inside a
  `<div>` with no `<tr>`/`<li>`/`<article>` ancestor at all.
  - Manipur's Acts listing (`assembly.mn.gov.in/acts/acts-enacted-through-
    bills`), single unpaginated page, 42 real rows.
  - Nagaland's Acts/Amendments listing (`nagaland.gov.in/act-rules`, note
    singular), single unpaginated page, 349 real rows; `include_pattern`
    requires `.pdf` specifically so each row's second, `.zip`-companion
    anchor (same row text) isn't also picked up.
  - Ladakh's Acts & Rules document category
    (`ladakh.gov.in/document-category/acts-rules/`), 3 documents currently
    published (inherited J&K Reorganisation Act material); PDFs resolve to
    the shared `s3waas.gov.in` NIC CDN, not `ladakh.gov.in` itself.
  - Odisha's Law Department Acts-and-Ordinances listing
    (`law.odisha.gov.in/publication/acts-and-ordinances/acts`), paginated
    (`?page=0..84`, 85 pages enumerated explicitly) -- almost certainly the
    actual source of this KB's existing 1,478-document Odisha corpus, now
    with an ongoing adapter for the first time.

  Added 2026-09-22 (third pass): Chhattisgarh's Law Department Acts listing
  (`law.cgstate.gov.in/act-details`, using the page's own server-side
  `?notice_type[]=act` filter -- the unfiltered page mixes in unrelated
  notifications). This needed TWO separate fixes, not just the generic-label
  one above: (1) `kb_central_adapters.py::_nearby_record_text`, a new
  bounded `<div>`/`<span>`-ancestor fallback used ONLY when no real
  `<tr>`/`<li>`/`<article>` ancestor exists anywhere (so every other
  adapter's behavior is unchanged -- confirmed by the full existing test
  suite passing unmodified), for the case where title and download button
  are sibling `<div>`s rather than cells in a shared row; and (2) adding
  `law.cgstate.gov.in` to `_LEGACY_RENEGOTIATION_HOSTS`
  (`app/schemas/law_monitoring.py`) after hitting the exact same `[SSL:
  UNSAFE_LEGACY_RENEGOTIATION_DISABLED]` error as `bombayhighcourt.gov.in`,
  confirmed independently against this different host.

  Added 2026-09-22 (fourth pass) -- the last 4 States/UTs that were "not yet
  attempted":
  - Madhya Pradesh's MP Code Acts listing (`code.mp.gov.in/stateacts.aspx`).
    The site has been REBUILT since the disabled `config/law_monitors.json`
    `MP Code` entry recorded its HTTP 404 (against the old
    `content/Eng/index.aspx` path) -- that recorded reason is now stale.
    Known limitation: only the first 25 of 889 reported records are
    reachable (later pages are JS `__doPostBack`, not real URLs).
  - Delhi's cross-department LAW-domain "Centralized COS" feed
    (`delhi.gov.in/centralized-cos`). Delhi's own dedicated Acts page
    (`law.delhi.gov.in/law/list-acts-extended-nct-delhi`) is a plain-text
    list with zero real hyperlinks. 56 real candidates.
  - Dadra and Nagar Haveli and Daman and Diu's Acts & Rules document
    category (`ddd.gov.in/document-category/acts-rules/`), the same NIC
    template as Ladakh. Only 1 concrete PDF currently published -- thinner
    even than Ladakh's 3, same "this UT's own legislative apparatus is new"
    reasoning already accepted there.
  - Goa was investigated but NOT added -- see the blocked-states list below.

**States checked 2026-09-22 and NOT added, with the specific reason** (kept
here so the next attempt doesn't repeat the same dead end):
  - Karnataka (`dpal.karnataka.gov.in`): the Acts-and-Ordinances index page is
    reachable, but its per-year PDF links were not present in the static HTML
    fetched (likely JS-rendered or behind a POST); needs a real browser-driven
    check before an adapter can be written.
  - Rajasthan (`law-justice.rajasthan.gov.in`, `rajassembly.nic.in`): DNS
    resolution failed from this same automation runtime for both candidate
    hosts.
  - Tamil Nadu (`lawdept.tn.gov.in`): DNS resolution failed; `tn.gov.in/acts`
    (the other candidate) returned HTTP 404.
  - West Bengal (`sarthac.gov.in`, the Law Department's own linked Acts
    portal): re-checked 2026-09-22 -- the original "TLS certificate
    verification failed" reason is now STALE (the server's TLS chain issue
    is a confirmed, fixable, missing-intermediate misconfiguration: it sends
    the wrong second certificate, but the correct one, "EM DV TLS CA -
    G2A-1", is fetchable from the CA's own AIA `CA Issuers` URL
    (`repository.emsign.com`) and completes full, real verification once
    loaded -- not attempted as a codebase fix since it wouldn't unlock a
    working adapter anyway, see below). The REAL, still-current blocker: the
    site's own "WB Acts & Rules" tab (`Act-Ordinance-View`, `#act`) is a
    search FORM (`method="post"`) whose results container
    (`<ul id="actt">`) is empty in the static response -- confirmed the
    listing is entirely POST/JS-driven, same class as Karnataka. The only
    static link on the page is a Google Docs "List of West Bengal Acts"
    document, which fails `official_url()`'s `.gov.in`/`.nic.in` domain rule
    outright regardless.
  - Andaman and Nicobar Islands: no reachable official Acts/Gazette listing --
    the UT's own gazette-search host (`andssw1.and.nic.in`) fails the TLS
    handshake itself (connection reset, not a cert or DNS issue); the Forest
    Department's Acts page (`forest.and.nic.in`) only lists central
    environmental Acts, not a law/legislative-department source;
    `law.and.nic.in`/`legal.and.nic.in` fail DNS; `and.nic.in`/
    `andaman.gov.in` time out.
  - Andhra Pradesh: six candidate Law Department/legislature hostnames
    (`apleg.ap.gov.in`, `lawdept.ap.gov.in`, `law.ap.gov.in`, `apld.ap.gov.in`,
    `legislature.ap.gov.in`, `aplegislature.ap.gov.in`) all fail DNS
    resolution; the one reachable candidate, the AP Legislative Council's NeVA
    portal (`apc.neva.gov.in`), has its Legislation path (`/Bill`) return
    HTTP 302 (fails under this runtime's no-redirect rule) into a
    login-gated CMS.
  - Arunachal Pradesh: `law.arunachal.gov.in/acts` is a JS-rendered React SPA
    shell (1,102-byte static response, empty `#root`, no real anchors); the
    NeVA Assembly portal's Legislation path (`arla.neva.gov.in/Bill`) returns
    HTTP 302 into the same login-gated CMS pattern as Andhra Pradesh.
  - Chandigarh: both `chandigarh.gov.in/act` and its Law & Prosecution
    department page are administrative-template boilerplate with placeholder
    (`inner.html`) links, not a real Acts listing (same pattern as Haryana/
    Kerala). `egazette.chd.gov.in` is reachable (HTTP 200, `text/html`) but
    its anchor shape is unverified -- worth a follow-up attempt, not yet a
    pass.
  - Himachal Pradesh: `himachal.nic.in/law`'s "State Acts" menu items resolve
    through `Content/LinkPageContent` and `Content/GetLinkPageContent`
    wrapper endpoints, three levels deep, ending in a `BindFileFolderGrid()`-
    driven table shell with no real file/folder rows in the static response
    -- the rows load via a further client-side call this fetcher path
    doesn't replicate. Same class of block as Karnataka.

**More states checked 2026-09-22 (second pass) and NOT added:**
  - Lakshadweep (`lakshadweep.gov.in/document-category/acts-rules/`): the
    UT's own dedicated "Acts & Rules" category is reachable but currently
    returns zero documents (server-rendered "Sorry, no posts matched your
    criteria", a real empty state, not a fetch failure). The site's broader
    Gazette Notifications feed is populated but mixes recruitment orders and
    other non-Act content -- not a dedicated Acts listing, so not treated as
    a substitute (same reasoning as Bihar's e-Gazette-search rejection).
  - Mizoram: `mizoram.gov.in` (and its `/acts-and-rules` path) is a Quasar/
    Vue SPA shell (997-byte static response, empty `#q-app` mount point) --
    JS-rendered, same class as Karnataka/Arunachal. The one site with real
    Acts PDFs found (`mizoramassembly.in`) fails `official_url()`'s domain
    rule outright (`.in`, not `.gov.in`/`.nic.in`). The legacy
    `mizoram.nic.in` host times out.
  - Sikkim: both real candidate pages (`sikkim.gov.in`'s own "List of Acts
    Passed by the Sikkim Legislative Assembly" and its Gazette "acts and
    legislatures published" page) return HTTP 200 but with an EMPTY content
    region in the raw response actually received -- zero `<table>` tags, no
    article/body content node, just the site's global nav repeated (likely
    JS/AJAX-populated on this specific page template). Other reachable pages
    on the same host (`/media/notification-circular`,
    `/media/repealed-withdrawn-acts`) are not Acts listings, and where they
    do have PDF links, hit the same generic-anchor-text bug class anyway
    (fixed elsewhere -- see below -- but moot here since these pages are not
    Acts listings regardless).
  - Goa: TWO independent real candidates found, both genuinely blocked.
    (a) `www.goa.gov.in/government/acts-and-rules/` -- 415 real
    `div.document_holder` entries, no `<tr>`/`<li>`/`<article>` ancestor
    (same shape class as Chhattisgarh); `_nearby_record_text`'s bounded
    `<div>` fallback DOES correctly extract real titles here (confirmed
    directly) -- but the page itself is not scoped to Acts: the SAME
    template and wrapper class (`policy_downloads`) is used for hundreds of
    unrelated documents (admission prospectuses, recruitment notices) mixed
    into the identical listing, with no distinguishing DOM signal found. A
    keyword filter (`\bact\b`/`\brule\b`) was tried and rejected: it still
    over-matched (unrelated nearby text also contains those words), risking
    silent junk in the KB -- not a config-only fix, and the safe/correct fix
    isn't identified yet. (b) `goaprintingpress.gov.in/state-acts/` has a
    clean `<tr>` listing hop, but its per-Act detail pages embed the real
    PDF only via a client-side `pdf.js` viewer (`onclick="downloadPDF(...)"`
    and a `<script type="module">` variable, never a real `<a href="....pdf">`)
    -- `_resolve_pdf_links` only walks real anchor tags, so this hop
    resolves to nothing; the index itself is also JS-paginated
    (`javascript:void(0)` page links, ~10 of an unknown larger total).

**To add the next State/UT**: verify the portal URL resolves AND returns a
real listing page from the SAME network path the automation runtime uses
(not just a browser) -- a page a browser renders fine can still be
unreachable to the backend's fetcher (DNS, TLS, or JS-rendering differences
all confirmed live above). Only then add a `CatalogueConfig` here with the
real `jurisdiction_code`. Register it disabled-by-omission (simply do not add
it to `state_source_adapters()`) until verified -- mirroring the existing MP
Code entry in `config/law_monitors.json`, which is registered but `enabled:
false` after returning HTTP 404 during local activation.
"""

from __future__ import annotations

from typing import Any

from app.services.kb_central_adapters import CatalogueConfig, OfficialCatalogueAdapter


class MadhyaPradeshActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22 against `code.mp.gov.in/stateacts.aspx`, the
    MP Code portal's real "ACTS" page (a REBUILT site: the disabled
    `config/law_monitors.json` `MP Code` entry recorded an HTTP 404 against
    the OLD path `content/Eng/index.aspx`, which is now stale -- the site
    has since been rebuilt as an ASP.NET GridView on this new path). Real
    `<tr class="animate-row">` rows, anchor text IS the real title. Known,
    accepted limitation: the footer reports "889 records" but the page-N
    links are `javascript:__doPostBack(...)`, not real URLs -- confirmed
    `?page=2` returns byte-identical page-1 content -- so only the first 25
    (apparently newest-first) are reachable, the same class of ceiling
    already accepted for UP Ordinances/JK (single real page, not everything
    the site has ever published).
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="mp_code_state_acts", authority="Law Department, Government of Madhya Pradesh (MP Code)",
            urls=("https://code.mp.gov.in/stateacts.aspx",),
            document_type="bare_act",
            include_pattern=r"\.pdf",
            jurisdiction_code="MP",
            expected_domain_suffixes=("code.mp.gov.in",),
        ), fetcher)


class DelhiActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22 against `delhi.gov.in/centralized-cos`, a
    cross-department "Centralized COS" (circulars/orders/schemes) feed,
    pre-filtered with `field_domain_id_value=LAW` in the query string --
    Delhi's own "List of Acts extended to NCT of Delhi" page
    (`law.delhi.gov.in/law/list-acts-extended-nct-delhi`) is a plain-text
    list with zero real hyperlinks, not usable. Real `<tr>` rows; anchor's
    own text is the generic label "Download" (handled by
    `_GENERIC_ANCHOR_LABEL`), real title in a sibling `views-field-field-
    subject` cell. `include_pattern` is scoped to the `/centralized-cos/`
    file-path segment specifically (not a bare `.pdf`/`act` match): this
    same page's raw HTML also carries site-wide mega-menu links to dozens of
    unrelated `delhi.gov.in` PDFs (MLA lists, e-auction notices, unrelated
    circulars under different path prefixes) that a broader pattern would
    wrongly pick up -- confirmed by testing both against the real page (56
    correctly-scoped candidates with this pattern vs. 60 candidates, mostly
    wrong, with a bare `.pdf` pattern). `document_type="unknown"` because the
    LAW-domain filter mixes real Acts with department circulars (e.g. a
    staffing circular was among the 56), not because the source is
    unidentified. No pagination found (single bounded "recent items" view).
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="delhi_law_dept_centralized_cos", authority="Government of NCT of Delhi (Law Department)",
            urls=(
                (
                    "https://delhi.gov.in/centralized-cos?field_cos_no_value=&title="
                    "&field_cos_type_target_id_1=All&field_date_value=&field_date_value_1="
                    "&field_domain_id_value=LAW&field_pdf_content_value=&field_subject_value="
                ),
            ),
            document_type="unknown",
            include_pattern=r"/centralized-cos/",
            jurisdiction_code="DL",
            expected_domain_suffixes=("delhi.gov.in",),
        ), fetcher)


class DadraNagarHaveliDamanDiuActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22 against `ddd.gov.in/document-category/
    acts-rules/` -- the exact same s3waas.gov.in-templated NIC "document
    category" page shape already accepted for `ladakh_acts_rules`. Only 1
    concrete Act/Rules PDF is currently reachable (a second row links out to
    an `indiacode.nic.in` browse/collection page, not a direct PDF) --
    thinner even than Ladakh's 3, but genuinely live and real, same
    reasoning already accepted for Ladakh ("this UT's own legislative
    apparatus is new"). `daman.nic.in/law-justice.aspx` (the other
    candidate) fails with a genuine untrusted-root TLS certificate error
    (`SEC_E_UNTRUSTED_ROOT`) -- a real rejection, not a hang, and not the
    same class as the documented legacy-renegotiation/TLS1.3-hang
    workarounds, so not treated as fixable here.
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="ddd_acts_rules", authority="Union Territory of Dadra and Nagar Haveli and Daman and Diu",
            urls=("https://ddd.gov.in/document-category/acts-rules/",),
            document_type="bare_act",
            include_pattern=r"\.pdf",
            jurisdiction_code="DH",
            expected_domain_suffixes=("s3waas.gov.in",),
        ), fetcher)


class UttarPradeshActsAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="up_acts", authority="Uttar Pradesh Vidhai (Legislative) Section",
            urls=("https://upvidhai.gov.in/Act-hi.aspx",),
            document_type="bare_act",
            include_pattern=r"act|अधिनियम|\.pdf",
            jurisdiction_code="UP",
            expected_domain_suffixes=("upvidhai.gov.in",),
        ), fetcher)


class UttarPradeshOrdinancesAdapter(OfficialCatalogueAdapter):
    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="up_ordinances", authority="Uttar Pradesh Vidhai (Legislative) Section",
            urls=("https://upvidhai.gov.in/Ordinance-hi.aspx",),
            document_type="ordinance",
            include_pattern=r"ordinance|अध्यादेश|\.pdf",
            jurisdiction_code="UP",
            expected_domain_suffixes=("upvidhai.gov.in",),
        ), fetcher)


class AssamActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, real `<a href="/documents-detail/...">
    Act Title, YYYY (Assam Act No.N of YYYY)</a>` entries on
    `legislative.assam.gov.in/documents/assam-acts` (a paginated Drupal
    listing, `?page=1..7+`). Unlike Maharashtra's single-hop PDF links, this
    listing links to a per-Act detail page that itself links to the real PDF
    -- `resolve_pdf_links=True` (the same mechanism `IndiaCodeAdapter` and
    `LegislativeDepartmentAdapter` already use) walks that second hop.
    `urls` enumerates each pagination page explicitly (mirroring
    `IndiaCodeAdapter`'s offset pages) since this engine only paginates
    across multiple `CatalogueConfig.urls`, not query parameters on one URL.
    """

    def __init__(self, fetcher: Any = None) -> None:
        urls = ("https://legislative.assam.gov.in/documents/assam-acts",) + tuple(
            f"https://legislative.assam.gov.in/documents/assam-acts?page={page}" for page in range(1, 8)
        )
        super().__init__(CatalogueConfig(
            name="assam_acts", authority="Assam Legislative Department",
            urls=urls,
            document_type="bare_act",
            include_pattern=r"act|documents-detail",
            resolve_pdf_links=True,
            jurisdiction_code="AS",
            expected_domain_suffixes=("legislative.assam.gov.in",),
        ), fetcher)


class JammuKashmirActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, 131 real
    `<tr class="trsearch"><td class="sno-cell">N</td><td class="rule-title">
    <a href="jklawact/backendportal/uploads/acts/actN_....pdf">Act Title</a>
    </td></tr>` rows on `law.jk.gov.in/LnJActNRulStaActs.php`, a single
    unpaginated page. `include_pattern` targets the `backendportal/uploads/
    acts/` path specifically (not a generic `act` match) because this same
    host also serves unrelated PDFs from its site-wide nav (e.g. an RTI pay
    structure PDF) that a broad pattern would wrongly pick up.
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="jk_law_acts", authority="Law Department, Union Territory of Jammu and Kashmir",
            urls=("https://law.jk.gov.in/LnJActNRulStaActs.php",),
            document_type="bare_act",
            include_pattern=r"backendportal/uploads/acts",
            jurisdiction_code="JK",
            expected_domain_suffixes=("law.jk.gov.in",),
        ), fetcher)


class JharkhandActsRulesPoliciesAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, real cross-department listing at
    `jharkhand.gov.in/Home/DocumentList?doctype=...&subdoctype=...` (the
    doctype/subdoctype query values select the "Acts, Rules & Policies"
    category; confirmed by direct fetch, not guessed from the UI). Anchor is
    an icon-only "view" button (`<a href="/Home/ViewDoc?id=...">`), so
    `_parse`'s record-text fallback (the surrounding `<tr>`, which carries
    the department name, title, and category columns) supplies the title --
    same pattern as the UP Ordinances page. `document_type="unknown"` because
    this single category genuinely mixes Acts, Rules, and Policies (its own
    page label is "Acts, Rules & Policies (Acts & Rules)"), not because the
    page is unidentified.

    132 items over 14 pages total (confirmed live: the pagination links go up
    to `page=14`, and page 2 fetched WITH the doctype/subdoctype params still
    returns the same category's rows 11-13, continuing the sequence -- so,
    unlike the bare `page=N` links rendered in the page body (which drop the
    filter), passing all three params together on each page does work and
    keeps the listing scoped). `urls` enumerates all 14 pages explicitly, the
    same pattern as `IndiaCodeAdapter`.
    """

    def __init__(self, fetcher: Any = None) -> None:
        base = (
            "https://jharkhand.gov.in/Home/DocumentList"
            "?doctype=c81e728d9d4c2f636f067f89cc14862c&subdoctype=c4ca4238a0b923820dcc509a6f75849b"
        )
        urls = (base,) + tuple(f"{base}&page={page}" for page in range(2, 15))
        super().__init__(CatalogueConfig(
            name="jharkhand_acts_rules_policies", authority="Government of Jharkhand",
            urls=urls,
            document_type="unknown",
            include_pattern=r"act|rule|polic",
            jurisdiction_code="JH",
            expected_domain_suffixes=("jharkhand.gov.in",),
        ), fetcher)


class MeghalayaActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, real `<li><a href="/acts/content/
    N">Act Title</a>...<small>Year: YYYY</small>...</li>` rows (Drupal views
    list) on `meghalaya.gov.in/acts`, paginated (`?page=0..13`, enumerated
    explicitly in `urls` like Assam). Anchor text IS the real title, so no
    fallback is needed for the listing hop itself.

    Two-hop like Assam (`resolve_pdf_links=True`): each detail page's real
    PDF is hosted on the CONCERNED DEPARTMENT's own separate `.gov.in`
    domain, not `meghalaya.gov.in` -- confirmed two different ones
    (`megurban.gov.in`, `megcooperation.gov.in`) for two different acts, so
    `expected_domain_suffixes` is deliberately left empty (no single expected
    host, same reasoning as Jharkhand's cross-department listing). Coverage
    is partial by construction: at least one department's PDF was observed
    served over plain HTTP (`megcooperation.gov.in`), which `official_url()`
    correctly refuses (HTTPS-only) -- that candidate is silently dropped by
    `_resolve_pdf_links`, not an adapter bug, just a real gap in what this
    engine can reach until that department serves HTTPS.
    """

    def __init__(self, fetcher: Any = None) -> None:
        urls = ("https://meghalaya.gov.in/acts",) + tuple(
            f"https://meghalaya.gov.in/acts?page={page}" for page in range(1, 14)
        )
        super().__init__(CatalogueConfig(
            name="meghalaya_acts", authority="Government of Meghalaya",
            urls=urls,
            document_type="bare_act",
            include_pattern=r"act",
            resolve_pdf_links=True,
            jurisdiction_code="ML",
        ), fetcher)


class PuducherryLawPublicationsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, real `<tr><td>The Puducherry Code
    Volume-N (Language)</td><td><a class="fas fa-download"
    href="./docs/CodeN.pdf"></a></td></tr>` rows on
    `law.py.gov.in/publications.html`. Anchor is icon-only (no visible text
    at all -- an empty `<a>`, not even a label), so `_parse`'s record-text
    fallback (the `<tr>`, carrying the volume title) supplies the title, same
    pattern as UP Ordinances. `include_pattern` targets the specific
    `docs/[T]codeN.pdf` filename shape (8 real English/Tamil "Puducherry
    Code" compiled-Acts volumes) because this same page/site also links many
    unrelated administrative PDFs (org chart, citizen's charter, etc.) a
    broad `act`/`\\.pdf` pattern would wrongly pick up.
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="py_law_publications", authority="Law Department, Union Territory of Puducherry",
            urls=("https://law.py.gov.in/publications.html",),
            document_type="bare_act",
            include_pattern=r"docs/t?code",
            jurisdiction_code="PY",
            expected_domain_suffixes=("law.py.gov.in",),
        ), fetcher)


class TripuraActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, 164 real
    `<a href="Tripura State Lagislation Acts/Act Name.pdf"><b>Act Title
    [icon glyph]</b></a>` anchors on `thc.nic.in/tsl_acts.html` -- the
    Tripura High Court's own hosted "Tripura State Legislation" library
    (Tripura's own Law & Parliamentary Affairs Dept page,
    `law.tripura.gov.in/acts_of_tripura`, returns HTTP 404). No `<tr>`/`<li>`/
    `<article>` ancestor and none needed: the anchor's own visible text IS
    the real Act title (a trailing private-use-area icon-font glyph rides
    along harmlessly). Single unpaginated page.
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="tripura_hc_acts_library", authority="Tripura High Court (Tripura State Legislation Library)",
            urls=("https://thc.nic.in/tsl_acts.html",),
            document_type="bare_act",
            include_pattern=r"act|\.pdf",
            jurisdiction_code="TR",
            expected_domain_suffixes=("thc.nic.in",),
        ), fetcher)


class UttarakhandActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, 195 real `<tr><td><a
    href="https://cdnbbsr.s3waas.gov.in/.../....pdf">Act Title (Act No. N of
    YYYY)</a></td>...</tr>` rows on
    `highcourtofuttarakhand.gov.in/uttarakhand-act/` -- Uttarakhand's own
    portals (`uk.gov.in/pages/display/1061-acts`: soft-404 template;
    `slsa.uk.gov.in`: HTTP 503) were not usable, so this reuses the High
    Court's library, the same pattern as Maharashtra/Tripura. `<tr>` ancestor
    present, and the anchor's own visible text IS the real title (an icon
    span carries no text), so no fallback is needed. `expected_domain_suffixes`
    is left empty: most PDFs resolve to the shared `cdnbbsr.s3waas.gov.in`
    NIC CDN (a different host from the listing page), and a handful of links
    point at a bare IP that `official_url()` correctly rejects (no
    `.gov.in`/`.nic.in` suffix) rather than a second real domain to name.
    Single unpaginated page.
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="uttarakhand_hc_acts", authority="High Court of Uttarakhand (Uttarakhand Acts Library)",
            urls=("https://highcourtofuttarakhand.gov.in/uttarakhand-act/",),
            document_type="bare_act",
            include_pattern=r"act|\.pdf",
            jurisdiction_code="UT",
        ), fetcher)


class ManipurActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, 42 real `<tr><td>N</td><td></td>
    <td><p>Act Title</p></td><td><p>Date</p></td><td></td><td><div
    class="btn-holder">...<a aria-label="Download Act Title.pdf" href="...
    .pdf" class="btn btn-download"><svg>...</svg><span class="text">
    <span class="name">Download</span><span class="filesize">N KB</span>
    </span></a></div></td></tr>` rows on
    `assembly.mn.gov.in/acts/acts-enacted-through-bills`, single unpaginated
    page. `<tr>` ancestor IS found correctly, but the anchor's own visible
    text is the generic label "Download N KB" -- `_GENERIC_ANCHOR_LABEL` (see
    `kb_central_adapters.py`) makes `_parse` fall back to the row text (which
    carries the real title in a sibling `<td>`) instead of that label,
    confirmed live after the 2026-09-22 fix.
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="manipur_assembly_acts", authority="Manipur Legislative Assembly",
            urls=("https://assembly.mn.gov.in/acts/acts-enacted-through-bills",),
            document_type="bare_act",
            include_pattern=r"\.pdf",
            jurisdiction_code="MN",
            expected_domain_suffixes=("assembly.mn.gov.in",),
        ), fetcher)


class NagalandActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, 349 real `<tr><td>Act Title</td>
    <td>ACT|AMENDMENT</td><td>Year</td><td><a href="....pdf">Download</a>
    </td><td><a href="....zip">Download</a></td></tr>` rows on
    `nagaland.gov.in/act-rules` (note the singular path -- the plural
    `/acts-rules` search engines surface is HTTP 404), single unpaginated
    page, not JS-rendered despite the client-side sort/search widget loaded.
    `<tr>` ancestor found correctly; anchor's own text is the generic label
    "Download" -- same `_GENERIC_ANCHOR_LABEL` fallback as Manipur.
    `include_pattern` requires `.pdf` specifically (not a broader `act`
    match) so the row's SECOND anchor, a `.zip` companion file sharing the
    same row text, is not also picked up as a separate spurious candidate.
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="nagaland_act_rules", authority="Government of Nagaland",
            urls=("https://nagaland.gov.in/act-rules",),
            document_type="bare_act",
            include_pattern=r"\.pdf",
            jurisdiction_code="NL",
            expected_domain_suffixes=("nagaland.gov.in",),
        ), fetcher)


class LadakhActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, a real `<tr><td role="rowheader">
    Act Title</td><td> </td><td><div><span class="pdf-downloads"><a
    aria-label="View, Act Title PDF N KB..." href="https://cdnbbsr.s3waas.
    gov.in/.../....pdf">View</a>(N KB)...</span></div></td></tr>` table on
    `ladakh.gov.in/document-category/acts-rules/` (only 3 documents
    currently published -- inherited J&K Reorganisation Act material, since
    Ladakh's own UT legislative apparatus is new). `<tr>` ancestor found
    correctly; anchor's own text is the generic label "View" --
    `_GENERIC_ANCHOR_LABEL` fallback applies, same as Manipur/Nagaland.
    `expected_domain_suffixes` names the resolved PDF host
    (`s3waas.gov.in`, the shared NIC document CDN), not the listing page's
    own `ladakh.gov.in` host, since every real PDF resolves there.
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="ladakh_acts_rules", authority="Union Territory of Ladakh",
            urls=("https://ladakh.gov.in/document-category/acts-rules/",),
            document_type="bare_act",
            include_pattern=r"\.pdf",
            jurisdiction_code="LA",
            expected_domain_suffixes=("s3waas.gov.in",),
        ), fetcher)


class OdishaActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, a real paginated (`?page=0..84`,
    85 pages, enumerated explicitly in `urls` like Assam/Jharkhand) listing
    at `law.odisha.gov.in/publication/acts-and-ordinances/acts` -- almost
    certainly the actual source of this KB's existing 1,478-document Odisha
    corpus (a prior one-time ingestion, not run through this adapter
    engine). Row shape: `<tr><td class="views-field-counter">N</td>
    <td class="views-field-title">Act Title</td><td class="views-field-
    nothing"><span><a href="....pdf" title="Act Title">Download(N KB)<img
    .../></a></span></td></tr>`. `<tr>` ancestor found correctly; anchor's
    own text is the generic label "Download(N KB)" -- `_GENERIC_ANCHOR_LABEL`
    fallback applies, same class as Manipur/Nagaland/Ladakh. This was the
    single highest-priority target for that fix, given the existing corpus
    size.
    """

    def __init__(self, fetcher: Any = None) -> None:
        base = "https://law.odisha.gov.in/publication/acts-and-ordinances/acts"
        urls = (base,) + tuple(f"{base}?page={page}" for page in range(1, 85))
        super().__init__(CatalogueConfig(
            name="odisha_law_dept_acts", authority="Law Department, Government of Odisha",
            urls=urls,
            document_type="bare_act",
            include_pattern=r"\.pdf",
            jurisdiction_code="OR",
            expected_domain_suffixes=("law.odisha.gov.in",),
        ), fetcher)


class ChhattisgarhActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22 against `law.cgstate.gov.in/act-details`,
    using the page's own server-side filter (`?notice_type[]=act`) -- the
    unfiltered page mixes Acts with unrelated notifications (e.g. Advocate
    General appointment orders); the filter narrows it to real Acts only,
    confirmed by fetching both and comparing (unfiltered: 6 mixed cards on
    one page; filtered: 2 real Acts, clean titles, no notifications).

    This is the page that motivated `_nearby_record_text`'s bounded `<div>`-
    ancestor fallback in `kb_central_adapters.py`: each item is a
    `div.act-card` with NO `<tr>`/`<li>`/`<article>` ancestor at all -- title
    lives in `div.act-content > div.act-title-box > p.act-title`, a SIBLING
    of `div.act-btn-grp > a` (icon-only-in-effect: the anchor's own text is
    just the language-code button label "ENG"/"HIN", handled either way by
    `_GENERIC_ANCHOR_LABEL`, but there was previously no ancestor at all to
    fall back TO). No pagination observed on the filtered page.
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="chhattisgarh_law_acts",
            authority="Law and Legislative Affairs Department, Government of Chhattisgarh",
            urls=("https://law.cgstate.gov.in/act-details?search=&notice_type%5B%5D=act",),
            document_type="bare_act",
            include_pattern=r"\.pdf",
            jurisdiction_code="CT",
            expected_domain_suffixes=("law.cgstate.gov.in",),
        ), fetcher)


class MaharashtraActsAdapter(OfficialCatalogueAdapter):
    """Confirmed live 2026-09-22: HTTP 200, 532 real `<a href=".../YYYY.NN.pdf">
    Some Act Name, Bombay/Maharashtra YYYY</a>` entries on one single page (no
    pagination to walk). Anchor text IS the Act title here, unlike
    `lj.maharashtra.gov.in`'s icon-only links, so no title fallback is needed.
    `expected_domain_suffixes` is set: every PDF resolves to this same host
    (relative `./YYYY.NN.pdf` links), matching every existing `mh_acts_*.pdf`
    document's recorded `source_url`.
    """

    def __init__(self, fetcher: Any = None) -> None:
        super().__init__(CatalogueConfig(
            name="mh_bombay_hc_acts_library", authority="Bombay High Court Library (Maharashtra Acts)",
            urls=("https://bombayhighcourt.gov.in/bhc/libweb/legislation/acts/listofmahacts.html",),
            document_type="bare_act",
            include_pattern=r"act|\.pdf",
            jurisdiction_code="MH",
            expected_domain_suffixes=("bombayhighcourt.gov.in",),
        ), fetcher)


def state_source_adapters() -> list[OfficialCatalogueAdapter]:
    return [
        UttarPradeshActsAdapter(), UttarPradeshOrdinancesAdapter(), MaharashtraActsAdapter(),
        AssamActsAdapter(), JammuKashmirActsAdapter(), JharkhandActsRulesPoliciesAdapter(),
        MeghalayaActsAdapter(), PuducherryLawPublicationsAdapter(), TripuraActsAdapter(),
        UttarakhandActsAdapter(), ManipurActsAdapter(), NagalandActsAdapter(),
        LadakhActsAdapter(), OdishaActsAdapter(), ChhattisgarhActsAdapter(),
        MadhyaPradeshActsAdapter(), DelhiActsAdapter(), DadraNagarHaveliDamanDiuActsAdapter(),
    ]
