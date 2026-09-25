from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.services.kb_high_court_adapters import (
    BombayHighCourtJudgmentsAdapter,
    DelhiHighCourtJudgmentsAdapter,
    high_court_source_adapters,
)
from app.services.kb_state_adapters import (
    AssamActsAdapter,
    ChhattisgarhActsAdapter,
    DadraNagarHaveliDamanDiuActsAdapter,
    DelhiActsAdapter,
    JammuKashmirActsAdapter,
    JharkhandActsRulesPoliciesAdapter,
    LadakhActsAdapter,
    MadhyaPradeshActsAdapter,
    MaharashtraActsAdapter,
    ManipurActsAdapter,
    MeghalayaActsAdapter,
    NagalandActsAdapter,
    OdishaActsAdapter,
    PuducherryLawPublicationsAdapter,
    TripuraActsAdapter,
    UttarakhandActsAdapter,
    UttarPradeshActsAdapter,
    UttarPradeshOrdinancesAdapter,
    state_source_adapters,
)


def test_state_adapters_are_registered_with_the_right_jurisdiction() -> None:
    adapters = state_source_adapters()
    assert {adapter.name for adapter in adapters} == {
        "up_acts", "up_ordinances", "mh_bombay_hc_acts_library", "assam_acts",
        "jk_law_acts", "jharkhand_acts_rules_policies", "meghalaya_acts",
        "py_law_publications", "tripura_hc_acts_library", "uttarakhand_hc_acts",
        "manipur_assembly_acts", "nagaland_act_rules", "ladakh_acts_rules",
        "odisha_law_dept_acts", "chhattisgarh_law_acts", "mp_code_state_acts",
        "delhi_law_dept_centralized_cos", "ddd_acts_rules",
    }
    by_name = {adapter.name: adapter for adapter in adapters}
    assert by_name["up_acts"].config.jurisdiction_code == "UP"
    assert by_name["up_ordinances"].config.jurisdiction_code == "UP"
    assert by_name["mh_bombay_hc_acts_library"].config.jurisdiction_code == "MH"
    assert by_name["assam_acts"].config.jurisdiction_code == "AS"
    assert by_name["jk_law_acts"].config.jurisdiction_code == "JK"
    assert by_name["jharkhand_acts_rules_policies"].config.jurisdiction_code == "JH"
    assert by_name["meghalaya_acts"].config.jurisdiction_code == "ML"
    assert by_name["py_law_publications"].config.jurisdiction_code == "PY"
    assert by_name["tripura_hc_acts_library"].config.jurisdiction_code == "TR"
    assert by_name["uttarakhand_hc_acts"].config.jurisdiction_code == "UT"
    assert by_name["manipur_assembly_acts"].config.jurisdiction_code == "MN"
    assert by_name["nagaland_act_rules"].config.jurisdiction_code == "NL"
    assert by_name["ladakh_acts_rules"].config.jurisdiction_code == "LA"
    assert by_name["odisha_law_dept_acts"].config.jurisdiction_code == "OR"
    assert by_name["chhattisgarh_law_acts"].config.jurisdiction_code == "CT"
    assert by_name["mp_code_state_acts"].config.jurisdiction_code == "MP"
    assert by_name["delhi_law_dept_centralized_cos"].config.jurisdiction_code == "DL"
    assert by_name["ddd_acts_rules"].config.jurisdiction_code == "DH"


def test_verified_high_court_adapters_are_registered() -> None:
    adapters = high_court_source_adapters()
    assert {adapter.name for adapter in adapters} == {
        "delhi_high_court_judgments", "bombay_high_court_judgments",
    }
    bombay = next(a for a in adapters if a.name == "bombay_high_court_judgments")
    assert bombay.config.jurisdiction_code == "MH"
    assert bombay.config.applicable_state_codes == ("GA", "DH")


def test_delhi_high_court_candidate_is_bound_to_delhi() -> None:
    html = b"""<html><table><tr><td>20 Aug 2026</td><td>
      <a href="/web/judgement/123?casepdf=1">Download judgment</a>
    </td></tr></table></html>"""
    candidates, _ = asyncio.run(DelhiHighCourtJudgmentsAdapter(
        fetcher=AsyncMock(return_value=(html, "text/html")),
    ).discover(None))
    assert len(candidates) == 1
    assert candidates[0].jurisdiction_code == "DL"
    assert candidates[0].document_type == "case_law"


def test_bombay_high_court_candidate_covers_shared_jurisdictions() -> None:
    html = b"""<html><table><tr><td>01-09-2026</td><td>
      <a href="/generatenewauth.php?bhcpar=abc">Reported Judgment</a>
    </td></tr></table></html>"""
    candidates, _ = asyncio.run(BombayHighCourtJudgmentsAdapter(
        fetcher=AsyncMock(return_value=(html, "text/html")),
    ).discover(None))
    assert len(candidates) == 1
    assert candidates[0].jurisdiction_code == "MH"
    assert candidates[0].applicable_state_codes == ("GA", "DH")


def test_up_acts_adapter_extracts_state_tagged_confident_candidate() -> None:
    html = b"""
    <html><body><table><tr><td>01-01-2026</td><td>
      <a href="/files/act-9-hindi.pdf">Uttar Pradesh Act No. 9 of 2026</a>
    </td></tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = UttarPradeshActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.jurisdiction_code == "UP"
    assert candidate.act_number == "9"
    # Real date + act number + .pdf + expected domain -> a high, explainable score.
    assert candidate.confidence_score >= 0.7


def test_up_ordinances_adapter_classifies_ordinance_change_type() -> None:
    html = b"""
    <html><body><table><tr><td>02-02-2026</td><td>
      <a href="/files/ordinance-3.pdf">Uttar Pradesh Ordinance No. 3 of 2026</a>
    </td></tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = UttarPradeshOrdinancesAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    assert candidates[0].change_type == "ordinance"


def test_maharashtra_acts_adapter_extracts_the_real_page_shape() -> None:
    # Byte-for-byte the row this Act's official listing carries -- confirmed
    # live 2026-09-22 against bombayhighcourt.gov.in's own Maharashtra Acts
    # library index. This is the SAME URL the already-indexed
    # `mh_acts_1936.04_payment-of-wages-act-1936.pdf` (source_url
    # `.../acts/1936.04.pdf`) was ingested from, by a prior one-time process
    # this adapter did not run.
    html = b"""
    <html><body><table><tr>
      <td style="vertical-align: top; width: 848px; font-family: arial;"><font
      size="-1"><a style="text-decoration: none;" target="_blank"
      href="./1936.04.pdf">Payment of Wages Act, 1936</a></font></td>
      <td style="vertical-align: top;"><br></td>
    </tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = MaharashtraActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == "https://bombayhighcourt.gov.in/bhc/libweb/legislation/acts/1936.04.pdf"
    assert candidate.title == "Payment of Wages Act, 1936"
    assert candidate.jurisdiction_code == "MH"
    assert candidate.document_type == "bare_act"


def test_maharashtra_acts_adapter_rejects_a_non_official_lookalike_domain() -> None:
    html = (
        b'<html><body><a href="https://not-bombayhighcourt.example.com/act.pdf">'
        b"Some Act, 1990</a></body></html>"
    )
    fetcher = AsyncMock(return_value=(html, "text/html"))
    candidates, _ = asyncio.run(MaharashtraActsAdapter(fetcher=fetcher).discover(None))
    assert candidates == []


def test_assam_acts_adapter_walks_listing_then_detail_page_to_the_real_pdf() -> None:
    # Byte-for-byte the listing-page anchor and the detail-page PDF anchor,
    # confirmed live 2026-09-22 against legislative.assam.gov.in/documents/
    # assam-acts. Two-hop shape (listing -> detail page -> PDF), unlike
    # Maharashtra's single-hop listing -- `resolve_pdf_links=True` fetches
    # the detail page as a second call.
    listing_html = b"""
    <html><body><div class="view-content">
      <a href="/documents-detail/the-assam-jan-vishwas-amendment-of-provisions-act-2026-assam-act-noxiii-of-2026">
        The Assam Jan Vishwas (Amendment of Provisions) Act, 2026 (Assam Act No.XIII of 2026)
      </a>
    </div></body></html>
    """
    detail_html = b"""
    <html><body>
      <a href="https://legislative.assam.gov.in/sites/default/files/swf_utility_folder/departments/legislative_medhassu_in_oid_3/menu/document/the_assam_jan_vishwas_amendment_of_provisions_act_2026_assam_act_no.xiii_of_2026_compressed.pdf"
         type="application/pdf; length=5274239">the_assam_jan_vishwas_amendment_of_provisions_act_2026_assam_act_no.xiii_of_2026_compressed.pdf</a>
    </body></html>
    """
    fetcher = AsyncMock(side_effect=[(listing_html, "text/html"), (detail_html, "text/html")])
    adapter = AssamActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == (
        "https://legislative.assam.gov.in/sites/default/files/swf_utility_folder/departments/"
        "legislative_medhassu_in_oid_3/menu/document/"
        "the_assam_jan_vishwas_amendment_of_provisions_act_2026_assam_act_no.xiii_of_2026_compressed.pdf"
    )
    assert candidate.jurisdiction_code == "AS"
    assert candidate.document_type == "bare_act"


def test_assam_acts_adapter_rejects_a_non_official_lookalike_domain() -> None:
    html = (
        b'<html><body><a href="https://not-legislative-assam.example.com/documents-detail/act-2026">'
        b"Some Act, 2026</a></body></html>"
    )
    fetcher = AsyncMock(return_value=(html, "text/html"))
    candidates, _ = asyncio.run(AssamActsAdapter(fetcher=fetcher).discover(None))
    assert candidates == []


def test_jk_law_acts_adapter_extracts_the_real_page_shape() -> None:
    # Byte-for-byte the row shape confirmed live 2026-09-22 against
    # law.jk.gov.in/LnJActNRulStaActs.php.
    html = b"""
    <html><body><table><tr class="trsearch">
      <td align="center" class="sno-cell" valign="middle">01</td>
      <td align="left" class="rule-title" valign="middle">
        <b><a href="jklawact/backendportal/uploads/acts/act15_1770488504.pdf" target="_blank">
          Aerial Ropeways Act, 2002
        </a></b>
      </td>
    </tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = JammuKashmirActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == "https://law.jk.gov.in/jklawact/backendportal/uploads/acts/act15_1770488504.pdf"
    assert candidate.title == "Aerial Ropeways Act, 2002"
    assert candidate.jurisdiction_code == "JK"


def test_jk_law_acts_adapter_ignores_unrelated_site_pdfs() -> None:
    # The same host also serves site-wide nav PDFs (e.g. an RTI pay-structure
    # document) outside the `backendportal/uploads/acts/` path -- confirmed
    # live these are NOT real Acts and must not be picked up.
    html = b"""<html><body><ul>
      <li><a href="jklawData/RTI/Details of Pay Level of Officer.pdf" target="_blank">Pay Structure</a></li>
    </ul></body></html>"""
    fetcher = AsyncMock(return_value=(html, "text/html"))
    candidates, _ = asyncio.run(JammuKashmirActsAdapter(fetcher=fetcher).discover(None))
    assert candidates == []


def test_jharkhand_acts_rules_policies_adapter_extracts_the_real_page_shape() -> None:
    # Byte-for-byte the row shape confirmed live 2026-09-22 against
    # jharkhand.gov.in/Home/DocumentList?doctype=...&subdoctype=... . The
    # anchor is icon-only, so the surrounding <tr>'s other columns supply
    # the title, same fallback pattern as the UP Ordinances page.
    html = b"""
    <html><body><table><tr class="darkMode">
      <td>1</td>
      <td>Jharkhand State Agricultural Marketing Board</td>
      <td>Sewa Viniyamavali</td>
      <td>Sewa Viniyamavali</td>
      <td>01/10/1978</td>
      <td>Acts, Rules &amp; Policies (Acts &amp; Rules)</td>
      <td><a class="btn btn-info" href="/Home/ViewDoc?id=JSAMBDO002SD00119062026122400489"
             onclick="basicPopup(this.href);return false" title="click here to view">
        <i aria-hidden="true" class="fa fa-eye"></i>
      </a></td>
    </tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = JharkhandActsRulesPoliciesAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == "https://jharkhand.gov.in/Home/ViewDoc?id=JSAMBDO002SD00119062026122400489"
    assert "Jharkhand State Agricultural Marketing Board" in candidate.title
    assert candidate.jurisdiction_code == "JH"
    assert candidate.document_type == "unknown"


def test_meghalaya_acts_adapter_walks_listing_then_detail_page_to_the_real_pdf() -> None:
    # Byte-for-byte the listing-page row and detail-page PDF anchor,
    # confirmed live 2026-09-22 against meghalaya.gov.in/acts. Two-hop like
    # Assam, but the resolved PDF lives on a DIFFERENT department domain
    # (megurban.gov.in), not meghalaya.gov.in.
    listing_html = b"""
    <html><body><ul>
      <li><span class="views-field views-field-title"><span class="field-content">
        <a href="/acts/content/46555">The Meghalaya Building (Amendment) Byelaws, 2024</a>
      </span></span> <span class="views-field views-field-field-year field-color">
        <span class="field-content"><small>Year: 2024</small></span>
      </span></li>
    </ul></body></html>
    """
    detail_html = b"""
    <html><body>
      <div class="views-field views-field-nothing"><span class="field-content">
        <a href="https://megurban.gov.in/laws/Building%20(Amendment)%20Byelaws,%202024.pdf" class="d-align" target="_blank">
          <em>Click here to View/Download.</em>
        </a>
      </span></div>
    </body></html>
    """
    fetcher = AsyncMock(side_effect=[(listing_html, "text/html"), (detail_html, "text/html")])
    adapter = MeghalayaActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == "https://megurban.gov.in/laws/Building%20(Amendment)%20Byelaws,%202024.pdf"
    assert candidate.title == "The Meghalaya Building (Amendment) Byelaws, 2024"
    assert candidate.jurisdiction_code == "ML"


def test_py_law_publications_adapter_extracts_the_real_page_shape() -> None:
    # Byte-for-byte the row shape confirmed live 2026-09-22 against
    # law.py.gov.in/publications.html. Anchor is fully icon-only (no text
    # node at all), so the surrounding <tr> supplies the title.
    html = b"""
    <html><body><table>
      <tr>
        <td>The Puducherry Code Volume-I (English Version)</td>
        <td class="text-center"><a class="fas fa-download" href="./docs/Code1.pdf" style="color:#039d9d" target="_blank"></a></td>
      </tr>
      <tr>
        <td>Organization Chart</td>
        <td><a href="./docs/Org_chart.pdf" target="_blank">Organization Chart</a></td>
      </tr>
    </table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = PuducherryLawPublicationsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == "https://law.py.gov.in/docs/Code1.pdf"
    assert candidate.title == "The Puducherry Code Volume-I (English Version)"
    assert candidate.jurisdiction_code == "PY"


def test_tripura_acts_adapter_extracts_the_real_page_shape() -> None:
    # Byte-for-byte the anchor shape confirmed live 2026-09-22 against
    # thc.nic.in/tsl_acts.html. No <tr>/<li>/<article> ancestor -- none
    # needed, since the anchor's own text IS the real title.
    html = b"""
    <html><body>
      <a href="Tripura State Lagislation Acts/Tripura Amusement Tax Act 1973.pdf" target="_blank">
        <b>Amusement Tax Act,Tripura,1973</b>
      </a>
    </body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = TripuraActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == "https://thc.nic.in/Tripura State Lagislation Acts/Tripura Amusement Tax Act 1973.pdf"
    assert candidate.title == "Amusement Tax Act,Tripura,1973"
    assert candidate.jurisdiction_code == "TR"


def test_uttarakhand_acts_adapter_extracts_the_real_page_shape() -> None:
    # Byte-for-byte the row shape confirmed live 2026-09-22 against
    # highcourtofuttarakhand.gov.in/uttarakhand-act/.
    html = b"""
    <html><body><table><tr>
      <td><a href="https://cdnbbsr.s3waas.gov.in/s3bc7f621451b4f5df308a8e098112185d/uploads/2025/03/20250312589803757.pdf">
        Aadhar (Targeted Delivery of Financial and Other Subsidies, Benefits and Services) Act, 2017 (Act No. 04 of 2018)
        <span aria-hidden="true" class="icon-pdf pdf-icon"></span>
      </a></td>
      <td></td>
      <td></td>
    </tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = UttarakhandActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == (
        "https://cdnbbsr.s3waas.gov.in/s3bc7f621451b4f5df308a8e098112185d/uploads/2025/03/20250312589803757.pdf"
    )
    assert "Aadhar" in candidate.title
    assert candidate.jurisdiction_code == "UT"


def test_manipur_acts_adapter_prefers_row_title_over_generic_download_label() -> None:
    # Byte-for-byte the row shape confirmed live 2026-09-22 against
    # assembly.mn.gov.in/acts/acts-enacted-through-bills. The anchor's own
    # text is "Download 926.24 KB" (a generic button label); the real title
    # is a sibling <td> -- the exact case `_GENERIC_ANCHOR_LABEL` fixes.
    html = b"""
    <html><body><table><tr>
      <td>1</td>
      <td></td>
      <td><p>The Manipur Contingency Fund of the Union Territory (Determination of Amount) Act, 1964</p></td>
      <td><p>11 August, 1964</p></td>
      <td></td>
      <td>
        <div class="btn-holder inline-flex flex-col">
          <a href="/user/pages/files/acts/enacted/The Manipur Contingency Fund of the Union Territory (Determination of Amount) Act, 1964.pdf"
             target="_blank" aria-label="Download ... .pdf" class="btn btn-download">
            <svg><use href="#file-pdf" /></svg>
            <span class="text"><span class="name">Download</span><span class="filesize">926.24 KB</span></span>
          </a>
        </div>
      </td>
    </tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = ManipurActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    # The generic "Download 926.24 KB" button label no longer wins outright
    # -- the real title (from the sibling <td>) is present in the fallback
    # row text used as the title.
    assert "The Manipur Contingency Fund" in candidates[0].title
    assert candidates[0].jurisdiction_code == "MN"


def test_nagaland_acts_adapter_ignores_the_zip_companion_link() -> None:
    # Byte-for-byte the row shape confirmed live 2026-09-22 against
    # nagaland.gov.in/act-rules. Two anchors share the same row: a real PDF
    # and a .zip companion -- only the .pdf one should become a candidate.
    html = b"""
    <html><body><table><tr>
      <td>Child Labour (Prohibition and Regulation) Amendment Act 2016.</td>
      <td>ACT</td>
      <td>2016</td>
      <td><a href="/storage/PostFiles/Public-Information-on-Child-Labour-Amendment-Act.pdf" target="_blank">Download</a></td>
      <td><a href="/storage/PostFiles/Child-Labour.zip" target="_blank">Download</a></td>
    </tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = NagalandActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    assert candidates[0].url.endswith(".pdf")
    assert "Child Labour" in candidates[0].title
    assert candidates[0].jurisdiction_code == "NL"


def test_ladakh_acts_adapter_prefers_row_title_over_generic_view_label() -> None:
    # Byte-for-byte the row shape confirmed live 2026-09-22 against
    # ladakh.gov.in/document-category/acts-rules/. The anchor's own text is
    # just "View" -- the real title is in the row's first <td>.
    html = b"""
    <html><body><table><tr>
      <td role="rowheader" scope="row">THE JAMMU AND KASHMIR REORGANISATION ACT, 2019- Appointed Day</td>
      <td> </td>
      <td>
        <div class="">
          <span class="pdf-downloads alternate">
            <span class="title">Accessible Version :</span>
            <a aria-label="View, THE JAMMU AND KASHMIR REORGANISATION ACT, 2019- Appointed Day PDF 243 KB"
               href="https://cdnbbsr.s3waas.gov.in/s395192c98732387165bf8e396c0f2dad2/uploads/2019/10/2019102957.pdf"
               target="_blank">View</a>(243 KB)
          </span>
        </div>
      </td>
    </tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = LadakhActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    assert "JAMMU AND KASHMIR REORGANISATION ACT" in candidates[0].title
    assert candidates[0].jurisdiction_code == "LA"


def test_odisha_acts_adapter_prefers_row_title_over_generic_download_label() -> None:
    # Byte-for-byte the row shape confirmed live 2026-09-22 against
    # law.odisha.gov.in/publication/acts-and-ordinances/acts. The anchor's
    # own text is "Download(495.23 KB)" -- the real title is a sibling <td>.
    html = b"""
    <html><body><table><tr>
      <td class="views-field views-field-counter" headers="view-counter-table-column">1</td>
      <td class="views-field views-field-title" headers="view-title-table-column">The Odisha Repealing Act, 2021</td>
      <td class="views-field views-field-nothing" headers="view-nothing-table-column">
        <span><a href="/sites/default/files/2026-09/Odisha%20Repealing%20Act.pdf" target="_blank"
                 title="The Odisha Repealing Act, 2021">Download(495.23 KB)<img alt="The Odisha Repealing Act, 2021"
                 src="https://law.odisha.gov.in/core/themes/classy/images/icons/application-pdf.png"/></a></span>
      </td>
    </tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = OdishaActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    assert "The Odisha Repealing Act, 2021" in candidates[0].title
    assert candidates[0].jurisdiction_code == "OR"


def test_chhattisgarh_acts_adapter_walks_the_div_only_card_to_the_real_title() -> None:
    # Byte-for-byte the act-card shape confirmed live 2026-09-22 against
    # law.cgstate.gov.in/act-details (filtered to notice_type=act). No
    # <tr>/<li>/<article> ancestor exists anywhere -- this is the page that
    # motivated `_nearby_record_text`'s bounded <div>-ancestor fallback.
    html = b"""
    <html><body>
    <div class="act-card mb-3">
      <div class="act-content">
        <div class="tag-date">
          <p class="act-tag">2013</p>
        </div>
        <div class="act-title-box">
          <p class="act-title">Chattisgarh Land Holdings (Validation) Act, 2013</p>
        </div>
      </div>
      <div class="act-btn-grp">
        <a class="act-button" href="https://law.cgstate.gov.in/custom-files/act_rules_files/ACT20250912_A_68c3aa261b393.pdf" target="_blank">
          ENG <i class="fas fa-download"></i>
        </a>
      </div>
    </div>
    </body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = ChhattisgarhActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == "https://law.cgstate.gov.in/custom-files/act_rules_files/ACT20250912_A_68c3aa261b393.pdf"
    assert "Chattisgarh Land Holdings (Validation) Act, 2013" in candidate.title
    assert candidate.jurisdiction_code == "CT"


def test_mp_code_state_acts_adapter_extracts_the_real_page_shape() -> None:
    # Byte-for-byte the row shape confirmed live 2026-09-22 against
    # code.mp.gov.in/stateacts.aspx (the rebuilt MP Code site).
    html = b"""
    <html><body><table><tr class="animate-row">
      <td valign="top" width="45%">
        <a href='/WriteReadData/Pdf/Madhya Pradesh Appropriation (No.05) Act 2025MPCode.pdf'
           target="_blank" class="doc-title-link">The Madhya Pradesh Appropriation (No.05) Act, 2025
          (Act No. 28 of 2025)</a>
      </td><td valign="top" width="130">18 Dec 2025</td><td valign="top">Finance Department</td>
      <td align="center" valign="middle" width="120"><span class="badge-doc-type">Acts</span></td>
      <td align="center" valign="middle" width="70">
        <a href='/WriteReadData/Pdf/Madhya Pradesh Appropriation (No.05) Act 2025MPCode.pdf'
           target="_blank" class="pdf-download-btn" title="Download PDF"><i class="bi bi-file-earmark-pdf-fill"></i></a>
      </td>
    </tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = MadhyaPradeshActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert "The Madhya Pradesh Appropriation (No.05) Act, 2025" in candidate.title
    assert candidate.jurisdiction_code == "MP"


def test_delhi_acts_adapter_scopes_to_the_centralized_cos_file_path() -> None:
    # Byte-for-byte the real row shape confirmed live 2026-09-22 against
    # delhi.gov.in/centralized-cos (LAW-domain filtered). The same raw
    # response also carries site-wide mega-menu links to unrelated
    # delhi.gov.in PDFs (MLA lists, e-auction notices) under DIFFERENT file
    # paths -- include_pattern must reject those while keeping the real row.
    html = b"""
    <html><body>
    <a href="/sites/default/files/2025-03/list_of_membersnn.pdf">MLAs</a>
    <table><tr>
      <td class="views-field views-field-counter">1</td>
      <td class="views-field views-field-field-cos-no">Act(2026)/8/14</td>
      <td class="views-field views-field-title">F.14 (112)/LA-2026/ala1/89-100</td>
      <td class="views-field views-field-field-subject">THE DELHI URBAN SHELTER IMPROVEMENT BOARD (AMENDMENT) ACT, 2026 (DELHI ACT No. 11 OF 2026)</td>
      <td class="views-field views-field-field-cos-type">Acts</td>
      <td class="views-field views-field-field-domain-id">LAW</td>
      <td class="views-field views-field-field-date"><time datetime="2026-09-01T12:00:00Z">01-09-2026</time></td>
      <td><a class="tab-view" href="/sites/default/files/centralized-cos/delhi_act_no_11_of_year_2026.pdf" target="_blank" title=""> Download </a><span class="file_size">| 295.62 KB</span></td>
    </tr></table>
    </body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = DelhiActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.url == "https://delhi.gov.in/sites/default/files/centralized-cos/delhi_act_no_11_of_year_2026.pdf"
    assert "DELHI URBAN SHELTER IMPROVEMENT BOARD" in candidate.title
    assert candidate.jurisdiction_code == "DL"


def test_ddd_acts_adapter_extracts_the_real_page_shape() -> None:
    # Byte-for-byte the row shape confirmed live 2026-09-22 against
    # ddd.gov.in/document-category/acts-rules/ -- same NIC template as
    # Ladakh.
    html = b"""
    <html><body><table><tr>
      <td role="rowheader" scope="row">Dadra and Nagar Haveli and Daman &amp; Diu Rules, 2021</td>
      <td> 31/03/2021</td>
      <td>
        <div class=" ">
          <span class="pdf-downloads alternate">
            <span class="title">Accessible Version :</span>
            <a target="_blank" href="https://cdnbbsr.s3waas.gov.in/s371e09b16e21f7b6919bbfc43f6a5b2f0/uploads/2021/03/2021033180.pdf"
               aria-label="View, Dadra and Nagar Haveli and Daman &amp; Diu Rules, 2021 PDF 306 KB">View</a>(306 KB)
          </span>
        </div>
      </td>
    </tr></table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = DadraNagarHaveliDamanDiuActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert "Dadra and Nagar Haveli and Daman & Diu Rules, 2021" in candidate.title
    assert candidate.jurisdiction_code == "DH"


def test_amendment_and_repeal_are_classified_from_listing_text() -> None:
    html = b"""
    <html><body><table>
    <tr><td>03-03-2026</td><td><a href="/files/a1.pdf">Act No. 1 of 2026 (Amendment)</a></td></tr>
    <tr><td>04-04-2026</td><td><a href="/files/a2.pdf">Act No. 2 of 2026 (Repeal)</a></td></tr>
    </table></body></html>
    """
    fetcher = AsyncMock(return_value=(html, "text/html"))
    adapter = UttarPradeshActsAdapter(fetcher=fetcher)

    candidates, _ = asyncio.run(adapter.discover(None))

    by_id = {c.act_number: c.change_type for c in candidates}
    assert by_id["1"] == "amendment"
    assert by_id["2"] == "repeal"
