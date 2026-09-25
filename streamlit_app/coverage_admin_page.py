"""Admin-only State/UT knowledge-base coverage dashboard.

Calls the single composed `GET /admin/phase3/state-coverage` route -- the same
role checks (`require_admin`) and data the JSON dashboard consumer sees.
Nothing here mutates state; retrying a stuck job or reviewing a discovery
still happens through chat or the existing KB review page.
"""

from collections.abc import MutableMapping
from typing import Any

import auth_client
import httpx
import streamlit as st

_DOCUMENT_TYPES = [
    "bare_act", "rules", "regulation", "notification", "ordinance", "case_law", "circular", "order",
]


def _fetch(session: MutableMapping[str, Any], api_base_url: str) -> dict[str, Any] | None:
    try:
        response = auth_client.request(session, api_base_url, "GET", "/admin/phase3/state-coverage", timeout=30)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        st.error("Could not reach the server to load the coverage dashboard.")
        return None


def _fetch_readiness(session: MutableMapping[str, Any], api_base_url: str) -> dict[str, Any] | None:
    try:
        response = auth_client.request(session, api_base_url, "GET", "/admin/phase3/kb-rollout/readiness", timeout=30)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        st.error("Could not reach the server to load rollout readiness.")
        return None


def _fetch_sources(session: MutableMapping[str, Any], api_base_url: str) -> list[dict[str, Any]] | None:
    try:
        response = auth_client.request(session, api_base_url, "GET", "/admin/phase3/kb-sources", timeout=30)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        st.error("Could not reach the server to load registered sources.")
        return None


def _fetch_production_readiness(session: MutableMapping[str, Any], api_base_url: str) -> dict[str, Any] | None:
    try:
        response = auth_client.request(session, api_base_url, "GET", "/admin/phase3/kb-production/readiness", timeout=30)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        st.error("Could not reach the server to load production release gates.")
        return None


def _by_stage_tab(data: dict[str, Any]) -> None:
    coverage = data["coverage"]
    st.caption(
        f"{coverage['scope']['jurisdictions']} jurisdictions catalogued "
        f"({coverage['scope']['jurisdictions_with_indexed_documents']} have at least one indexed document)."
    )
    rows = [
        {
            "Code": row["code"], "Name": row["name"], "Status": row["status"],
            "Discovered": row.get("discovered", 0), "Downloaded": row.get("downloaded", 0),
            "Indexed docs": row["indexed_documents"], "Approved chunks": row["approved_chunks"],
            "Tested (pass/total)": f"{row.get('tested_passed', 0)}/{row.get('tested_total', 0)}",
            "Complete": "✅" if row["complete"] else "—",
        }
        for row in coverage["jurisdictions"]
    ]
    st.dataframe(rows, hide_index=True, use_container_width=True)

    st.markdown("#### Adapter health")
    automation = data["automation"]
    st.caption(
        f"Automation enabled: {automation['enabled']} · "
        f"exception queue (quarantined): {automation['exception_queue']} · "
        f"manual access required: {automation['dead_letter']['manual_access_required']}"
    )
    st.dataframe(automation["adapters"], hide_index=True, use_container_width=True)


def _exceptions_tab(data: dict[str, Any]) -> None:
    manual_jobs = data["manual_access_required"]
    st.markdown("#### Portals needing a human to fetch the document")
    if not manual_jobs:
        st.info("No portal is currently blocked behind a CAPTCHA or access wall.")
    else:
        for job in manual_jobs:
            candidate = job.get("candidate", {})
            with st.container(border=True):
                st.markdown(f"**{candidate.get('title', 'Untitled')}**")
                st.caption(f"{candidate.get('url')} · state: {candidate.get('jurisdiction_code')}")
                if job.get("error_detail"):
                    st.caption(f"Reason: {job['error_detail']}")
                st.caption(
                    "Fetch this document manually and upload it through the Knowledge Base "
                    "review flow — this job will not retry itself."
                )

    st.markdown("#### Recent portal layout-change alerts")
    alerts = data["layout_change_alerts"]
    if not alerts:
        st.info("No adapter has reported a layout change recently.")
    else:
        for event in alerts:
            details = event.get("details", {})
            st.caption(f"{event.get('created_at')} · {details.get('adapter')}: {details.get('error')}")


def _complete_tab(data: dict[str, Any]) -> None:
    coverage = data["coverage"]
    complete_rows = [row for row in coverage["jurisdictions"] if row["complete"]]
    st.caption(
        f"{len(complete_rows)} of {len(coverage['jurisdictions'])} jurisdictions are fully covered "
        "(approved and, where a benchmark applies, retrieval-tested)."
    )
    if coverage["manual_access_exceptions"]:
        st.warning(
            "Manual-access exceptions (excluded from automation expectations, not from completeness): "
            + ", ".join(coverage["manual_access_exceptions"])
        )
    if not complete_rows:
        st.info("No jurisdiction is fully covered yet.")
    else:
        st.dataframe(
            [{"Code": row["code"], "Name": row["name"], "Approved chunks": row["approved_chunks"]}
             for row in complete_rows],
            hide_index=True, use_container_width=True,
        )


def _readiness_tab(session: MutableMapping[str, Any], api_base_url: str) -> None:
    rollout = _fetch_readiness(session, api_base_url)
    if rollout is not None:
        st.markdown("#### Phase 4 rollout readiness (per jurisdiction)")
        banner = st.success if rollout["production_ready"] else st.warning
        banner(
            f"{rollout['ready_jurisdictions']} of {rollout['total_jurisdictions']} jurisdictions ready · "
            f"manual downloads required: {rollout['manual_downloads_required']}"
        )
        st.dataframe(
            [
                {
                    "Code": row["code"], "Name": row["name"], "Source": row["source_status"],
                    "Adapters": ", ".join(row["adapters"]) or "—",
                    "Source verified": "✅" if row["gates"]["official_source_verified"] else "—",
                    "Adapter healthy": "✅" if row["gates"]["adapter_healthy"] else "—",
                    "Corpus approved": "✅" if row["gates"]["approved_corpus_present"] else "—",
                    "Benchmark passed": "✅" if row["gates"]["retrieval_benchmark_passed"] else "—",
                    "No manual-access block": "✅" if row["gates"]["no_manual_access_exception"] else "—",
                    "Ready": "✅" if row["ready"] else "—",
                }
                for row in rollout["jurisdictions"]
            ],
            hide_index=True, use_container_width=True,
        )

    st.markdown("#### Phase 6 production release gates")
    st.caption("This report never deploys or publishes anything by itself — it only says whether release is blocked.")
    release = _fetch_production_readiness(session, api_base_url)
    if release is not None:
        banner = st.success if release["production_ready"] else st.error
        banner(f"release_status: {release['release_status']}")
        if release["blockers"]:
            st.markdown("**Blockers:** " + ", ".join(release["blockers"]))
        st.json(release["gates"])


def _sources_tab(session: MutableMapping[str, Any], api_base_url: str) -> None:
    st.markdown("#### Register a new official catalogue source")
    st.caption(
        "Registering only stores the configuration (`pending_probe`). Nothing is fetched until you "
        "probe it, and nothing joins the automation pipeline until you activate a probe that passed."
    )
    with st.form("register_kb_source_form", clear_on_submit=True):
        name = st.text_input("Adapter name (lowercase, e.g. `mh_gazette`)")
        authority = st.text_input("Issuing authority (e.g. \"Government of Maharashtra, Law and Judiciary Dept.\")")
        url = st.text_input("Official catalogue URL (https, .gov.in / .nic.in)")
        jurisdiction_code = st.text_input("Primary jurisdiction code (e.g. MH, DL)")
        applicable_state_codes = st.text_input("Extra state codes this source also covers (comma-separated, optional)")
        document_type = st.selectbox("Document type", _DOCUMENT_TYPES)
        include_pattern = st.text_input("Include pattern (regex matched against link text/URL)", value=r"act|rule|\.pdf")
        resolve_pdf_links = st.checkbox("Resolve detail-page links to their actual PDF files", value=False)
        authority_confirmed = st.checkbox(
            "I have personally opened this URL and confirmed it belongs to the named issuing authority", value=False,
        )
        submitted = st.form_submit_button("Register")
        if submitted:
            if not authority_confirmed:
                st.warning("Authority confirmation is required before a source can be registered.")
            else:
                payload = {
                    "name": name.strip(), "authority": authority.strip(), "url": url.strip(),
                    "jurisdiction_code": jurisdiction_code.strip().upper(),
                    "applicable_state_codes": [c.strip().upper() for c in applicable_state_codes.split(",") if c.strip()],
                    "document_type": document_type, "include_pattern": include_pattern,
                    "resolve_pdf_links": resolve_pdf_links, "authority_confirmed": authority_confirmed,
                }
                try:
                    response = auth_client.request(
                        session, api_base_url, "PUT", "/admin/phase3/kb-sources", json=payload, timeout=30,
                    )
                    response.raise_for_status()
                    st.toast(f"Registered **{name}** — probe it before activating.", icon="✅")
                    st.rerun()
                except httpx.HTTPError as exc:
                    st.error(f"Registration failed: {exc}")

    st.markdown("#### Registered sources")
    sources = _fetch_sources(session, api_base_url)
    if not sources:
        st.info("No custom sources registered yet.")
        return
    for source in sources:
        with st.container(border=True):
            st.markdown(f"**{source['_id']}** · {source.get('authority', '—')}")
            st.caption(
                f"{source.get('url')} · status: {source.get('status')} · "
                f"jurisdiction: {source.get('jurisdiction_code')}"
            )
            if source.get("last_probe_error"):
                st.caption(f"Last probe error: {source['last_probe_error']}")
            probe_col, activate_col = st.columns(2)
            if probe_col.button("🔍 Probe", key=f"probe_{source['_id']}"):
                try:
                    response = auth_client.request(
                        session, api_base_url, "POST",
                        f"/admin/phase3/kb-sources/{source['_id']}/probe", timeout=60,
                    )
                    response.raise_for_status()
                    st.toast(f"Probed {source['_id']}: {response.json().get('status')}", icon="🔍")
                    st.rerun()
                except httpx.HTTPError as exc:
                    st.error(f"Probe failed: {exc}")
            if activate_col.button(
                "🚀 Activate", key=f"activate_{source['_id']}",
                disabled=source.get("status") != "probe_passed",
            ):
                try:
                    response = auth_client.request(
                        session, api_base_url, "POST",
                        f"/admin/phase3/kb-sources/{source['_id']}/activate", timeout=30,
                    )
                    response.raise_for_status()
                    st.toast(f"Activated {source['_id']}.", icon="🚀")
                    st.rerun()
                except httpx.HTTPError as exc:
                    st.error(f"Activation failed: {exc}")


def render(session: MutableMapping[str, Any], api_base_url: str) -> None:
    st.markdown("## 🗺️ State/UT Coverage")
    st.caption(
        "Discover → download → index → test progress for every State, UT and Central jurisdiction, "
        "plus open exceptions, rollout/release readiness, and source onboarding."
    )

    tab_stage, tab_exceptions, tab_complete, tab_readiness, tab_sources = st.tabs(
        ["📊 By stage", "⚠️ Exceptions", "✅ Complete", "🚀 Readiness", "🔌 Sources"]
    )
    data = _fetch(session, api_base_url)
    with tab_stage:
        if data is not None:
            _by_stage_tab(data)
    with tab_exceptions:
        if data is not None:
            _exceptions_tab(data)
    with tab_complete:
        if data is not None:
            _complete_tab(data)
    with tab_readiness:
        _readiness_tab(session, api_base_url)
    with tab_sources:
        _sources_tab(session, api_base_url)
