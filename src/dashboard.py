"Displays saved neuro-symbolic misuse decisions and governance information in a read-only Streamlit dashboard."

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from streamlit import runtime as streamlit_runtime

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_DATA_PATH = OUTPUTS_DIR / "fusion_dashboard_data.json"

if __name__ == "__main__" and not streamlit_runtime.exists():
    from streamlit.web import cli as streamlit_cli

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        "--server.headless=false",
        "--server.fileWatcherType=none",
        "--theme.base=light",
        "--theme.primaryColor=#315f78",
        "--theme.backgroundColor=#f4f6f8",
        "--theme.secondaryBackgroundColor=#e9eef2",
        "--theme.textColor=#1e2933",
    ]
    raise SystemExit(streamlit_cli.main())

st.set_page_config(
    page_title="AI Misuse Governance Dashboard", page_icon=None, layout="wide"
)
st.markdown(
    """
<style>
.stApp,[data-testid="stAppViewContainer"]{background:#f4f6f8;color:#1e2933}
.block-container{max-width:1380px;padding-top:1.4rem;padding-bottom:2rem}
h1,h2,h3{color:#172b3a;letter-spacing:0}h1{font-size:1.8rem!important;font-weight:650!important}
h2{font-size:1.25rem!important;font-weight:620!important}h3{font-size:1.05rem!important}
[data-testid="stMetric"]{background:#fff;border:1px solid #d8dee4;border-radius:3px;padding:.75rem .9rem}
[data-testid="stMetricLabel"]{color:#53616c}[data-testid="stMetricValue"]{color:#17354a;font-size:1.55rem}
[data-testid="stSidebar"]{background:#e9eef2;border-right:1px solid #d0d8de}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span{color:#263640!important}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] [data-baseweb="select"]>div{background:#fff!important;color:#1e2933!important}
[data-testid="stSidebar"] [data-baseweb="tag"]{background:#d9e4ea!important;border-radius:2px!important}
[data-testid="stTabs"] button p{color:#40515c!important}[data-testid="stTabs"] button[aria-selected="true"] p{color:#244e64!important}
div[data-testid="stDataFrame"]{border:1px solid #d8dee4}.status-line{background:#fff;border-left:4px solid #315f78;padding:.65rem .85rem;margin:.25rem 0 1rem}.small-note{color:#5c6872;font-size:.88rem}
</style>""",
    unsafe_allow_html=True,
)

CATEGORY_LABELS = {
    "benign": "No detected misuse",
    "prompt_injection": "Prompt injection",
    "sensitive_info_disclosure": "Sensitive information disclosure",
    "excessive_agency": "Excessive agency",
    "unbounded_consumption": "Unbounded consumption",
    "misinformation": "Misinformation",
    "hidden_context_exposure": "Hidden context exposure",
    "vector_embedding_weakness": "Vector and embedding weakness",
    "improper_output_handling": "Improper output handling",
}
OUTCOME_LABELS = {
    "normal_activity": "Normal approved AI use",
    "shadow_ai_usage": "Unapproved AI use (no misuse)",
    "agentic_ai_misuse": "Agentic misuse on approved AI",
    "shadow_ai_and_agentic_ai_misuse": "Agentic misuse on unapproved AI",
}
CASE_LABELS = {
    "C-agreement": "Neural and direct-rule agreement",
    "A-neural-only": "Neural decision without a direct rule",
    "benign-confident": "Clear no-misuse decision",
    "B-symbolic-rescue": "Direct rule corrected the neural decision",
    "A-neural-and-supporting-agree": "Neural and supporting-rule agreement",
    "E-conflict-symbolic-favoured": "Reliable direct rule resolved a conflict",
    "F-agreement-and-strength": "Multiple direct findings resolved",
    "A-neural-moderate": "Moderate neural decision",
    "A-neural-weak": "Weak neural decision",
    "benign-weak": "Uncertain no-misuse decision",
    "benign-with-supporting": "No-misuse decision with a supporting finding",
    "benign-moderate": "Moderate no-misuse decision",
    "A-supporting-contradicts": "Neural and supporting evidence conflict",
    "E-conflict-neural-favoured": "Neural decision with an unresolved direct conflict",
    "A-supporting-corroborates": "Supporting rule corroborated the neural decision",
    "F-strength-arbitrates": "Strongest direct evidence selected",
    "F-unresolved": "Unresolved multiple-rule conflict",
    "G-compliance-override": "Complete symbolic compliance overrode the neural alert",
}


@st.cache_data(show_spinner=False)
def load_results(path: str, modified_time: int) -> dict:
    del modified_time
    with Path(path).open("r", encoding="utf-8") as handle:
        results = json.load(handle)
    missing = {"schema_version", "metrics", "fusion_metrics", "events"}.difference(
        results
    )
    if missing:
        raise ValueError(
            f"Saved dashboard data is missing: {', '.join(sorted(missing))}"
        )
    if results["schema_version"] != 2 or not results["events"]:
        raise ValueError("The saved fusion run is empty or uses an unsupported format.")
    return results


def build_event_table(results: dict) -> pd.DataFrame:
    rows = []
    for event in results["events"]:
        context, neural = event["context"], event["neural"]
        rows.append(
            {
                "event_id": event["event_id"],
                "timestamp": context["timestamp"],
                "department": context["department"],
                "platform": context["ai_platform"],
                "platform_status": context["platform_status"],
                "shadow_ai": context["shadow_ai"],
                "category": event["final_category"],
                "category_label": CATEGORY_LABELS.get(
                    event["final_category"], event["final_category"]
                ),
                "outcome": event["governance_outcome"],
                "outcome_label": OUTCOME_LABELS.get(
                    event["governance_outcome"], event["governance_outcome"]
                ),
                "confidence": event["confidence"],
                "human_review": event["human_review"],
                "decision_case": event["decision_case"],
                "neural_category": neural["top_category"],
                "neural_p1": neural["p1"],
                "neural_margin": neural["margin"],
                "is_misuse": event["is_misuse"],
            }
        )
    return pd.DataFrame(rows)


def choices(label: str, values: list[str], key: str) -> list[str]:
    return st.sidebar.multiselect(label, values, placeholder="All", key=key)


def apply_filters(table: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    search = st.sidebar.text_input("Event ID contains", key="event_search")
    selections = [
        (
            "outcome_label",
            choices(
                "Governance outcome",
                sorted(table.outcome_label.unique()),
                "outcome_filter",
            ),
        ),
        (
            "category_label",
            choices(
                "Misuse category",
                sorted(table.category_label.unique()),
                "category_filter",
            ),
        ),
        (
            "confidence",
            choices("Confidence", ["high", "medium", "low"], "confidence_filter"),
        ),
        (
            "platform_status",
            choices(
                "Platform status",
                sorted(table.platform_status.unique()),
                "platform_filter",
            ),
        ),
        (
            "department",
            choices(
                "Department", sorted(table.department.unique()), "department_filter"
            ),
        ),
    ]
    if "review_filter" not in st.session_state:
        st.session_state.review_filter = "All events"
    review = st.sidebar.selectbox(
        "Review status",
        ["All events", "Review required", "No review required"],
        key="review_filter",
    )
    filtered = table.copy()
    for column, selected in selections:
        if selected:
            filtered = filtered[filtered[column].isin(selected)]
    if search:
        filtered = filtered[
            filtered.event_id.str.contains(search, case=False, na=False)
        ]
    if review == "Review required":
        filtered = filtered[filtered.human_review]
    elif review == "No review required":
        filtered = filtered[~filtered.human_review]
    return filtered


def open_review_queue() -> None:
    st.session_state.event_search = ""
    for key in (
        "outcome_filter",
        "category_filter",
        "confidence_filter",
        "platform_filter",
        "department_filter",
    ):
        st.session_state[key] = []
    st.session_state.review_filter = "Review required"


def show_summary(filtered: pd.DataFrame, total: int) -> None:
    visible = len(filtered)
    values = [
        visible,
        int(filtered.is_misuse.sum()),
        int(filtered.shadow_ai.sum()),
        int(filtered.human_review.sum()),
    ]
    labels = ["Events shown", "Misuse alerts", "Shadow AI events"]
    columns = st.columns(4)
    for index, (column, label, value) in enumerate(
        zip(columns[:3], labels, values[:3])
    ):
        column.metric(label, f"{value:,}")
        column.caption(
            f"{visible:,} of {total:,} evaluated events"
            if index == 0
            else (
                f"{value/visible:.1%} of events shown" if visible else "No events shown"
            )
        )
    with columns[3]:
        st.metric("Review queue", f"{values[3]:,}")
        st.button(
            "Open review queue",
            width="stretch",
            on_click=open_review_queue,
            help="Show all events currently requiring review",
        )


def show_monitoring(filtered: pd.DataFrame, results: dict) -> None:
    if filtered.empty:
        st.info("No events match the selected filters.")
        return
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Governance outcomes")
        data = (
            filtered.outcome_label.value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Events")
        )
        chart = (
            alt.Chart(data)
            .mark_bar(color="#315f78", size=24)
            .encode(
                x=alt.X("Events:Q", title="Event count"),
                y=alt.Y(
                    "Outcome:N", title=None, sort="-x", axis=alt.Axis(labelLimit=300)
                ),
                tooltip=["Outcome:N", "Events:Q"],
            )
            .properties(height=190)
        )
        st.altair_chart(chart, width="stretch", theme=None)
        st.caption(
            "Approved AI is organisation-sanctioned. Unapproved AI represents Shadow AI use. "
            "Agentic misuse identifies unsafe or policy-violating AI behaviour."
        )
    with right:
        st.subheader("Confidence")
        data = (
            filtered.confidence.str.title()
            .value_counts()
            .reindex(["High", "Medium", "Low"], fill_value=0)
            .rename_axis("Confidence")
            .reset_index(name="Events")
        )
        chart = (
            alt.Chart(data)
            .mark_bar(color="#748b98", size=42)
            .encode(
                x=alt.X("Confidence:N", title=None, sort=["High", "Medium", "Low"]),
                y=alt.Y("Events:Q", title="Event count"),
                tooltip=["Confidence:N", "Events:Q"],
            )
            .properties(height=190)
        )
        st.altair_chart(chart, width="stretch", theme=None)
    st.subheader("Events")
    selected_table = filtered.reset_index(drop=True)
    display = selected_table[
        [
            "timestamp",
            "event_id",
            "department",
            "platform_status",
            "outcome_label",
            "category_label",
            "confidence",
            "human_review",
        ]
    ].copy()
    display["timestamp"] = pd.to_datetime(display.timestamp).dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    display["event_id"] = display.event_id.str.slice(0, 12) + "…"
    display.columns = [
        "Time",
        "Event ID",
        "Department",
        "Platform status",
        "Governance outcome",
        "Misuse category",
        "Confidence",
        "Human review",
    ]
    selection = st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        height=390,
        on_select="rerun",
        selection_mode="single-row",
        key="governance_event_table",
    )
    selected_rows = selection.selection.rows
    if not selected_rows:
        st.caption("Click an event row to view its details.")
        return

    st.subheader("Event details")
    event_id = selected_table.iloc[selected_rows[0]]["event_id"]
    event = next(item for item in results["events"] if item["event_id"] == event_id)
    context, neural, symbolic = event["context"], event["neural"], event["symbolic"]
    st.markdown(
        f'<div class="status-line"><strong>{OUTCOME_LABELS.get(event["governance_outcome"],event["governance_outcome"])}</strong><br>'
        f'{CATEGORY_LABELS.get(event["final_category"],event["final_category"])} · {event["confidence"].title()} confidence · '
        f'{"Human review required" if event["human_review"] else "No human review required"}</div>',
        unsafe_allow_html=True,
    )
    detail_left, detail_right = st.columns([1.25, 1])
    with detail_left:
        if event["is_misuse"]:
            st.markdown("**Reason for flagging**")
            st.write(
                event.get("reason_for_flagging")
                or "No reason was recorded for this alert."
            )
        st.markdown("**Fusion explanation**")
        st.write(event["explanation"])
        st.markdown("**Prompt**")
        st.write(event["prompt"] or "Not recorded")
        st.markdown("**Model output**")
        st.write(event["model_output"] or "Not recorded")
    with detail_right:
        st.markdown("**Context**")
        st.write(f'Department: {context["department"].replace("_"," ").title()}')
        st.write(f'Task type: {context["task_type"].replace("_"," ").title()}')
        st.write(f'Platform: {context["ai_platform"].replace("_"," ").title()}')
        st.write(f'Platform status: {context["platform_status"].title()}')
        st.write(
            f'Decision basis: {CASE_LABELS.get(event["decision_case"],event["decision_case"])}'
        )
        st.write(
            f'Neural result: {CATEGORY_LABELS.get(neural["top_category"],neural["top_category"])} ({neural["p1"]:.2f}, margin {neural["margin"]:.2f})'
        )
        st.markdown("**Symbolic findings**")
        findings = [("Direct", item) for item in symbolic["direct_findings"]] + [
            ("Supporting", item) for item in symbolic["supporting_findings"]
        ]
        if not findings:
            st.write("No symbolic rule fired.")
        for role, finding in findings:
            st.write(
                f'{role} — {finding["rule"].replace("_"," ").title()}: {finding["explanation"]}'
            )
        compliance = symbolic.get("compliance_assessment", {})
        if compliance.get("applicable"):
            st.markdown("**Compliance assessment**")
            st.write(
                compliance.get("explanation", "No compliance explanation recorded.")
            )


st.title("Shadow AI and Agentic AI Misuse Governance")
st.markdown(
    '<p class="small-note">Fused neural and symbolic monitoring results for the simulated enterprise dataset.</p>',
    unsafe_allow_html=True,
)
if not DASHBOARD_DATA_PATH.exists():
    st.error(
        "Dashboard data has not been generated yet. Run neurosymbolic_fusion.py once, then run dashboard.py again."
    )
    st.stop()
try:
    analysis = load_results(
        str(DASHBOARD_DATA_PATH), DASHBOARD_DATA_PATH.stat().st_mtime_ns
    )
except (OSError, ValueError, json.JSONDecodeError) as error:
    st.error(f"The saved fusion results could not be read: {error}")
    st.stop()
events_table = build_event_table(analysis)
filtered_events = apply_filters(events_table)
show_summary(filtered_events, len(events_table))
show_monitoring(filtered_events, analysis)
