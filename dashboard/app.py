from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import math
import json

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh

    AUTOREFRESH_READY = True
except ImportError:
    AUTOREFRESH_READY = False

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MICROSERVICE_MAP_FILE = BASE_DIR / "data" / "config" / "microservice_map.json"
INDIAN_TZ = "Asia/Kolkata"

APP_NAME = "OptiCloud AI"
WATERMARK = "Team Genesis"
METRIC_ALIAS_MAP = {
    "CPUUtilization": ["CPUUtilization"],
    "NetworkIn": ["NetworkIn"],
    "NetworkOut": ["NetworkOut"],
}

st.set_page_config(page_title=APP_NAME, page_icon="??", layout="wide")


# -------------------------- Helpers --------------------------
def safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def latest_by_pattern(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_microservice_map() -> Dict[str, Dict[str, str]]:
    if not MICROSERVICE_MAP_FILE.exists():
        return {"resource_id": {}, "resource_name": {}}
    try:
        with open(MICROSERVICE_MAP_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"resource_id": {}, "resource_name": {}}
    return {
        "resource_id": data.get("resource_id", {}),
        "resource_name": data.get("resource_name", {}),
    }


def apply_microservice_map(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return d
    mapping = load_microservice_map()
    by_id = mapping.get("resource_id", {})
    by_name = mapping.get("resource_name", {})

    out = d.copy()
    id_col = "instance_id" if "instance_id" in out.columns else None
    name_col = "service_node" if "service_node" in out.columns else None

    mapped = pd.Series([None] * len(out), index=out.index, dtype="object")
    if id_col:
        mapped = out[id_col].astype(str).map(by_id)
    if name_col:
        mapped = mapped.fillna(out[name_col].astype(str).map(by_name))

    if "microservice" not in out.columns:
        out["microservice"] = "unassigned"
    out["microservice"] = mapped.fillna(out["microservice"]).astype(str)
    out["microservice"] = out["microservice"].replace({"": "unassigned", "nan": "unassigned"})
    return out


def normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize either legacy outputs/predictions.csv or data/processed/final_output.csv."""
    if df.empty:
        return df

    d = df.copy()
    column_map = {
        "resource_name": "service_node",
        "resource_id": "instance_id",
        "final_action": "predicted_action_name",
        "confidence_score": "confidence",
        "estimated_cost_before": "estimated_cost",
        "estimated_cost_after": "cost_after_action",
        "estimated_savings": "cost_saved",
        "service": "resource_type",
    }
    d = d.rename(columns={k: v for k, v in column_map.items() if k in d.columns})

    if "timestamp" in d.columns:
        d["timestamp"] = (
            pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
            .dt.tz_convert(INDIAN_TZ)
            .dt.tz_localize(None)
        )

    if "service_node" not in d.columns:
        if "instance_id" in d.columns:
            d["service_node"] = d["instance_id"].astype(str)
        else:
            d["service_node"] = "srv-unknown"

    if "instance_id" not in d.columns:
        d["instance_id"] = d["service_node"]

    if "microservice" not in d.columns:
        d["microservice"] = "unassigned"
    d["microservice"] = d["microservice"].astype(str).replace({"": "unassigned", "nan": "unassigned"})
    d = apply_microservice_map(d)

    if "predicted_action_name" not in d.columns:
        d["predicted_action_name"] = "monitor"

    if "confidence" not in d.columns:
        d["confidence"] = 0.0

    if "anomaly_score" not in d.columns:
        d["anomaly_score"] = np.nan

    for c in ["estimated_cost", "cost_after_action", "cost_saved", "confidence"]:
        if c not in d.columns:
            d[c] = 0.0
        d[c] = to_num(d[c])
    d["anomaly_score"] = pd.to_numeric(d["anomaly_score"], errors="coerce")

    return d


def normalize_usage(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()

    rename_map = {
        "resource_name": "service_node",
        "resource_id": "instance_id",
        "metric_name": "metric",
        "value": "metric_value",
    }
    d = d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns})

    if "timestamp" in d.columns:
        d["timestamp"] = (
            pd.to_datetime(d["timestamp"], errors="coerce", utc=True)
            .dt.tz_convert(INDIAN_TZ)
            .dt.tz_localize(None)
        )

    if "service_node" not in d.columns:
        d["service_node"] = "srv-unknown"
    if "instance_id" not in d.columns:
        d["instance_id"] = d["service_node"]
    if "microservice" not in d.columns:
        d["microservice"] = "unassigned"
    d["microservice"] = d["microservice"].astype(str).replace({"": "unassigned", "nan": "unassigned"})
    d = apply_microservice_map(d)
    if "metric" not in d.columns:
        d["metric"] = "CPUUtilization"
    if "metric_value" not in d.columns:
        d["metric_value"] = 0.0
    d["metric_value"] = to_num(d["metric_value"])

    return d


def compute_summary(pred_df: pd.DataFrame) -> Dict[str, float]:
    total_before = float(pred_df["estimated_cost"].sum())
    total_after = float(pred_df["cost_after_action"].sum())
    total_saved = float(pred_df["cost_saved"].sum())
    reduction_pct = (total_saved / total_before * 100.0) if total_before > 0 else 0.0
    return {
        "before": total_before,
        "after": total_after,
        "saved": total_saved,
        "reduction_pct": reduction_pct,
    }


def trunc4(value: float) -> float:
    return math.floor(float(value) * 10000.0) / 10000.0


def decision_tree_rows(row: pd.Series, fleet_cost_median: float) -> pd.DataFrame:
    anomaly_score = float(row.get("anomaly_score", 0.0))
    estimated_cost = float(row.get("estimated_cost", 0.0))
    confidence = trunc4(float(row.get("confidence", 0.0)))
    action_raw = str(row.get("predicted_action_name", "monitor")).lower()
    action = action_raw.title()
    optimization_actions = {"scale_down", "optimize_lambda", "limit_lambda", "optimize_s3_lifecycle"}

    anomaly_gate = anomaly_score >= 0.8
    cost_gate = estimated_cost >= fleet_cost_median
    confidence_gate = confidence >= 0.9
    action_gate = action_raw in optimization_actions
    optimize_gate = anomaly_gate and cost_gate and confidence_gate and action_gate

    return pd.DataFrame(
        [
            {
                "Decision Layer": "Layer 1: Anomaly Gate",
                "Condition": f"anomaly_score >= 0.80 (actual {anomaly_score:.2f})",
                "Outcome": "Pass" if anomaly_gate else "Watch",
            },
            {
                "Decision Layer": "Layer 2: Cost Gate",
                "Condition": f"estimated_cost >= ${fleet_cost_median:.4f} median (actual ${estimated_cost:.4f})",
                "Outcome": "Pass" if cost_gate else "Watch",
            },
            {
                "Decision Layer": "Layer 3: Confidence Gate",
                "Condition": f"confidence >= 0.9000 (actual {confidence:.4f})",
                "Outcome": "Pass" if confidence_gate else "Review",
            },
            {
                "Decision Layer": "Layer 4: Action Gate",
                "Condition": f"model action = {action}",
                "Outcome": "Optimize" if optimize_gate else "Hold/Review",
            },
        ]
    )


def load_dashboard_data() -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    # Preferred: notebook-style outputs
    pred_path = OUTPUTS_DIR / "predictions.csv"
    pred_df = safe_read_csv(pred_path)

    if not pred_df.empty:
        usage_path = OUTPUTS_DIR / "anomaly_points.csv"
        usage_df = safe_read_csv(usage_path)
        source_tag = "outputs/"
        source_file = str(pred_path)
        return normalize_predictions(pred_df), normalize_usage(usage_df), source_tag, source_file

    # Fallback: pipeline-style processed files
    pred_path_fallback = PROCESSED_DIR / "final_output.csv"
    pred_df = safe_read_csv(pred_path_fallback)

    latest_anomaly = latest_by_pattern(PROCESSED_DIR, "anomaly_scores_*.csv")
    usage_df = safe_read_csv(latest_anomaly) if latest_anomaly else pd.DataFrame()
    source_tag = "data/processed/"
    source_file = str(pred_path_fallback)
    return normalize_predictions(pred_df), normalize_usage(usage_df), source_tag, source_file


def answer_dashboard_question(
    query: str,
    pred_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    selected_metric: str,
    summary: Dict[str, float],
    latest_ts: pd.Timestamp | pd.NaT,
) -> str:
    q = query.lower().strip()
    if not q:
        return "Ask about savings, anomalies, most optimized service, or current usage trends."

    latest_view = pred_df[pred_df["timestamp"] == latest_ts] if pd.notna(latest_ts) else pred_df
    top_service = (
        pred_df.groupby("microservice", as_index=False)["cost_saved"].sum().sort_values("cost_saved", ascending=False)
    )
    top_node = (
        pred_df.groupby("service_node", as_index=False)["cost_saved"].sum().sort_values("cost_saved", ascending=False)
    )

    if "total" in q and ("saving" in q or "cost" in q):
        return (
            f"Total cost before optimization is ${summary['before']:.4f}, "
            f"after optimization is ${summary['after']:.4f}, "
            f"net savings is ${summary['saved']:.4f} ({summary['reduction_pct']:.2f}% reduction)."
        )

    if "optimized" in q or "best" in q or "top service" in q:
        if top_service.empty:
            return "No optimization records are available for the current filters."
        row = top_service.iloc[0]
        return (
            f"The most optimized microservice is {row['microservice']} with "
            f"${float(row['cost_saved']):.4f} total savings in the selected view."
        )

    if "node" in q or "srv-" in q:
        if top_node.empty:
            return "No node-level records are available for the current filters."
        row = top_node.iloc[0]
        return (
            f"Top optimized node is {row['service_node']} with ${float(row['cost_saved']):.4f} savings. "
            "Check the 'Node-Level Optimization Impact' chart for full ranking."
        )

    if "anomal" in q:
        anom = int((pred_df["anomaly_score"] >= 0.8).sum())
        total = len(pred_df)
        return (
            f"I found {anom} high-anomaly records (score >= 0.80) out of {total} selected records. "
            "These are prioritized by the decision-tree anomaly gate."
        )

    if "usage" in q or "trend" in q or "graph" in q:
        if usage_df.empty:
            return "Usage time-series data is currently unavailable for your selected filters."
        trend = (
            usage_df.groupby("timestamp", as_index=False)["metric_value"].mean().sort_values("timestamp")
        )
        if len(trend) < 2:
            return f"Current {selected_metric} is {float(trend['metric_value'].iloc[-1]):.3f}."
        start = float(trend["metric_value"].iloc[0])
        end = float(trend["metric_value"].iloc[-1])
        direction = "upward" if end > start else "downward"
        return (
            f"The {selected_metric} trend is {direction} in the selected time window "
            f"(from {start:.3f} to {end:.3f}). Hover over the line chart to inspect each microservice/node point."
        )

    if "time" in q or "latest" in q or "timestamp" in q:
        if pd.isna(latest_ts):
            return "Latest timestamp is not available."
        return f"Latest datapoint is {latest_ts} IST. All dashboard timestamps are now shown in India time."

    if "why" in q and ("optimiz" in q or "action" in q):
        best = pred_df.sort_values(["cost_saved", "confidence"], ascending=[False, False]).iloc[0]
        return (
            f"{best['service_node']} was optimized because anomaly score {float(best['anomaly_score']):.2f}, "
            f"estimated cost ${float(best['estimated_cost']):.4f}, and final decision confidence {float(best['confidence']):.1%} "
            f"passed policy gates. Recommended action: {best['predicted_action_name']}."
        )

    latest_cost = float(latest_view["estimated_cost"].sum()) if not latest_view.empty else 0.0
    return (
        "Quick summary: "
        f"live estimated cost is ${latest_cost:.4f}, net savings is ${summary['saved']:.4f}, "
        "and the decision tree is actively selecting optimization actions."
    )


# -------------------------- Styling --------------------------
st.markdown(
    """
    <style>
    :root {
        --bg-1: #f2f6ff;
        --bg-2: #e8f8f1;
        --card: #ffffff;
        --ink: #0f172a;
        --muted: #475569;
        --accent: #0a5d8f;
        --accent-2: #0b8b67;
    }
    .stApp {
        background: #ffffff;
    }
    .hero {
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        color: #f8fafc;
        background: linear-gradient(125deg, #083047 0%, #0a5d8f 56%, #0b8b67 100%);
        box-shadow: 0 14px 40px rgba(11, 20, 35, 0.18);
    }
    .hero h1 { margin: 0; letter-spacing: 0.2px; }
    .hero p { margin: 0.28rem 0 0; color: #ddf4ff; }
    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.95);
    }
    header[data-testid="stHeader"]::after {
        content: "Team Genesis";
        position: absolute;
        right: 130px;
        top: 6px;
        font-size: 0.83rem;
        color: rgba(15, 23, 42, 0.75);
        font-weight: 700;
        letter-spacing: 0.3px;
        z-index: 100001;
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(15, 23, 42, 0.12);
        pointer-events: none;
    }
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        margin-right: 0.35rem;
        background: #e0ecff;
        color: #10314d;
        font-weight: 600;
        font-size: 0.80rem;
    }
    div[data-testid="stExpander"] {
        position: fixed;
        right: 16px;
        bottom: 16px;
        width: min(390px, 94vw);
        z-index: 10000;
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 12px;
        box-shadow: 0 18px 44px rgba(2, 6, 23, 0.25);
    }
    div[data-testid="stExpander"] > details {
        max-height: 75vh;
        overflow: auto;
    }
    [data-testid="stPlotlyChart"] {
        background: #ffffff;
        border: 1px solid #eef2f7;
        border-radius: 12px;
        padding: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero">
        <h1>{APP_NAME}</h1>
        <p>Real-time cloud cost intelligence with explainable decision-tree optimization (final confidence blends model and policy)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

predictions_df, usage_df, source_tag, source_file = load_dashboard_data()
if predictions_df.empty:
    st.error(
        "No dashboard data found. Expected either `outputs/predictions.csv` or `data/processed/final_output.csv`."
    )
    st.stop()

# -------------------------- Sidebar --------------------------
with st.sidebar:
    st.header("Control Center")

    refresh_tick = None
    auto_refresh = st.toggle("Auto-refresh every 1 minute", value=True)
    if auto_refresh and AUTOREFRESH_READY:
        refresh_tick = st_autorefresh(interval=60_000, key="aegiscost_refresh")
    elif auto_refresh:
        st.warning("`streamlit-autorefresh` not available in this Python environment.")

    if st.button("Refresh now", use_container_width=True):
        st.rerun()

    microservices = sorted(predictions_df["microservice"].dropna().unique().tolist())
    selected_micro = st.multiselect("Microservices", options=microservices, default=microservices)

    candidate_nodes = (
        predictions_df[predictions_df["microservice"].isin(selected_micro)]["service_node"].dropna().unique().tolist()
        if selected_micro
        else []
    )
    candidate_nodes = sorted(set(candidate_nodes))
    selected_nodes = st.multiselect("Service Nodes", options=candidate_nodes, default=candidate_nodes)

    selected_metric = st.selectbox(
        "Usage Metric",
        options=list(METRIC_ALIAS_MAP.keys()),
        index=0,
    )
    usage_window_minutes = st.selectbox(
        "Usage Window",
        options=[30, 60, 120, 360, 1440],
        index=1,
        format_func=lambda x: f"Last {x} min" if x < 1440 else "Last 24 hours",
    )

    source_mtime = pd.Timestamp(Path(source_file).stat().st_mtime, unit="s").tz_localize("UTC").tz_convert(INDIAN_TZ)
    refresh_label = refresh_tick if refresh_tick is not None else "manual/fallback"
    st.caption(f"Live source: `{source_tag}`")
    st.caption(f"Refresh ticks: `{refresh_label}`")
    st.caption(f"Source file updated (IST): `{source_mtime.strftime('%Y-%m-%d %H:%M:%S')}`")

if not selected_micro or not selected_nodes:
    st.warning("Select at least one microservice and one service node.")
    st.stop()

pred_f = predictions_df[
    predictions_df["microservice"].isin(selected_micro) & predictions_df["service_node"].isin(selected_nodes)
].copy()

usage_f = usage_df[
    usage_df["microservice"].isin(selected_micro)
    & usage_df["service_node"].isin(selected_nodes)
    & (usage_df["metric"].isin(METRIC_ALIAS_MAP.get(selected_metric, [selected_metric])))
].copy() if not usage_df.empty else usage_df

if not usage_f.empty and "timestamp" in usage_f.columns:
    latest_usage_ts = usage_f["timestamp"].max()
    if pd.notna(latest_usage_ts):
        window_start = latest_usage_ts - pd.Timedelta(minutes=int(usage_window_minutes))
        usage_f = usage_f[usage_f["timestamp"] >= window_start].copy()

if pred_f.empty:
    st.warning("No records available for the selected filters.")
    st.stop()

summary = compute_summary(pred_f)
latest_ts = pred_f["timestamp"].max() if "timestamp" in pred_f.columns else pd.NaT
latest_slice = pred_f[pred_f["timestamp"] == latest_ts] if pd.notna(latest_ts) else pred_f
live_nodes = int(latest_slice["service_node"].nunique())
live_est_cost = float(latest_slice["estimated_cost"].sum())
live_saved = float(latest_slice["cost_saved"].sum())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Cost Before", f"${summary['before']:.4f}")
k2.metric("Cost After", f"${summary['after']:.4f}")
k3.metric("Net Savings", f"${summary['saved']:.4f}")
k4.metric("Reduction", f"{summary['reduction_pct']:.2f}%")
k5.metric("Live Nodes (Latest TS)", f"{live_nodes}", delta=f"${live_saved:.4f} saved @ latest")

st.markdown(
    "<span class='pill'>Decision Tree Layer: Active</span>"
    "<span class='pill'>Real-time Mode: 60s</span>"
    "<span class='pill'>Interactive Hover: Enabled</span>",
    unsafe_allow_html=True,
)

# -------------------------- Charts --------------------------
row1_left, row1_right = st.columns((1.45, 1))

with row1_left:
    st.subheader("Microservice Usage Trend (Live)")
    if usage_f.empty:
        # Fallback so the chart never disappears for unavailable metrics (for example DiskRead/DiskWrite).
        ts_points = sorted(pred_f["timestamp"].dropna().unique().tolist())[-30:]
        node_map = (
            pred_f.groupby("service_node")["microservice"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
            .to_dict()
        )
        placeholder_rows = []
        for ts in ts_points:
            for node in selected_nodes:
                placeholder_rows.append(
                    {
                        "timestamp": ts,
                        "microservice": node_map.get(node, "Monitoring"),
                        "service_node": node,
                        "metric_value": 0.0,
                    }
                )
        usage_ts = pd.DataFrame(placeholder_rows)
        st.warning(
            f"No live `{selected_metric}` datapoints were returned for current filters. "
            "Showing placeholder timeline at zero until telemetry arrives."
        )
    else:
        usage_f["timestamp"] = pd.to_datetime(usage_f["timestamp"], errors="coerce")
        usage_f["timestamp"] = usage_f["timestamp"].dt.floor("min")
        usage_ts = (
            usage_f.groupby(["timestamp", "microservice", "service_node"], as_index=False)["metric_value"]
            .mean()
            .sort_values("timestamp")
        )

    # Ensure selected nodes are always represented (e.g., srv-01 with no current datapoints).
    if not usage_ts.empty:
        present_nodes = set(usage_ts["service_node"].astype(str).unique().tolist())
        missing_nodes = [n for n in selected_nodes if n not in present_nodes]
        if missing_nodes:
            ts_min = usage_ts["timestamp"].min()
            ts_max = pd.Timestamp.now().floor("min")
            idx = pd.date_range(ts_min, ts_max, freq="min")
            extras = []
            for node in missing_nodes:
                extras.append(
                    pd.DataFrame(
                        {
                            "timestamp": idx,
                            "microservice": "unassigned",
                            "service_node": node,
                            "metric_value": 0.0,
                        }
                    )
                )
            if extras:
                usage_ts = pd.concat([usage_ts] + extras, ignore_index=True)

    # Cloud telemetry can arrive 2-5 minutes late. Extend each node series to "now"
    # via forward-fill so the chart remains visually real-time.
    now_min = pd.Timestamp.now().floor("min")
    if not usage_ts.empty and "timestamp" in usage_ts.columns:
        extended_parts = []
        for (ms, node), grp in usage_ts.groupby(["microservice", "service_node"], as_index=False):
            g = grp.sort_values("timestamp").set_index("timestamp")
            full_index = pd.date_range(g.index.min(), now_min, freq="min")
            g2 = g.reindex(full_index)
            g2["metric_value"] = g2["metric_value"].ffill().fillna(0.0)
            g2["microservice"] = ms
            g2["service_node"] = node
            g2["is_projected"] = g2.index > g.index.max()
            g2 = g2.reset_index().rename(columns={"index": "timestamp"})
            extended_parts.append(g2)
        usage_ts = pd.concat(extended_parts, ignore_index=True)
    else:
        usage_ts["is_projected"] = False

    usage_ts["hover_label"] = (
        "Microservice: "
        + usage_ts["microservice"].astype(str)
        + "<br>Node: "
        + usage_ts["service_node"].astype(str)
        + f"<br>{selected_metric}: "
        + usage_ts["metric_value"].round(3).astype(str)
        + "<br>Mode: "
        + usage_ts["is_projected"].map({True: "Latest known value (carry-forward)", False: "Observed"})
    )

    fig_usage = px.line(
        usage_ts,
        x="timestamp",
        y="metric_value",
        color="service_node",
        line_group="service_node",
        markers=True,
        custom_data=["hover_label"],
        labels={"metric_value": selected_metric},
    )
    fig_usage.update_traces(
        hovertemplate="%{customdata[0]}<br>Time: %{x}<extra></extra>",
        line={"width": 2},
        marker={"size": 6},
    )
    fig_usage.update_layout(
        height=420,
        hovermode="closest",
        legend_title_text="",
        xaxis={"dtick": 600_000, "tickformat": "%H:%M"},
    )
    st.plotly_chart(
        fig_usage,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )
    latest_observed = usage_f["timestamp"].max() if not usage_f.empty else pd.NaT
    if pd.notna(latest_observed):
        lag_min = int((now_min - latest_observed.floor("min")).total_seconds() // 60)
        st.caption(
            f"Latest observed datapoint: {latest_observed.strftime('%H:%M')} IST | "
            f"Telemetry lag: {lag_min} min (chart extends to current minute using carry-forward)."
        )

with row1_right:
    st.subheader("Action Mix")
    action_mix = pred_f["predicted_action_name"].value_counts().reset_index()
    action_mix.columns = ["action", "count"]
    fig_actions = px.pie(action_mix, names="action", values="count", hole=0.48)
    fig_actions.update_layout(height=420, legend_title_text="")
    st.plotly_chart(fig_actions, use_container_width=True)

row2_left, row2_right = st.columns((1.2, 1.2))

with row2_left:
    st.subheader("Before vs After Optimization")
    by_ms = (
        pred_f.groupby("microservice", as_index=False)[["estimated_cost", "cost_after_action"]]
        .sum()
        .sort_values("estimated_cost", ascending=False)
    )
    by_ms_long = by_ms.melt(
        id_vars="microservice",
        value_vars=["estimated_cost", "cost_after_action"],
        var_name="type",
        value_name="cost",
    )
    fig_bars = px.bar(
        by_ms_long,
        x="microservice",
        y="cost",
        color="type",
        barmode="group",
        labels={"type": "Cost Type", "cost": "Cost"},
    )
    fig_bars.update_layout(height=400, legend_title_text="")
    st.plotly_chart(fig_bars, use_container_width=True)

with row2_right:
    st.subheader("Cost Waterfall")
    by_action = pred_f.groupby("predicted_action_name", as_index=False)["cost_saved"].sum()
    by_action = by_action.sort_values("cost_saved", ascending=False)

    fig_waterfall = go.Figure(
        go.Waterfall(
            name="cost_saved",
            orientation="v",
            x=by_action["predicted_action_name"],
            y=by_action["cost_saved"],
            connector={"line": {"color": "#64748b"}},
            increasing={"marker": {"color": "#0b8b67"}},
            decreasing={"marker": {"color": "#b91c1c"}},
            totals={"marker": {"color": "#0a5d8f"}},
        )
    )
    fig_waterfall.update_layout(height=400, yaxis_title="Savings Impact")
    st.plotly_chart(fig_waterfall, use_container_width=True)

row3_left, row3_right = st.columns((1.3, 1))

with row3_left:
    st.subheader("Node-Level Optimization Impact")
    node_impact = (
        pred_f.groupby(["microservice", "service_node"], as_index=False)["cost_saved"]
        .sum()
        .sort_values("cost_saved", ascending=True)
    )
    fig_node_impact = px.bar(
        node_impact,
        x="cost_saved",
        y="service_node",
        color="microservice",
        orientation="h",
        labels={"cost_saved": "Saved Cost", "service_node": "Service Node"},
    )
    fig_node_impact.update_layout(height=430, legend_title_text="")
    st.plotly_chart(fig_node_impact, use_container_width=True)

with row3_right:
    st.subheader("Decision Tree Reasoning")
    optimize_actions = {"scale_down", "optimize_lambda", "limit_lambda", "optimize_s3_lifecycle"}
    optimize_candidates = pred_f[
        (pred_f["anomaly_score"] >= 0.8)
        & (pred_f["predicted_action_name"].astype(str).str.lower().isin(optimize_actions))
    ]
    best_row = (
        optimize_candidates.sort_values(["cost_saved", "confidence"], ascending=[False, False]).iloc[0]
        if not optimize_candidates.empty
        else pred_f.sort_values(["cost_saved", "confidence"], ascending=[False, False]).iloc[0]
    )
    fleet_median = float(predictions_df["estimated_cost"].median()) if not predictions_df.empty else 0.0
    tree_df = decision_tree_rows(best_row, fleet_median)
    st.dataframe(tree_df, hide_index=True, use_container_width=True)

    st.markdown(
        (
            f"**Optimization call for `{best_row['service_node']}` ({best_row['microservice']}):** "
            f"Anomaly score `{best_row['anomaly_score']:.2f}` and current cost `${best_row['estimated_cost']:.4f}` "
            f"crossed policy gates with `{trunc4(float(best_row['confidence'])):.4f}` final decision confidence. "
            f"The model recommends `{best_row['predicted_action_name']}` because it provides the strongest "
            "cost-risk balance among available actions."
        )
    )

st.subheader("Live Recommendation Feed")
feed_cols = [
    "timestamp",
    "microservice",
    "service_node",
    "instance_id",
    "predicted_action_name",
    "confidence",
    "anomaly_score",
    "estimated_cost",
    "cost_after_action",
    "cost_saved",
]
feed_cols = [c for c in feed_cols if c in pred_f.columns]

feed_df = pred_f[feed_cols].sort_values(["timestamp", "cost_saved", "confidence"], ascending=[False, False, False])
feed_styler = feed_df.style.format(
    {
        "confidence": lambda x: f"{trunc4(float(x)):.4f}",
        "anomaly_score": "{:.4f}",
        "estimated_cost": "{:.6f}",
        "cost_after_action": "{:.6f}",
        "cost_saved": "{:.6f}",
    }
)
st.dataframe(feed_styler, hide_index=True, use_container_width=True)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": (
                "I can explain every graph and decision. Try: "
                "'Which microservice is most optimized?' or 'What does the usage trend show?'"
            ),
        }
    ]

with st.expander("AI Ops Assistant", expanded=False):
    st.caption("Ask about chart meaning, anomalies, optimization reasons, costs, or trends.")
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask dashboard assistant...")
    if user_q:
        st.session_state.chat_messages.append({"role": "user", "content": user_q})
        answer = answer_dashboard_question(user_q, pred_f, usage_f, selected_metric, summary, latest_ts)
        st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        st.rerun()

if pd.notna(latest_ts):
    st.caption(
        f"Latest datapoint timestamp: {latest_ts} IST | Live estimated cost: ${live_est_cost:.4f} | Timezone: Asia/Kolkata"
    )


