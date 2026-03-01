"""
Streamlit dashboard for model monitoring and drift detection.

Launch:
    uv run streamlit run dashboard/app.py

Requires the API to be running at http://localhost:8000.
"""

import time

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Passos Mágicos — Model Monitoring",
    page_icon="📊",
    layout="wide",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

@st.cache_data(ttl=10)
def fetch_monitoring_stats() -> dict | None:
    """Fetch monitoring statistics from the API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/monitoring/stats", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
        return None


@st.cache_data(ttl=30)
def fetch_model_info() -> dict | None:
    """Fetch model metadata from the API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/model/info", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=10)
def fetch_health() -> dict | None:
    """Fetch API health status."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

st.sidebar.title("⚙️ Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
if auto_refresh:
    time.sleep(0)  # placeholder so we can use st.rerun below

st.sidebar.markdown("---")
st.sidebar.markdown("**API endpoint:**")
st.sidebar.code(API_BASE_URL)

health = fetch_health()
if health:
    status_color = "🟢" if health.get("status") == "healthy" else "🔴"
    model_color = "🟢" if health.get("model_loaded") else "🔴"
    st.sidebar.markdown(f"{status_color} API: {health.get('status', 'unknown')}")
    st.sidebar.markdown(f"{model_color} Model: {'loaded' if health.get('model_loaded') else 'not loaded'}")
else:
    st.sidebar.markdown("🔴 API: unreachable")


# ------------------------------------------------------------------
# Main content
# ------------------------------------------------------------------

st.title("📊 Passos Mágicos — Model Monitoring Dashboard")
st.markdown("Real-time monitoring of prediction drift and model performance.")
st.markdown("---")

# ---- Model Info ----
model_info = fetch_model_info()
if model_info:
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Version", model_info.get("version", "unknown"))
    metrics = model_info.get("metrics", {})
    col2.metric("Test F1 Macro", f"{metrics.get('test_f1_macro', 'N/A'):.4f}" if isinstance(metrics.get('test_f1_macro'), (int, float)) else "N/A")
    col3.metric("Test Accuracy", f"{metrics.get('test_accuracy', 'N/A'):.4f}" if isinstance(metrics.get('test_accuracy'), (int, float)) else "N/A")
    st.markdown("---")

# ---- Monitoring Stats ----
stats = fetch_monitoring_stats()

if stats is None:
    st.warning("Could not fetch monitoring data. Is the API running?")
    st.stop()

total = stats.get("total_predictions", 0)

# KPIs row
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Predictions", total)
kpi2.metric("Avg Confidence", f"{stats.get('avg_confidence', 0):.4f}")

drift_status = stats.get("drift_status", {})
severity = drift_status.get("severity", "none")
severity_emoji = {"none": "🟢", "warning": "🟡", "critical": "🔴"}.get(severity, "⚪")
kpi3.metric("Drift Status", f"{severity_emoji} {severity.upper()}")

st.markdown("---")

# ---- Drift Details ----
st.subheader("🔍 Drift Detection")

if total == 0:
    st.info("No predictions logged yet. Send some predictions via the /api/v1/predict endpoint to see monitoring data.")
else:
    drift_details = drift_status.get("details", {})

    if drift_details:
        drift_df = pd.DataFrame.from_dict(drift_details, orient="index")
        drift_df.index.name = "Class"

        col_chart, col_table = st.columns(2)

        with col_chart:
            st.markdown("**Prediction Distribution: Baseline vs. Current**")
            chart_data = pd.DataFrame({
                "Baseline": {k: v["baseline"] for k, v in drift_details.items()},
                "Current": {k: v["current"] for k, v in drift_details.items()},
            })
            st.bar_chart(chart_data)

        with col_table:
            st.markdown("**Drift Details by Class**")
            st.dataframe(drift_df, use_container_width=True)

        max_diff = drift_status.get("max_difference", 0)
        if drift_status.get("is_drifted"):
            st.warning(
                f"⚠️ Drift detected! Max distribution difference: **{max_diff:.4f}** "
                f"(threshold: 0.15). Severity: **{severity}**."
            )
        else:
            st.success(
                f"✅ No significant drift detected. Max difference: **{max_diff:.4f}**."
            )
    else:
        msg = drift_status.get("message", "No drift data available.")
        st.info(msg)

    # ---- Prediction Distribution ----
    st.subheader("📈 Current Prediction Distribution")
    dist = stats.get("prediction_distribution", {})
    if dist:
        dist_df = pd.DataFrame(
            {"Class": list(dist.keys()), "Proportion": list(dist.values())}
        ).set_index("Class")
        st.bar_chart(dist_df)
    else:
        st.info("No distribution data yet.")

    # ---- Recent Predictions ----
    st.subheader("🕐 Recent Predictions")
    recent = stats.get("recent_predictions", [])
    if recent:
        recent_df = pd.DataFrame(recent)
        if "timestamp" in recent_df.columns:
            recent_df = recent_df.sort_values("timestamp", ascending=False)
        st.dataframe(recent_df, use_container_width=True)
    else:
        st.info("No recent predictions.")

# ---- Auto-refresh ----
if auto_refresh:
    time.sleep(30)
    st.rerun()
