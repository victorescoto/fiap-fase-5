"""
Streamlit dashboard for model monitoring and drift detection.

Launch:
    uv run streamlit run dashboard/app.py

Requires the API to be running at http://localhost:8000 for live
monitoring data.  Model metadata is loaded from disk as fallback.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000"
METADATA_PATH = (
    Path(__file__).resolve().parent.parent / "app" / "model" / "model_metadata.json"
)

st.set_page_config(
    page_title="Passos Mágicos — Model Monitoring",
    page_icon="📊",
    layout="wide",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------

st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 0.75rem;
        color: white;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .metric-card p  { margin: 0; font-size: 1.8rem; font-weight: bold; }
    .drift-ok   { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .drift-warn { background: linear-gradient(135deg, #f7971e, #ffd200); }
    .drift-crit { background: linear-gradient(135deg, #e53935, #e35d5b); }
    .info-card  { background: linear-gradient(135deg, #2193b0, #6dd5ed); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Helpers — API calls
# ------------------------------------------------------------------


@st.cache_data(ttl=10)
def fetch_monitoring_stats() -> dict | None:
    """Fetch monitoring statistics from the API."""
    try:
        resp = requests.get(f"{API_BASE_URL}/api/v1/monitoring/stats", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
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


def load_local_metadata() -> dict | None:
    """Load model metadata from the local JSON file (fallback)."""
    try:
        with open(METADATA_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def _metric_card(label: str, value: str, extra_class: str = "") -> str:
    """Return HTML for a styled metric card."""
    return (
        f'<div class="metric-card {extra_class}">'
        f"<h3>{label}</h3><p>{value}</p></div>"
    )


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

st.sidebar.title("⚙️ Configurações")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
st.sidebar.markdown("---")

st.sidebar.markdown("**API endpoint:**")
st.sidebar.code(API_BASE_URL, language=None)

health = fetch_health()
api_online = health is not None

if api_online:
    status_color = "🟢" if health.get("status") == "healthy" else "🔴"
    model_color = "🟢" if health.get("model_loaded") else "🔴"
    st.sidebar.markdown(f"{status_color} API: **{health.get('status', 'unknown')}**")
    st.sidebar.markdown(
        f"{model_color} Modelo: "
        f"**{'carregado' if health.get('model_loaded') else 'não carregado'}**"
    )
else:
    st.sidebar.markdown("🔴 API: **offline**")
    st.sidebar.info("O dashboard usará metadados locais do modelo.")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Desenvolvido para o **Tech Challenge — Fase 5**  \n"
    "Pós Tech · FIAP · Passos Mágicos"
)

# ------------------------------------------------------------------
# Title
# ------------------------------------------------------------------

st.title("📊 Passos Mágicos — Model Monitoring Dashboard")
st.caption(
    "Painel de monitoramento contínuo do modelo preditivo "
    "de risco de defasagem escolar"
)
st.markdown("---")

# ------------------------------------------------------------------
# Load data — prefer API, fallback to local metadata
# ------------------------------------------------------------------

model_info = fetch_model_info() if api_online else None
local_meta = load_local_metadata()
meta = model_info or local_meta or {}

metrics: dict = meta.get("metrics", {})
stats = fetch_monitoring_stats() if api_online else None

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------

tab_overview, tab_drift, tab_predictions, tab_model = st.tabs(
    [
        "📈 Visão Geral",
        "🔍 Drift Detection",
        "🕐 Predições",
        "🔧 Detalhes do Modelo",
    ]
)

# ==================================================================
# TAB 1 — Visão Geral
# ==================================================================

with tab_overview:
    st.subheader("Métricas de Performance do Modelo")

    # Row 1: key metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        val = metrics.get("test_accuracy")
        st.markdown(
            _metric_card("Accuracy (teste)", f"{val:.4f}" if val else "N/A"),
            unsafe_allow_html=True,
        )
    with c2:
        val = metrics.get("test_f1_macro")
        st.markdown(
            _metric_card("F1 Macro (teste)", f"{val:.4f}" if val else "N/A"),
            unsafe_allow_html=True,
        )
    with c3:
        val = metrics.get("test_recall_macro")
        st.markdown(
            _metric_card("Recall Macro (teste)", f"{val:.4f}" if val else "N/A"),
            unsafe_allow_html=True,
        )
    with c4:
        val = metrics.get("test_precision_macro")
        st.markdown(
            _metric_card(
                "Precision Macro (teste)", f"{val:.4f}" if val else "N/A"
            ),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Row 2: additional metrics
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        val = metrics.get("test_f1_weighted")
        st.markdown(
            _metric_card(
                "F1 Weighted (teste)",
                f"{val:.4f}" if val else "N/A",
                "info-card",
            ),
            unsafe_allow_html=True,
        )
    with c6:
        val = metrics.get("cv_f1_mean")
        st.markdown(
            _metric_card(
                "CV F1 Mean", f"{val:.4f}" if val else "N/A", "info-card"
            ),
            unsafe_allow_html=True,
        )
    with c7:
        val = metrics.get("cv_f1_std")
        st.markdown(
            _metric_card(
                "CV F1 Std", f"{val:.4f}" if val else "N/A", "info-card"
            ),
            unsafe_allow_html=True,
        )
    with c8:
        version = meta.get("version", "unknown")
        st.markdown(
            _metric_card("Versão do Modelo", version, "info-card"),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Bar chart of metrics
    st.subheader("Comparativo de Métricas")
    metric_names = [
        "test_accuracy",
        "test_f1_macro",
        "test_f1_weighted",
        "test_recall_macro",
        "test_precision_macro",
        "cv_f1_mean",
    ]
    metric_labels = [
        "Accuracy",
        "F1 Macro",
        "F1 Weighted",
        "Recall Macro",
        "Precision Macro",
        "CV F1 Mean",
    ]
    metric_values = [metrics.get(m, 0) for m in metric_names]

    if any(v > 0 for v in metric_values):
        chart_df = pd.DataFrame(
            {"Métrica": metric_labels, "Valor": metric_values}
        ).set_index("Métrica")
        st.bar_chart(chart_df, horizontal=True)
    else:
        st.info("Nenhuma métrica disponível.")

    # Monitoring KPIs
    st.markdown("---")
    st.subheader("Status de Monitoramento em Tempo Real")

    # Resolve values: prefer live stats, fallback to baseline metadata
    _baseline_meta = (local_meta or {}).get("baseline_stats", {})

    if stats and stats.get("total_predictions", 0) > 0:
        total = stats["total_predictions"]
        avg_conf = stats.get("avg_confidence", 0)
        drift_status = stats.get("drift_status", {})
        severity = drift_status.get("severity", "none")
        _source_label = "dados em tempo real"
    else:
        total = _baseline_meta.get("total_samples", 0)
        avg_conf = _baseline_meta.get("avg_confidence", 0)
        severity = "none"
        _source_label = "baseline (treinamento)"

    severity_map = {
        "none": ("🟢 Sem Drift", "drift-ok"),
        "warning": ("🟡 Warning", "drift-warn"),
        "critical": ("🔴 Crítico", "drift-crit"),
    }
    sev_label, sev_class = severity_map.get(severity, ("⚪ N/A", ""))

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(
            _metric_card(
                "Total de Amostras", str(total), "info-card"
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            _metric_card(
                "Confiança Média", f"{avg_conf:.4f}", "info-card"
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            _metric_card("Status de Drift", sev_label, sev_class),
            unsafe_allow_html=True,
        )

    st.caption(f"📌 Fonte dos dados: **{_source_label}**")

    if not (stats and stats.get("total_predictions", 0) > 0):
        st.info(
            "Nenhuma predição registrada via API ainda. "
            "Os valores acima refletem o baseline de treinamento. "
            "Envie predições via `POST /api/v1/predict` para ver "
            "monitoramento em tempo real."
        )

# ==================================================================
# TAB 2 — Drift Detection
# ==================================================================

with tab_drift:
    st.subheader("🔍 Detecção de Drift")
    st.markdown(
        "O drift é detectado comparando a distribuição das predições atuais "
        "com a distribuição de baseline obtida durante o treinamento. "
        "Um desvio superior a **0.15** por classe é considerado *warning* e "
        "acima de **0.30** é *critical*."
    )

    # Always show baseline distribution from metadata
    baseline_meta = (local_meta or {}).get("baseline_stats", {})
    baseline_dist = baseline_meta.get("prediction_distribution", {})

    if baseline_dist:
        st.markdown("---")
        st.markdown("#### Distribuição de Baseline (Treinamento)")
        bl_df = pd.DataFrame(
            {
                "Classe": list(baseline_dist.keys()),
                "Proporção": list(baseline_dist.values()),
            }
        ).set_index("Classe")
        st.bar_chart(bl_df)

    if stats and stats.get("total_predictions", 0) > 0:
        drift_status = stats.get("drift_status", {})
        drift_details = drift_status.get("details", {})
        severity = drift_status.get("severity", "none")
        max_diff = drift_status.get("max_difference", 0)

        st.markdown("---")
        st.markdown("#### Baseline vs. Distribuição Atual")

        if drift_details:
            col_chart, col_table = st.columns([3, 2])

            with col_chart:
                chart_data = pd.DataFrame(
                    {
                        "Baseline": {
                            k: v["baseline"] for k, v in drift_details.items()
                        },
                        "Atual": {
                            k: v["current"] for k, v in drift_details.items()
                        },
                    }
                )
                st.bar_chart(chart_data)

            with col_table:
                detail_rows = []
                for cls, vals in drift_details.items():
                    detail_rows.append(
                        {
                            "Classe": cls,
                            "Baseline": f"{vals['baseline']:.4f}",
                            "Atual": f"{vals['current']:.4f}",
                            "Diferença": f"{vals['difference']:+.4f}",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(detail_rows).set_index("Classe"),
                    use_container_width=True,
                )

            # Drift difference gauge
            st.markdown("#### Diferença Máxima por Classe")
            diff_data = {
                k: abs(v["difference"]) for k, v in drift_details.items()
            }
            diff_df = pd.DataFrame(
                {
                    "Classe": list(diff_data.keys()),
                    "Diferença Absoluta": list(diff_data.values()),
                }
            ).set_index("Classe")
            st.bar_chart(diff_df)

            # Alert box
            if drift_status.get("is_drifted"):
                if severity == "critical":
                    st.error(
                        f"🔴 **Drift Crítico detectado!** "
                        f"Diferença máxima: **{max_diff:.4f}** "
                        f"(limiar: 0.15). Recomenda-se re-treinar o modelo."
                    )
                else:
                    st.warning(
                        f"⚠️ **Drift Warning detectado!** "
                        f"Diferença máxima: **{max_diff:.4f}** "
                        f"(limiar: 0.15). Monitore de perto."
                    )
            else:
                st.success(
                    f"✅ **Sem drift significativo.** "
                    f"Diferença máxima: **{max_diff:.4f}**."
                )
        else:
            msg = drift_status.get(
                "message", "Sem dados de drift disponíveis."
            )
            st.info(msg)

        # Current prediction distribution
        st.markdown("---")
        st.markdown("#### Distribuição Atual das Predições")
        dist = stats.get("prediction_distribution", {})
        if dist:
            dist_df = pd.DataFrame(
                {
                    "Classe": list(dist.keys()),
                    "Proporção": list(dist.values()),
                }
            ).set_index("Classe")
            st.bar_chart(dist_df)
        else:
            st.info("Nenhuma distribuição disponível.")
    else:
        st.markdown("---")
        st.info(
            "Nenhuma predição registrada ainda. Envie predições via "
            "`POST /api/v1/predict` para ver dados de monitoramento."
        )

# ==================================================================
# TAB 3 — Predições Recentes
# ==================================================================

with tab_predictions:
    st.subheader("🕐 Predições Recentes")

    if stats and stats.get("total_predictions", 0) > 0:
        recent = stats.get("recent_predictions", [])

        if recent:
            recent_df = pd.DataFrame(recent)

            # Parse timestamp
            if "timestamp" in recent_df.columns:
                recent_df["timestamp"] = pd.to_datetime(
                    recent_df["timestamp"]
                )
                recent_df = recent_df.sort_values(
                    "timestamp", ascending=False
                )

            # KPIs
            r1, r2, r3 = st.columns(3)
            r1.metric("Predições Mostradas", len(recent_df))

            if "probability" in recent_df.columns:
                avg_p = recent_df["probability"].mean()
                r2.metric("Confiança Média (últimas)", f"{avg_p:.4f}")

                min_p = recent_df["probability"].min()
                r3.metric("Confiança Mínima (últimas)", f"{min_p:.4f}")

            st.markdown("---")

            # Table
            display_cols = [
                c
                for c in [
                    "timestamp",
                    "prediction",
                    "probability",
                    "model_version",
                ]
                if c in recent_df.columns
            ]
            st.dataframe(
                recent_df[display_cols],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("---")

            # Confidence distribution
            if "probability" in recent_df.columns:
                st.markdown(
                    "#### Distribuição de Confiança (Predições Recentes)"
                )
                prob_bins = pd.cut(
                    recent_df["probability"],
                    bins=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    labels=[
                        "0-50%",
                        "50-60%",
                        "60-70%",
                        "70-80%",
                        "80-90%",
                        "90-100%",
                    ],
                )
                bin_counts = prob_bins.value_counts().sort_index()
                st.bar_chart(bin_counts)

            # Prediction class distribution (recent)
            if "prediction" in recent_df.columns:
                st.markdown(
                    "#### Distribuição por Classe (Predições Recentes)"
                )
                class_counts = recent_df["prediction"].value_counts()
                st.bar_chart(class_counts)
        else:
            st.info("Nenhuma predição recente disponível.")
    else:
        st.info(
            "Nenhuma predição registrada ainda. Envie predições via "
            "`POST /api/v1/predict` para ver dados aqui."
        )

# ==================================================================
# TAB 4 — Detalhes do Modelo
# ==================================================================

with tab_model:
    st.subheader("🔧 Detalhes do Modelo")

    m1, m2 = st.columns(2)

    with m1:
        st.markdown("#### Informações Gerais")
        info_rows = {
            "Nome do Modelo": meta.get("model_name", "N/A"),
            "Versão": meta.get("version", "N/A"),
            "Treinado em": meta.get("trained_at", "N/A"),
        }
        train_shape = meta.get("training_data_shape")
        test_shape = meta.get("test_data_shape")
        if train_shape:
            info_rows["Dados de Treino"] = (
                f"{train_shape[0]} amostras × {train_shape[1]} features"
            )
        if test_shape:
            info_rows["Dados de Teste"] = (
                f"{test_shape[0]} amostras × {test_shape[1]} features"
            )

        class_order = meta.get("class_order", [])
        if class_order:
            info_rows["Classes"] = ", ".join(str(c) for c in class_order)

        st.table(
            pd.DataFrame(
                info_rows.items(), columns=["Campo", "Valor"]
            ).set_index("Campo")
        )

    with m2:
        st.markdown("#### Hiperparâmetros")
        hyper = meta.get("hyperparameters", {})
        if hyper:
            st.table(
                pd.DataFrame(
                    [(k, str(v)) for k, v in hyper.items()],
                    columns=["Parâmetro", "Valor"],
                ).set_index("Parâmetro")
            )
        else:
            st.info("Sem hiperparâmetros disponíveis.")

    st.markdown("---")

    # Features list
    st.markdown("#### Features de Entrada do Modelo")
    input_features = meta.get("input_features", [])
    processed_features = meta.get("features", [])

    if input_features:
        feat_col1, feat_col2 = st.columns(2)
        with feat_col1:
            st.markdown("**Features Brutas (input)**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "#": range(1, len(input_features) + 1),
                        "Feature": input_features,
                    }
                ).set_index("#"),
                use_container_width=True,
            )
        with feat_col2:
            st.markdown("**Features Processadas (após pipeline)**")
            st.dataframe(
                pd.DataFrame(
                    {
                        "#": range(1, len(processed_features) + 1),
                        "Feature": processed_features,
                    }
                ).set_index("#"),
                use_container_width=True,
            )
    elif processed_features:
        st.dataframe(
            pd.DataFrame(
                {
                    "#": range(1, len(processed_features) + 1),
                    "Feature": processed_features,
                }
            ).set_index("#"),
            use_container_width=True,
        )
    else:
        st.info("Sem lista de features disponível.")

    st.markdown("---")

    # Baseline stats
    st.markdown("#### Estatísticas de Baseline")
    baseline_stats_data = (local_meta or meta).get("baseline_stats", {})
    if baseline_stats_data:
        bs1, bs2 = st.columns(2)
        with bs1:
            st.metric(
                "Total de Amostras (treino)",
                baseline_stats_data.get("total_samples", "N/A"),
            )
            st.metric(
                "Confiança Média (baseline)",
                f"{baseline_stats_data.get('avg_confidence', 0):.2f}",
            )
        with bs2:
            bl_dist = baseline_stats_data.get("prediction_distribution", {})
            if bl_dist:
                st.markdown("**Distribuição de Baseline**")
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Classe": list(bl_dist.keys()),
                            "Proporção": [
                                f"{v:.4f}" for v in bl_dist.values()
                            ],
                        }
                    ).set_index("Classe"),
                    use_container_width=True,
                )
    else:
        st.info("Sem dados de baseline disponíveis.")

    # Raw metadata JSON (collapsible)
    st.markdown("---")
    with st.expander("📄 Metadata JSON (raw)"):
        st.json(meta)


# ------------------------------------------------------------------
# Auto-refresh
# ------------------------------------------------------------------

if auto_refresh:
    time.sleep(30)
    st.rerun()
