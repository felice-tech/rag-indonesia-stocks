"""Streamlit dashboard — entry point for interactive visualisation.

Spec section 9 — provides a lightweight UI that calls the FastAPI
endpoints under the hood.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import streamlit as st

from rag_geopolitik.features.store import FeatureStore
from rag_geopolitik.models.flow_model import FlowModel
from rag_geopolitik.models.stock_model import StockModel
from rag_geopolitik.recommendation.recommender import Recommender

# Page configuration — must be the first Streamlit command.
st.set_page_config(
    page_title="RAG Geopolitik & Investasi",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def run_dashboard(
    flow_model: FlowModel | None = None,
    stock_model: StockModel | None = None,
    recommender: Recommender | None = None,
    feature_store: FeatureStore | None = None,
) -> None:
    """Launch the Streamlit dashboard.

    Parameters
    ----------
    flow_model : FlowModel, optional
    stock_model : StockModel, optional
    recommender : Recommender, optional
    feature_store : FeatureStore, optional
    """
    st.title("📈 RAG Geopolitik & Investasi")
    st.markdown(
        "Prediksi foreign flow & saham LQ45 berdasarkan analisis berita geopolitik, "
        "makro, dan komoditas."
    )

    # Sidebar
    st.sidebar.header("🔧 Controls")
    selected_ticker = st.sidebar.selectbox(
        "Pilih Ticker LQ45",
        ["BBCA", "BBRI", "BMRI", "BBNI", "ANTM", "ADRO", "TLKM", "UNVR"],
    )
    refresh = st.sidebar.button("🔄 Refresh")

    # Main content — three columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌊 Foreign Flow Prediction")
        if flow_model is not None and flow_model.is_trained:
            pred = flow_model.predict({})
            st.metric(
                label="Direction",
                value=pred.direction.value.upper(),
                delta=pred.estimated_value,
            )
            st.progress(pred.confidence)
            st.caption(f"Confidence: {pred.confidence:.0%}")
            if pred.driving_factors:
                st.markdown("**Driving factors:**")
                for f in pred.driving_factors:
                    st.markdown(f"- {f}")
        else:
            st.info("Model A (flow) belum di-train atau tidak tersedia.")

        st.subheader("📰 Latest Events")
        st.caption("(Coming soon — event feed from Qdrant)")

    with col2:
        st.subheader(f"📊 {selected_ticker} — Stock Opportunity")
        if recommender is not None:
            opp = recommender.recommend(
                ticker=selected_ticker,
                ticker_features={},
                events=[],
            )
            st.metric(
                label="Outperform Probability",
                value=f"{opp.outperform_probability:.0%}",
                delta=f"Confidence: {opp.confidence:.0%}",
            )
            st.progress(opp.outperform_probability)

            if opp.summary:
                st.markdown(f"**Summary:** {opp.summary}")

            if opp.explanation_bullets:
                st.markdown("**Explanation:**")
                for b in opp.explanation_bullets:
                    st.markdown(f"- {b}")

            if opp.risk_flags:
                st.warning("**Risk Flags:**")
                for flag in opp.risk_flags:
                    st.markdown(f"- ⚠️ {flag}")
        else:
            st.info("Recommender belum dikonfigurasi.")

    # Footer / data section
    st.divider()
    with st.expander("⚙️ System Status"):
        cols = st.columns(3)
        cols[0].metric("Flow Model", "Ready" if flow_model and flow_model.is_trained else "N/A")
        cols[1].metric("Stock Model", "Ready" if stock_model and stock_model.is_trained else "N/A")
        cols[2].metric("Feature Store", "Connected" if feature_store else "N/A")


if __name__ == "__main__":
    # Allow running as `streamlit run rag_geopolitik/dashboard/app.py`
    run_dashboard()