"""Streamlit dashboard for SOMNUS telemetry, hippocampus writes, and evaluation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from core.simulator import Simulator
from core.state_store import read_state
from eval.baseline import BaselineRAG, SomnusQuery
from infra.aws_client import AWSClient
from memory.hippocampus import Hippocampus
from sleep_cycle.lambda_handler import consolidate_episodes

load_dotenv()

st.set_page_config(page_title="SOMNUS Dashboard", page_icon="🧠", layout="wide")

st.title("SOMNUS — Brain-Inspired AI Agent")
st.caption("Active inference · Dual memory · Neuromodulation")

tab_telemetry, tab_hippocampus, tab_eval = st.tabs(
    ["Telemetry", "Hippocampus", "Evaluation"]
)

with tab_telemetry:
    col1, col2, col3 = st.columns(3)
    state = read_state()
    wake = state.get("wake", {})

    with col1:
        st.metric("Wake Cycle", wake.get("cycle", 0))
    with col2:
        error = wake.get("last_error", 0.0)
        st.metric("Prediction Error", f"{error:.3f}", delta=None)
        if error > 0.35:
            st.error("Neuromodulation spike — surprise threshold exceeded")
    with col3:
        st.metric("Surprised", "Yes" if wake.get("last_surprised") else "No")

    st.subheader("Live Telemetry (simulated)")
    sim = Simulator()
    telemetry = sim.emit()
    st.json(telemetry)

    if st.button("Inject Anomaly"):
        sim.trigger_anomaly()
        st.warning("Anomaly triggered — next wake cycle should detect surprise")

    st.subheader("Agent State")
    st.json(state)

    if st.button("Force REM Sleep"):
        with st.spinner("Consolidating hippocampus → cortex..."):
            result = consolidate_episodes()
        st.success(f"Sleep cycle complete: {json.dumps(result, indent=2)}")

with tab_hippocampus:
    st.subheader("Recent S3 Hippocampus Writes")
    try:
        hippocampus = Hippocampus()
        keys = hippocampus.list_recent_keys(limit=15)
        aws = AWSClient()

        if not keys:
            st.info("No episodes in hippocampus yet.")
        else:
            for key in keys:
                with st.expander(key):
                    try:
                        episode = aws.read_json(key)
                        st.json(episode)
                    except Exception as exc:
                        st.error(str(exc))
    except Exception as exc:
        st.warning(f"Hippocampus unavailable: {exc}")

with tab_eval:
    st.subheader("SOMNUS vs Baseline RAG")
    st.markdown(
        """
        **Baseline RAG** only searches consolidated cortex vectors.
        **SOMNUS** also reads recent hippocampus episodes (un-vectorized context),
        demonstrating superiority for continuous learning scenarios.
        """
    )

    question = st.text_input(
        "Evaluation query",
        value="What remediation should we apply for a sudden traffic spike?",
    )

    if st.button("Run Comparison"):
        baseline = BaselineRAG()
        somnus = SomnusQuery()

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Baseline RAG")
            try:
                baseline_result = baseline.query(question)
                st.write(baseline_result.answer)
                st.caption(f"Sources: {len(baseline_result.sources)} cortex rules")
                with st.expander("Sources"):
                    st.json(baseline_result.to_dict())
            except Exception as exc:
                st.error(f"Baseline failed: {exc}")

        with col_b:
            st.markdown("### SOMNUS (Dual Memory)")
            try:
                somnus_result = somnus.query(question)
                st.write(somnus_result.answer)
                st.caption(
                    f"Sources: {len(somnus_result.sources)} "
                    "(cortex rules + hippocampus episodes)"
                )
                with st.expander("Sources"):
                    st.json(somnus_result.to_dict())
            except Exception as exc:
                st.error(f"SOMNUS query failed: {exc}")

st.sidebar.markdown(f"Last refresh: {datetime.now(timezone.utc).isoformat()}")
if st.sidebar.button("Refresh"):
    st.rerun()

time.sleep(0.1)
