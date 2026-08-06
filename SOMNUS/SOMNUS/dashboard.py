"""SOMNUS dashboard.

Two structural fixes over the previous version:

  * @st.cache_resource on every client. Streamlit reruns the whole script on
    each interaction; the old code opened a fresh connection pool (1-5 conns)
    per rerun and never closed them, exhausting CockroachDB Serverless's
    connection limit within a couple of minutes of clicking. On stage.
  * Buttons send COMMANDS to the running agent instead of mutating a local
    throwaway Simulator in a different process. The old "Inject Anomaly"
    button silently did nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.control import send_command  # noqa: E402
from core.state_store import read_state  # noqa: E402
from infra.config import CONFIG, OFFLINE  # noqa: E402

st.set_page_config(page_title="SOMNUS", page_icon="brain", layout="wide")


@st.cache_resource
def get_store():
    if OFFLINE or not CONFIG.db_url:
        from memory.inmemory import InMemoryStore

        return InMemoryStore()
    from memory.cortex import CockroachStore

    return CockroachStore()


@st.cache_data(ttl=10)
def get_memory_stats() -> dict:
    from mcp_server.introspection import table_stats

    try:
        return table_stats(get_store())
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@st.cache_data(ttl=30)
def load_benchmark() -> dict | None:
    path = Path("data/benchmark.json")
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


st.title("SOMNUS")
st.caption("Predictive coding | neuromodulation | complementary learning systems")

tab_mind, tab_memory, tab_bench = st.tabs(["The mind", "Memory", "Forgetting benchmark"])

# ---------------------------------------------------------------- the mind
with tab_mind:
    state = read_state()
    wake = state.get("wake", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cycle", wake.get("cycle", 0))
    c2.metric("Surprise S(t)", f"{wake.get('surprise', 0):.3f}")
    c3.metric("ACh (expected unc.)", f"{wake.get('ach', 0):.3f}")
    c4.metric("NA (z-score)", f"{wake.get('na', 0):.2f}")
    c5.metric("DA", f"{wake.get('da', 0):.3f}")

    if wake.get("boundary"):
        st.error("CONTEXT BOUNDARY — noradrenaline exceeded threshold. Priors reset.")

    lb = wake.get("last_boundary")
    if lb:
        if lb.get("outcome") == "restored":
            st.success(
                f"Prior RESTORED from schema `{lb.get('schema_label')}` "
                f"(distance {lb.get('distance')}) — the agent recognised this world."
            )
        else:
            st.info(f"New context `{lb.get('label')}` — no matching schema; learning from scratch.")

    history = wake.get("history", [])
    if history:
        df = pd.DataFrame(history).set_index("cycle")
        st.line_chart(df[["surprise", "ach"]], height=220)
        st.caption("Surprise against the learned noise model. NA fires when the gap is many sigma.")

    st.subheader("Perturb the world")
    st.caption("Commands go to the running agent process via the shared command channel.")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Inject anomaly"):
        send_command("inject_anomaly", ticks=8)
        st.toast("Anomaly queued")
    if b2.button("Shift to surge"):
        send_command("set_regime", regime="surge")
        st.toast("Regime -> surge")
    if b3.button("Return to steady"):
        send_command("set_regime", regime="steady")
        st.toast("Regime -> steady")
    if b4.button("Force sleep"):
        from sleep_cycle.consolidation import consolidate

        with st.spinner("Consolidating hippocampus -> neocortex..."):
            report = consolidate(get_store())
        st.success(report.to_dict())

    with st.expander("Raw agent state"):
        st.json(state)

# ------------------------------------------------------------------ memory
with tab_memory:
    st.subheader("Substrate")
    st.json(get_memory_stats())

    st.subheader("Consolidated schemas")
    try:
        schemas = get_store().all_schemas()
        if not schemas:
            st.info("No schemas yet. Run the agent, then force a sleep cycle.")
        else:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "label": s.label[:60],
                            "support": s.support_count,
                            "stability": s.stability,
                            "alpha": round(0.06 * 2**-s.stability, 5),
                            "origin": s.origin,
                            "skill": s.skill_ref or "",
                        }
                        for s in schemas
                    ]
                ),
                width="stretch",
            )
            st.caption(
                "Metaplasticity: alpha = alpha_base x 2^-stability. A schema confirmed "
                "many times is effectively frozen; an NA violation decrements stability "
                "and re-opens it."
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Store unavailable: {exc}")

    skills_dir = Path("skills")
    if skills_dir.exists():
        files = sorted(skills_dir.glob("*.md"))
        if files:
            st.subheader(f"Compiled skills ({len(files)})")
            st.caption("Schemas that hardened past threshold became procedures. The agent wrote these.")
            for f in files:
                with st.expander(f.name):
                    st.markdown(f.read_text(encoding="utf-8"))

# --------------------------------------------------------------- benchmark
with tab_bench:
    st.subheader("Catastrophic forgetting: SOMNUS vs control")
    payload = load_benchmark()
    if payload is None:
        st.warning("No results yet. Run:  `python -m eval.benchmark --seeds 12`")
    else:
        results = pd.DataFrame(payload["results"])
        st.dataframe(results, width="stretch")

        curves = payload["curves"]
        frame = pd.DataFrame(
            {name: [p["model_error"] for p in pts] for name, pts in curves.items()}
        )
        frame["tick"] = [p["tick"] for p in next(iter(curves.values()))]
        show = [c for c in ("control", "somnus") if c in frame.columns]
        st.line_chart(frame.set_index("tick")[show], height=320)
        st.caption(
            "Model error against the true regime mean, so the irreducible noise floor "
            "is excluded. Phase 4 is the return to Task A — the divergence there is "
            "catastrophic forgetting."
        )
        with st.expander("All ablation arms"):
            st.line_chart(frame.set_index("tick")[list(curves.keys())], height=320)

st.sidebar.metric("Mode", "offline" if OFFLINE or not CONFIG.db_url else "cockroachdb")
if st.sidebar.button("Refresh"):
    st.rerun()
