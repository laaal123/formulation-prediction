"""
Formulation Intelligence - Streamlit entry point.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fi_design_diagnostics import diagnose, spurious_correlation_risk  # noqa: E402


DEMOS = {
    "Matrix tablet DoE (n=48, real signal)": "demo_matrix_doe.csv",
    "Pure noise (n=30, must report NO SIGNAL)": "demo_pure_noise.csv",
    "Real n=3 Eudragit case (must be BLOCKED)": "demo_n3_eudragit.csv",
    "Mixture design (n=30, Scheffe)": "demo_mixture.csv",
    "Percolation series (n=6, threshold)": "demo_percolation.csv",
    "Multi-campaign DoE (n=48, high ICC)": "demo_multi_campaign.csv",
    "Campaign-banded DoE (CV leakage)": "demo_campaign_banded.csv",
    "Crossed site x campaign": "demo_crossed_site_campaign.csv",
    "Highly variable drug scenarios": "demo_highly_variable.csv",
}


st.title("⚗️ Formulation Intelligence")
st.caption(
    "Factor identification, dissolution modelling and virtual "
    "bioequivalence for small-n pharmaceutical formulation data."
)

st.markdown("""
### The rule this app is built around

Below a minimum sample size, or whenever the design matrix is rank
deficient, **no statistical model is admissible** — and this app will
refuse to fit one rather than return a number that looks authoritative
and is not. It routes to mechanistic analysis instead.

That refusal is the feature. Small-n formulation data produces chance
correlations constantly, and a model fitted past the gate will find them.
""")

tab_load, tab_demo, tab_why = st.tabs(
    ["📁 Load data", "🎲 Demo datasets", "📖 Why the gate exists"])

with tab_load:
    up = st.file_uploader("Upload a CSV of formulations", type=["csv"])
    if up is not None:
        st.session_state.df = pd.read_csv(up)
        st.success(f"Loaded {up.name}")

with tab_demo:
    choice = st.selectbox("Demo dataset", list(DEMOS))
    if st.button("Load demo", type="primary"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            DEMOS[choice])
        if os.path.exists(path):
            st.session_state.df = pd.read_csv(path)
            st.success(f"Loaded: {choice}")
        else:
            st.error("Demo file missing. Run: python simulate_data.py")

with tab_why:
    c1, c2 = st.columns(2)
    with c1:
        n_demo = st.slider("Sample size (n)", 3, 30, 3)
        p_demo = st.slider("Number of factors", 2, 12, 4)
    with c2:
        with st.spinner("Simulating..."):
            risk = spurious_correlation_risk(n_demo, p_demo, n_sim=2000)
        st.markdown(f"**With n={n_demo} and {p_demo} pure-noise factors:**")
        for k, v in risk.items():
            st.metric(k, f"{v:.1%}")
    st.warning(
        f"At n={n_demo} with {p_demo} factors, a correlation bar chart is "
        f"not evidence. The probability that at least one *entirely "
        f"random* factor reaches |r| ≥ 0.95 against a random response is "
        f"{risk['P(max|r|>=0.95)']:.0%}. This is why the app ranks on "
        f"cross-validated Q² against a permuted null, never on correlation."
    )

if st.session_state.df is not None:
    df = st.session_state.df
    st.divider()
    st.subheader("Preview")
    st.dataframe(df.head(12), width="stretch")

    num = df.select_dtypes(include=[np.number]).columns.tolist()
    c1, c2 = st.columns(2)
    with c1:
        y_col = st.selectbox(
            "Response (y)", num,
            index=len(num) - 1 if num else 0,
            key="y_select")
    with c2:
        X_cols = st.multiselect(
            "Factors (X)", [c for c in num if c != y_col],
            default=[c for c in num if c != y_col][:8],
            key="x_select")

    st.session_state.y_col = y_col
    st.session_state.X_cols = X_cols

    cat_like = [c for c in df.columns
                if df[c].dtype == object
                or (df[c].nunique() <= max(len(df) // 3, 2) and c not in X_cols)]
    gcol = st.selectbox(
        "Grouping column (batch / site / campaign) — optional",
        ["(none)"] + cat_like,
        help="If rows come from more than one campaign, the row count "
             "overstates the information available. Selecting a grouping "
             "column makes the gate test the EFFECTIVE sample size instead.")
    st.session_state.group_col = None if gcol == "(none)" else gcol

    if X_cols:
        groups = (df[gcol].to_numpy()
                  if st.session_state.group_col else None)
        v = diagnose(df[X_cols], df[y_col], groups=groups)
        st.divider()
        st.subheader("Design gate")
        m = st.columns(6)
        m[0].metric("n", v.n_samples)
        m[1].metric("Factors", v.n_features)
        m[2].metric("Rank", v.rank_centered,
                    delta=f"{v.rank_centered - v.n_features}" if
                    v.rank_centered < v.n_features else None)
        m[3].metric("Effective d.o.f.", v.effective_dof)
        m[4].metric("Effective n",
                    f"{v.effective_n:.1f}" if v.n_groups else "—",
                    delta=(f"{v.effective_n - v.n_samples:.1f}"
                           if v.n_groups else None))
        m[5].metric("Mode", v.mode.upper())

        if v.admissible:
            st.success("✅ Admissible — statistical modelling is available. "
                       "Go to **Model Arena**.")
        else:
            st.error("🚫 BLOCKED — statistical modelling is not admissible.")
            for b in v.blockers:
                st.markdown(f"- {b}")
            st.info("Routed to mechanistic mode. Use **Percolation & "
                    "Threshold** and **IVIVC & Virtual BE**.")
        for w in v.warnings:
            st.warning(w)
else:
    st.info("Load a dataset or a demo above to begin.")
