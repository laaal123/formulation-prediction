import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fi_model_arena import run_arena
from fi_models import lenth_effects, regime_label

st.title("🏆 Model Arena")
st.caption("detect task → gate → candidates → nested CV → rank by Q² → "
           "1-SE rule → y-randomisation → applicability domain")

df = st.session_state.get("df")
X_cols = st.session_state.get("X_cols") or []
y_col = st.session_state.get("y_col")
if df is None or not X_cols or not y_col:
    st.warning("Load data and select factors on the main page first.")
    st.stop()

X, y = df[X_cols], df[y_col]
st.info(regime_label(len(X), False))

c1, c2, c3 = st.columns(3)
with c1:
    budget = st.slider("Permutation budget per model (s)", 5, 120, 25, 5)
with c2:
    top_k = st.slider("Contenders permutation-tested", 2, 8, 4)
with c3:
    force = st.checkbox("Override the design gate", value=False,
                        help="Fits models on an inadmissible design. "
                             "Results are not defensible.")

if force:
    st.error("Gate override is ON. Anything below is for exploration only "
             "and must not support a regulatory decision.")

if st.button("Run arena", type="primary"):
    with st.spinner("Cross-validating candidates and building permuted nulls…"):
        st.session_state.arena = run_arena(
            X, y, seed=st.session_state.get("seed", 42),
            top_k=top_k, perm_budget_s=budget, force=force)

a = st.session_state.get("arena")
if a is None:
    st.info("Press **Run arena** to start.")
    st.stop()

for m in a.messages:
    (st.error if "BLOCK" in m or "NO SIGNAL" in m else st.warning)(m)

if a.leaderboard.empty:
    st.subheader("Design gate blocked modelling")
    st.code(a.verdict.summary())
    st.stop()

st.divider()
st.subheader("Selection")
st.code(a.explain_selection(), language=None)

st.subheader("Leaderboard")
st.caption("Ranked on Q² (predicted R²). Training R² is shown for "
           "diagnosis only and never used to rank.")
lb = a.leaderboard.copy()


def _style(row):
    if a.selected is not None and row["model"] == a.selected.name:
        return ["background-color: #1b4332; color: white"] * len(row)
    if not row["signal"] and row["tested"]:
        return ["background-color: #4a1f1f; color: white"] * len(row)
    return [""] * len(row)


st.dataframe(lb.style.apply(_style, axis=1), width="stretch", hide_index=True)
st.caption("`tested` = permutation-tested. `signal` = beat its permuted "
           "null. Untested models were too expensive within the budget and "
           "are excluded from selection.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("Q² vs training R²")
    m = lb.dropna(subset=["R2_train"])
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Q² (predicted)", x=m["model"], y=m["Q2"]))
    fig.add_trace(go.Bar(name="R² (training)", x=m["model"], y=m["R2_train"]))
    fig.update_layout(barmode="group", height=380, yaxis_title="R²",
                      xaxis_tickangle=-40)
    st.plotly_chart(fig, width="stretch")
    st.caption("A large gap is overfitting. Trust the Q² bar.")
with c2:
    st.subheader("Observed vs cross-validated prediction")
    if a.selected is not None and a.selected.y_true is not None:
        yt, yp = a.selected.y_true, a.selected.y_pred
        fig = px.scatter(x=yt, y=yp, labels={"x": f"observed {y_col}",
                                             "y": "out-of-fold prediction"})
        lo, hi = float(np.nanmin(yt)), float(np.nanmax(yt))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 name="identity",
                                 line=dict(dash="dash", color="grey")))
        fig.update_traces(marker=dict(size=10),
                          selector=dict(mode="markers"))
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")

st.divider()
st.subheader("Y-randomisation (permutation test)")
st.caption("The response is shuffled and the model refitted. If the real "
           "Q² is not clearly above this null, the model is reporting "
           "chance structure.")
rows = [
    {"model": r.name, "observed Q²": round(r.q2, 4),
     "null mean Q²": round(r.perm_null_mean, 4),
     "null 95th pct": round(r.perm_null_q95, 4),
     "p-value": round(r.perm_p, 4),
     "shuffles": getattr(r, "_n_perm", 0),
     "signal": "✅" if r.signal else "❌"}
    for r in a.results
    if np.isfinite(getattr(r, "perm_p", np.nan))
]
if rows:
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

if a.no_signal:
    st.error(
        "**NO SIGNAL.** No candidate separated from its permuted null. "
        "Reporting a model here would be reporting noise. Either the "
        "factors do not drive this response, or the design cannot resolve "
        "the effect. Design a better experiment rather than a better model."
    )

# ---------------- Lenth's method for small structured designs ----------
if len(X) < 20 and not a.no_signal:
    st.divider()
    st.subheader("Lenth's method (unreplicated design)")
    st.caption("Pseudo standard error for designs with no replication — "
               "the ANOVA-adjacent view regulators recognise.")
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        Z = StandardScaler().fit_transform(X)
        eff = LinearRegression().fit(Z, y).coef_.ravel() * 2
        le = pd.DataFrame(lenth_effects(eff, X_cols))
        st.dataframe(
            le[["factor", "effect", "ME", "SME",
                "significant_ME", "significant_SME"]],
            width="stretch", hide_index=True)
        fig = px.bar(le.sort_values("abs_effect"), x="effect", y="factor",
                     orientation="h", color="significant_ME")
        fig.add_vline(x=float(le["ME"].iloc[0]), line_dash="dash")
        fig.add_vline(x=-float(le["ME"].iloc[0]), line_dash="dash")
        fig.update_layout(height=340)
        st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.warning(f"Lenth's method unavailable: {exc}")
