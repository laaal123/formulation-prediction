import os, sys
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fi_factor_id import consensus, VIP_THRESHOLD, STABILITY_THRESHOLD, HAS_SHAP

st.title("🎯 Factor Identification")
st.caption("Several methods, reported by agreement. Correlated factors mislead any single one.")

df = st.session_state.get("df"); X_cols = st.session_state.get("X_cols") or []
y_col = st.session_state.get("y_col")
if df is None or not X_cols or not y_col:
    st.warning("Load data and select factors on the main page first."); st.stop()

X, y = df[X_cols], df[y_col]
a = st.session_state.get("arena")
est = a.selected_estimator if a is not None else None

if a is not None and a.no_signal:
    st.error("The Model Arena found NO SIGNAL. Importance scores below "
             "describe noise structure and must not be used to choose the "
             "next experiment.")

c1, c2, c3 = st.columns(3)
with c1: ncomp = st.slider("PLS components for VIP", 1, min(5, max(len(X_cols),1)), 2)
with c2: run_perm = st.checkbox("Permutation importance (slow)", value=len(X) < 60)
with c3: st.caption(f"SHAP {'available' if HAS_SHAP else 'not installed'}")

if st.button("Identify drivers", type="primary"):
    with st.spinner("Running VIP, bootstrap coefficients, stability selection..."):
        st.session_state.fid = consensus(X, y, est, ncomp,
                                         seed=st.session_state.get("seed", 42),
                                         run_permutation=run_perm)

res = st.session_state.get("fid")
if res is None:
    st.info("Press **Identify drivers**."); st.stop()

st.subheader("Consensus")
show = [c for c in ["cluster","VIP","stability","coef_mean","coef_lo95","coef_hi95",
                    "sign_consistency","importance","imp_lo","imp_hi",
                    "SHAP_mean_abs","votes","verdict"] if c in res.columns]
st.dataframe(res[show].round(4), width="stretch")
st.caption(f"`votes` counts how many of {int(res['n_methods'].iloc[0])} independent "
           f"methods flag the factor. DRIVER = near-unanimous. One vote means "
           f"design a better experiment, not a conclusion.")

c1, c2 = st.columns(2)
with c1:
    if "VIP" in res:
        st.subheader("VIP scores")
        fig = px.bar(res.sort_values("VIP"), x="VIP", y=res.sort_values("VIP").index,
                     orientation="h", color="VIP", color_continuous_scale="Viridis")
        fig.add_vline(x=VIP_THRESHOLD, line_dash="dash", line_color="red",
                      annotation_text="VIP = 1")
        fig.update_layout(height=360, yaxis_title="")
        st.plotly_chart(fig, width="stretch")
with c2:
    if "stability" in res:
        st.subheader("Stability selection")
        s = res.sort_values("stability")
        fig = px.bar(s, x="stability", y=s.index, orientation="h",
                     color="stability", color_continuous_scale="Plasma",
                     range_color=[0,1])
        fig.add_vline(x=STABILITY_THRESHOLD, line_dash="dash", line_color="red")
        fig.update_layout(height=360, yaxis_title="", xaxis_title="selection frequency")
        st.plotly_chart(fig, width="stretch")
        st.caption("Bootstrap Elastic Net. A factor selected in 90% of subsamples "
                   "is real; 40% is a coin flip however large its coefficient looks.")

if {"coef_lo95","coef_hi95"}.issubset(res.columns):
    st.divider(); st.subheader("Bootstrap coefficient intervals")
    r = res.reset_index().rename(columns={"index": "factor"})
    fig = px.scatter(r, x="coef_mean", y="factor",
                     error_x=r["coef_hi95"]-r["coef_mean"],
                     error_x_minus=r["coef_mean"]-r["coef_lo95"])
    fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_traces(marker=dict(size=11)); fig.update_layout(height=360)
    st.plotly_chart(fig, width="stretch")
    st.caption("An interval crossing zero means the sign of the effect is not "
               "established, regardless of the point estimate.")

if res["cluster"].nunique() < len(res):
    st.divider()
    st.warning("**Correlated clusters present.** Factors sharing a cluster id "
               "cannot be separated by this design — report them as a group "
               "and break the correlation in the next experiment.")
    st.dataframe(res.reset_index().rename(columns={"index":"factor"})
                 [["factor","cluster","votes","verdict"]].sort_values("cluster"),
                 width="stretch", hide_index=True)
