import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fi_mixed_models import (RandomTerm, compare_pooled_vs_mixed,
                               effective_sample_size, fit_lmm,
                               fit_random_intercept, lrt_random_effect,
                               lrt_terms)

st.title("🏭 Mixed Effects — batch, site and campaign")
st.caption("Pooling across campaigns is the quiet error in formulation "
           "analytics. This page measures what it costs.")

st.info("""
If batches differ systematically, observations within a batch are correlated.
Pooled regression then reports standard errors that are too small and treats
the row count as the sample size. **48 rows across six campaigns with a high
ICC can be a genuine n of twelve** — and the design gate will admit it on the
row count unless you tell it about the grouping.
""")

df = st.session_state.get("df")
if df is None:
    st.warning("Load data on the main page first.")
    st.stop()

cat_like = [c for c in df.columns
            if df[c].dtype == object or df[c].nunique() <= max(len(df) // 2, 2)]
num = df.select_dtypes(include=[np.number]).columns.tolist()

if not cat_like:
    st.error("No column looks like a grouping variable (batch, site, "
             "campaign). Add one to use this page.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    gcol = st.selectbox("Grouping column (batch / site / campaign)", cat_like)
with c2:
    gcol2 = st.selectbox("Second grouping column — optional (crossed)",
                         ["(none)"] + [c for c in cat_like if c != gcol],
                         help="Two factors are CROSSED when every level of "
                              "one appears with every level of the other — "
                              "site and campaign, typically. Nesting them "
                              "would charge site-to-site variation to "
                              "campaigns.")
with c3:
    ycol = st.selectbox("Response", num,
                        index=len(num) - 1 if num else 0)

xcols = st.multiselect("Fixed-effect factors",
                       [c for c in num if c != ycol],
                       default=[c for c in num if c != ycol][:4])
slope_on = st.selectbox("Random slope on — optional", ["(none)"] + xcols,
                        help="Use when a factor's effect plausibly differs "
                             "between groups. The intercept and slope "
                             "variances are estimated independently; their "
                             "correlation is not.")

groups = df[gcol].to_numpy()
y = df[ycol].to_numpy(float)
sizes = pd.Series(groups).value_counts().sort_index()

st.divider()
m = st.columns(3)
m[0].metric("Rows", len(df))
m[1].metric("Groups", int(sizes.shape[0]))
m[2].metric("Group sizes", f"{int(sizes.min())}–{int(sizes.max())}")

if sizes.shape[0] < 2:
    st.error("Need at least two groups.")
    st.stop()

if st.button("Fit mixed model", type="primary"):
    X = df[xcols].to_numpy(float) if xcols else np.zeros((len(df), 0))
    terms = [RandomTerm(groups, gcol)]
    if gcol2 != "(none)":
        terms.append(RandomTerm(df[gcol2].to_numpy(), gcol2))
    if slope_on != "(none)" and xcols:
        terms.append(RandomTerm(groups, gcol,
                                slope=df[slope_on].to_numpy(float),
                                slope_name=slope_on))
    with st.spinner("Fitting REML mixed model…"):
        st.session_state.lmm = fit_lmm(X, y, terms, feature_names=xcols)
        st.session_state.lmm_terms = (
            lrt_terms(X, y, terms) if len(terms) > 1 else None)
        st.session_state.lmm_lrt = lrt_random_effect(X, y, groups)
        st.session_state.lmm_cmp = (
            compare_pooled_vs_mixed(X, y, groups, xcols) if xcols else None)
        st.session_state.lmm_eff = effective_sample_size(
            y, groups, X=X if xcols else None)

fit = st.session_state.get("lmm")
if fit is None:
    st.info("Press **Fit mixed model**.")
    st.stop()

st.divider()
st.subheader("Variance components")
st.code(fit.summary(), language=None)
for w in fit.warnings:
    st.warning(w)

eff = st.session_state.lmm_eff
m = st.columns(4)
m[0].metric("ICC", f"{fit.icc:.4f}")
m[1].metric("Design effect", f"{fit.design_effect:.2f}")
m[2].metric("Effective n", f"{fit.effective_n:.1f}",
            delta=f"{fit.effective_n - fit.n_obs:.1f} vs rows")
m[3].metric("Information lost", f"{eff['shrinkage_pct']:.0f}%")

if fit.icc > 0.10:
    st.error(
        f"**ICC = {fit.icc:.3f}.** The grouping is material. Your "
        f"{fit.n_obs} rows carry roughly **{fit.effective_n:.0f} independent "
        f"observations**. Re-check any conclusion drawn from a pooled "
        f"analysis of this dataset — including whether it should have "
        f"passed the design gate at all."
    )
else:
    st.success(f"ICC = {fit.icc:.3f}. Group structure is minor; pooling is "
               f"defensible for this response.")

terms_tbl = st.session_state.get("lmm_terms")
if terms_tbl is not None:
    st.divider()
    st.subheader("Which random term carries the structure?")
    st.dataframe(terms_tbl.round(5), width="stretch", hide_index=True)
    st.caption("Each term is dropped in turn and tested by likelihood ratio "
               "against the boundary mixture. With crossed factors this is "
               "the only way to separate them — a single pooled ICC cannot.")

lrt = st.session_state.lmm_lrt
st.divider()
st.subheader("Is the group structure real?")
c = st.columns(3)
c[0].metric("LR statistic", f"{lrt['lr_statistic']:.3f}")
c[1].metric("p-value", f"{lrt['p_value']:.4f}")
c[2].metric("Significant", "yes" if lrt["significant"] else "no")
(st.error if lrt["significant"] else st.success)(lrt["verdict"])
st.caption(f"Reference distribution: {lrt['reference']}. The null sits on "
           f"the boundary of the parameter space, so a plain χ²₁ test would "
           f"roughly double the p-value and hide real batch structure.")

if xcols:
    st.divider()
    st.subheader("Pooled OLS vs mixed model")
    cmp = st.session_state.lmm_cmp
    st.dataframe(cmp["table"].round(5), width="stretch", hide_index=True)
    if cmp["max_se_inflation"] > 1.15:
        st.error(cmp["verdict"])
    else:
        st.info(cmp["verdict"])
    st.caption("The column that matters is the standard-error ratio. Above 1, "
               "the pooled analysis was reporting false precision. Below 1, "
               "the mixed model has gained precision by removing the group "
               "offset from the residual — which only happens for factors "
               "that vary *within* group.")

    tb = cmp["table"].iloc[1:]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="OLS SE", x=tb["term"], y=tb["OLS_se"]))
    fig.add_trace(go.Bar(name="LMM SE", x=tb["term"], y=tb["LMM_se"]))
    fig.update_layout(barmode="group", height=350,
                      yaxis_title="standard error")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Fixed effects (mixed model)")
    tbl = fit.table()
    st.dataframe(tbl.round(5), width="stretch", hide_index=True)
    sig = tbl[(tbl["p_value"] < 0.05) & (tbl["term"] != "(Intercept)")]
    st.caption(f"{len(sig)} factor(s) significant at p<0.05 after accounting "
               f"for group structure.")
    st.info(
        "**Read the `df_satterthwaite` column.** A factor that varies "
        "*between* groups gets roughly as many degrees of freedom as there "
        "are groups; one that varies *within* groups gets close to the row "
        "count. Two factors in the same table can legitimately differ by an "
        "order of magnitude in df, and using the row count for both is how "
        "a between-group effect gets declared significant when it is not.")

st.divider()
st.subheader("Response by group")
plot = pd.DataFrame({gcol: groups.astype(str), ycol: y})
fig = px.box(plot, x=gcol, y=ycol, points="all")
fig.add_hline(y=float(np.mean(y)), line_dash="dash", line_color="grey",
              annotation_text="grand mean")
fig.update_layout(height=420)
st.plotly_chart(fig, width="stretch")
st.caption("Visible separation between boxes is the ICC made concrete. If "
           "the boxes barely overlap, pooling them was never defensible.")
