import os, sys
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fi_design_diagnostics import diagnose, spurious_correlation_risk
from fi_factor_id import correlation_clusters

st.title("🔬 Design Diagnostics")
st.caption("Run before any model. Decides whether a regression problem exists at all.")

df = st.session_state.get("df")
X_cols = st.session_state.get("X_cols") or []
y_col = st.session_state.get("y_col")
if df is None or not X_cols or not y_col:
    st.warning("Load data and select factors on the main page first.")
    st.stop()

X, y = df[X_cols], df[y_col]
v = diagnose(X, y)

st.subheader("Verdict")
st.code(v.summary(), language=None)

c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Correlation with the response - and why it misleads")
    corr = X.corrwith(y).sort_values(key=np.abs, ascending=False)
    fig = px.bar(x=corr.values, y=corr.index, orientation="h",
                 labels={"x": "Pearson r", "y": ""},
                 color=corr.values, color_continuous_scale="RdBu",
                 range_color=[-1, 1])
    fig.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig, width="stretch")
with c2:
    st.subheader("Chance-correlation risk")
    risk = spurious_correlation_risk(len(X), len(X_cols), n_sim=2000)
    for k, val in risk.items():
        st.metric(k, f"{val:.1%}")
    if risk["P(max|r|>=0.95)"] > 0.10:
        st.error(
            f"At n={len(X)} with {len(X_cols)} factors there is a "
            f"{risk['P(max|r|>=0.95)']:.0%} chance that a *pure noise* "
            f"factor reaches |r| >= 0.95. The bar chart on the left cannot "
            f"be read as evidence of a driver."
        )

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Collinearity (VIF)")
    vif = pd.DataFrame({"factor": list(v.vif), "VIF": list(v.vif.values())})
    vif["VIF_display"] = vif["VIF"].apply(
        lambda x: "inf (perfectly determined)" if np.isinf(x) else f"{x:.2f}")
    st.dataframe(vif[["factor", "VIF_display"]], width="stretch", hide_index=True)
    st.caption("VIF > 10 means the factor is largely predictable from the "
               "others; its individual coefficient is not interpretable.")
with c2:
    st.subheader("Correlated factor clusters")
    cl = correlation_clusters(X, threshold=0.8)
    cdf = pd.DataFrame({"factor": list(cl), "cluster": list(cl.values())})
    st.dataframe(cdf.sort_values("cluster"), width="stretch", hide_index=True)
    st.caption("Factors in the same cluster must be reported as a GROUP. "
               "Splitting importance between them is meaningless.")

if v.mixture_detected:
    st.divider()
    st.error(
        f"**Mixture constraint detected** on {v.mixture_columns} "
        f"(row-sum CV = {v.mixture_sum_cv:.2e}).\n\n"
        f"These components sum to a constant, so an ordinary regression "
        f"with an intercept is mathematically invalid - the intercept is "
        f"not identifiable. The Model Arena will use Scheffe canonical "
        f"models instead."
    )
    s = df[v.mixture_columns].sum(axis=1)
    st.plotly_chart(
        go.Figure(go.Scatter(y=s, mode="markers+lines"))
        .update_layout(title="Row sums of the constrained group",
                       yaxis_title="sum", height=260),
        width="stretch")

st.divider()
st.subheader("PCA of the design space")
if len(X) > 2 and len(X_cols) > 1:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(X)
    k = min(3, min(Z.shape))
    p = PCA(n_components=k).fit(Z)
    T = p.transform(Z)
    fig = px.scatter(x=T[:, 0], y=T[:, 1] if k > 1 else np.zeros(len(T)),
                     color=y.values, labels={"x": "PC1", "y": "PC2",
                                             "color": y_col},
                     color_continuous_scale="Viridis")
    fig.update_traces(marker=dict(size=12))
    st.plotly_chart(fig, width="stretch")
    st.caption("Explained variance: " + ", ".join(
        f"PC{i+1} {r:.1%}" for i, r in enumerate(p.explained_variance_ratio_)))
