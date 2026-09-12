import os, sys, itertools
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fi_applicability import gp_predictive_interval

st.title("🔮 Predict & Optimise")
st.caption("Every prediction carries an applicability-domain flag. Outside the "
           "training hull you get a warning, not a number.")

a = st.session_state.get("arena")
df = st.session_state.get("df"); X_cols = st.session_state.get("X_cols") or []
y_col = st.session_state.get("y_col")
if a is None or a.selected_estimator is None:
    st.warning("Run the **Model Arena** first and obtain a validated model.")
    st.stop()
if a.no_signal:
    st.error("The selected run reported NO SIGNAL. Predictions are not available.")
    st.stop()

est, ad = a.selected_estimator, a.applicability
st.success(f"Using **{a.selected.name}** (Q² = {a.selected.q2:.3f}, "
           f"RMSECV = {a.selected.rmsecv:.4g}, permutation p = {a.selected.perm_p:.4f})")

t1, t2, t3 = st.tabs(["🎯 Single prediction", "📋 Batch prediction", "🗺️ Response surface"])

X = df[X_cols]
with t1:
    st.subheader("Enter a candidate formulation")
    cols = st.columns(min(4, len(X_cols)))
    vals = {}
    for i, c in enumerate(X_cols):
        with cols[i % len(cols)]:
            lo, hi, med = float(X[c].min()), float(X[c].max()), float(X[c].median())
            span = hi - lo if hi > lo else abs(med) or 1.0
            vals[c] = st.number_input(c, lo - span, hi + span, med,
                                      step=span/100, format="%.4f")
    if st.button("Predict", type="primary"):
        xn = np.array([[vals[c] for c in X_cols]], float)
        mu, plo, phi = gp_predictive_interval(est, xn)
        chk = ad.check(xn)[0]
        c = st.columns(3)
        c[0].metric(f"Predicted {y_col}", f"{float(mu[0]):.4f}")
        if plo is not None:
            c[1].metric("95% PI low", f"{float(plo[0]):.4f}")
            c[2].metric("95% PI high", f"{float(phi[0]):.4f}")
        else:
            c[1].metric("RMSECV (±1 SD guide)", f"{a.selected.rmsecv:.4g}")
            c[2].caption("This model cannot express predictive uncertainty. "
                         "Only a GP gives a true prediction interval.")
        if chk["inside"]:
            st.success(f"✅ {chk['verdict']}")
        else:
            st.error(f"⚠️ {chk['verdict']}")
            st.warning("This is an extrapolation. The number above is what the "
                       "model says, not what the tablet will do. Run a batch here "
                       "before trusting it.")
        st.caption(f"leverage = {chk['leverage']:.3f} (h* = {chk['h_star']:.3f}), "
                   f"Hotelling T² = {chk['T2']:.2f} (critical {chk['T2_crit']:.2f})")
        if chk["out_of_range"]:
            st.dataframe(pd.DataFrame(chk["out_of_range"]).round(3),
                         width="stretch", hide_index=True)

with t2:
    st.subheader("Batch prediction from CSV")
    up = st.file_uploader("CSV with the same factor columns", type=["csv"], key="bp")
    if up is not None:
        nd = pd.read_csv(up)
        missing = [c for c in X_cols if c not in nd.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            xn = nd[X_cols].to_numpy(float)
            mu, plo, phi = gp_predictive_interval(est, xn)
            out = nd.copy(); out[f"predicted_{y_col}"] = mu
            if plo is not None:
                out["PI_low"], out["PI_high"] = plo, phi
            checks = ad.check(xn)
            out["in_domain"] = [c["inside"] for c in checks]
            out["AD_note"] = [c["verdict"] for c in checks]
            st.dataframe(out.round(4), width="stretch")
            n_out = int((~out["in_domain"]).sum())
            if n_out:
                st.error(f"{n_out} row(s) are extrapolations — treat those "
                         f"predictions as unreliable.")
            st.download_button("Download predictions",
                               out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

with t3:
    st.subheader("Response surface over two factors")
    if len(X_cols) < 2:
        st.info("Need at least two factors.")
    else:
        c1, c2 = st.columns(2)
        with c1: f1 = st.selectbox("X axis", X_cols, 0)
        with c2: f2 = st.selectbox("Y axis", [c for c in X_cols if c != f1], 0)
        others = [c for c in X_cols if c not in (f1, f2)]
        fixed = {}
        if others:
            st.caption("Held constant at:")
            oc = st.columns(min(4, len(others)))
            for i, c in enumerate(others):
                with oc[i % len(oc)]:
                    fixed[c] = st.number_input(c, value=float(X[c].median()),
                                               key=f"fx_{c}", format="%.4f")
        res = st.slider("Grid resolution", 12, 60, 30)
        if st.button("Compute surface", type="primary"):
            g1 = np.linspace(X[f1].min(), X[f1].max(), res)
            g2 = np.linspace(X[f2].min(), X[f2].max(), res)
            G1, G2 = np.meshgrid(g1, g2)
            grid = pd.DataFrame({f1: G1.ravel(), f2: G2.ravel()})
            for c in others: grid[c] = fixed[c]
            grid = grid[X_cols]
            mu, _, _ = gp_predictive_interval(est, grid.to_numpy(float))
            Z = np.asarray(mu).reshape(G1.shape)
            inside = np.array([c["inside"] for c in ad.check(grid.to_numpy(float))]).reshape(G1.shape)
            Zm = np.where(inside, Z, np.nan)
            fig = go.Figure(go.Contour(x=g1, y=g2, z=Zm, colorscale="Viridis",
                                       contours=dict(showlabels=True)))
            fig.add_trace(go.Scatter(x=X[f1], y=X[f2], mode="markers",
                                     marker=dict(size=9, color="white",
                                                 line=dict(width=1.5, color="black")),
                                     name="observed batches"))
            fig.update_layout(height=560, xaxis_title=f1, yaxis_title=f2,
                              title=f"Predicted {y_col} (blank = outside applicability domain)")
            st.plotly_chart(fig, width="stretch")
            st.caption("Blank regions are extrapolations and are deliberately "
                       "not coloured — a contour drawn there would be fiction.")
