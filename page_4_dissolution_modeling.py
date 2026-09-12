import os, sys
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fi_dissolution as D

st.title("💊 Dissolution Modelling")
st.caption("Never model timepoints independently. Parameterise, then predict.")

t1, t2, t3, t4 = st.tabs(["📥 Load profiles", "📐 Parameterise", "🔍 Discrimination power", "📊 f2 with CI"])

with t1:
    st.markdown("Upload a CSV: **rows = formulations, columns = timepoints**. "
                "A first text column is used as the batch id.")
    up = st.file_uploader("Profile CSV", type=["csv"], key="prof_up")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Load demo profiles"):
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "demo_dissolution_profiles.csv")
            if os.path.exists(p):
                d = pd.read_csv(p)
                st.session_state.profiles = d.set_index(d.columns[0])
                st.session_state.profile_times = [
                    float(c.replace("t_", "").replace("h", ""))
                    for c in st.session_state.profiles.columns]
                st.success("Demo profiles loaded")
            else:
                st.error("Run: python simulate_data.py")
    if up is not None:
        d = pd.read_csv(up)
        idx = d.columns[0]
        st.session_state.profiles = d.set_index(idx) if d[idx].dtype == object else d
        try:
            st.session_state.profile_times = [
                float("".join(ch for ch in c if ch.isdigit() or ch == "."))
                for c in st.session_state.profiles.columns]
        except Exception:
            st.session_state.profile_times = list(range(st.session_state.profiles.shape[1]))
        st.success(f"Loaded {up.name}")

    P = st.session_state.get("profiles")
    if P is not None:
        st.dataframe(P.head(10), width="stretch")
        t = np.array(st.session_state.profile_times, float)
        fig = go.Figure()
        for i, (b, row) in enumerate(P.iterrows()):
            if i > 30: break
            fig.add_trace(go.Scatter(x=t, y=row.to_numpy(float), mode="lines+markers", name=str(b)))
        fig.update_layout(height=430, xaxis_title="time (h)", yaxis_title="% released",
                          showlegend=P.shape[0] <= 12)
        st.plotly_chart(fig, width="stretch")

with t2:
    P = st.session_state.get("profiles")
    if P is None:
        st.info("Load profiles first.")
    else:
        t = np.array(st.session_state.profile_times, float)
        model = st.selectbox("Release model", list(D.MODEL_LIB), index=0)
        st.caption("Fitting 2-3 parameters per formulation collapses the profile "
                   "into response variables the Model Arena can handle, preserves "
                   "monotonicity, and needs far less data than modelling each "
                   "timepoint.")
        if st.button("Fit all profiles", type="primary"):
            with st.spinner("Fitting..."):
                st.session_state.diss_params = D.parameterise_batch(P, t, model)
                st.session_state.diss_model = model
        pr = st.session_state.get("diss_params")
        if pr is not None:
            st.subheader(f"{st.session_state.diss_model} parameters")
            st.dataframe(pr.round(4), width="stretch")
            st.metric("Median fit R²", f"{pr['R2'].median():.4f}")
            bad = pr[pr["R2"] < 0.95]
            if len(bad): st.warning(f"{len(bad)} profile(s) fit poorly (R² < 0.95).")
            csv = pr.reset_index().to_csv(index=False).encode()
            st.download_button("Download parameters (use as response in the Arena)",
                               csv, "dissolution_parameters.csv", "text/csv")
            st.divider(); st.subheader("Single-profile model ranking")
            b = st.selectbox("Formulation", list(P.index))
            rank = D.best_model(t, P.loc[b].to_numpy(float))
            st.dataframe(rank.round(4), width="stretch", hide_index=True)
            st.caption("Ranked by AIC. For Korsmeyer-Peppas the exponent n indicates "
                       "the mechanism: ~0.45 Fickian, 0.45-0.89 anomalous, ~0.89 Case-II.")
            st.divider(); st.subheader("Functional PCA (alternative route)")
            sc, pca, evr = D.functional_pca(P, 3)
            st.dataframe(sc.round(3).head(10), width="stretch")
            st.caption("Explained variance: " + ", ".join(f"PC{i+1} {r:.1%}" for i, r in enumerate(evr)))

with t3:
    st.subheader("Can this method see a difference you know exists?")
    st.markdown("""
If batches differ in vivo but the dissolution profiles are superimposable,
the method is non-discriminating. **Change the method, not the model.**
Enter the batches ordered *fastest to slowest in vivo*.
""")
    P = st.session_state.get("profiles")
    if P is None:
        st.info("Load profiles first.")
    else:
        order = st.multiselect("In vivo rank order (fastest first)", list(P.index),
                               default=list(P.index)[:3])
        reps = st.number_input("Vessels per batch (if profiles are means, use 1)", 1, 24, 1)
        if len(order) >= 3 and st.button("Assess discrimination", type="primary"):
            t = np.array(st.session_state.profile_times, float)
            prof = {b: np.atleast_2d(P.loc[b].to_numpy(float)) for b in order}
            r = D.discrimination_power(prof, order, t)
            if "error" in r:
                st.error(r["error"])
            else:
                c = st.columns(4)
                c[0].metric("Spearman ρ vs in vivo", f"{r['spearman_rho_vs_invivo']:.2f}")
                c[1].metric("Max spread", f"{r['max_spread_pct']:.1f}%")
                c[2].metric("Best timepoint", f"{r['best_timepoint']:.2g} h")
                c[3].metric("f2 extremes", f"{r['f2_extremes']:.1f}")
                (st.success if r["discriminating"] else st.error)(r["verdict"])
                st.dataframe(r["per_timepoint"].round(3), width="stretch", hide_index=True)
                if not r["discriminating"]:
                    st.info("""
**Next moves, in order:**
1. **Ionic strength series** (50/100/150/300 mM) and counterion (chloride vs
   phosphate). For quaternary-ammonium acrylics permeability is driven by
   counterion exchange, essentially independent of pH — a pH range at constant
   ionic strength tests the one axis the polymer ignores.
2. **Early sampling** at 5/10/15/20/30/45/60 min. Cmax is set in the first
   1–2 h; if the first sample is at 1 h a burst is simply not in the dataset.
3. **Mechanical stress** — USP 3 at 20–30 dpm, or a stress-test device with
   pressure waves. Note USP 2 and USP 3 both vary agitation, so if USP 2
   failed, USP 3 tests the same axis twice.
4. **Vessel-to-vessel %RSD**, not the mean profile. A matrix near its
   percolation threshold has an unreliable pore network and shows elevated
   scatter even when mean profiles superimpose. f2 on means discards this.
5. If dissolution still cannot separate them, measure the pore network
   directly: hydrated-tablet conductivity, gravimetric water uptake, porosimetry.
""")

with t4:
    st.subheader("f2 similarity with a confidence interval")
    st.warning("The f2 point estimate is statistically unstable. Report the "
               "lower bound of the 90% CI.")
    P = st.session_state.get("profiles")
    if P is None or len(P) < 2:
        st.info("Load at least two profiles.")
    else:
        c1, c2 = st.columns(2)
        with c1: ref = st.selectbox("Reference", list(P.index), 0)
        with c2: tst = st.selectbox("Test", list(P.index), min(1, len(P)-1))
        cv = st.slider("Assumed vessel %RSD (for bootstrap)", 1.0, 15.0, 4.0, 0.5)
        if st.button("Bootstrap f2", type="primary"):
            rng = np.random.default_rng(st.session_state.get("seed", 42))
            R = np.array([P.loc[ref].to_numpy(float) * (1 + rng.normal(0, cv/100, P.shape[1])) for _ in range(12)])
            T = np.array([P.loc[tst].to_numpy(float) * (1 + rng.normal(0, cv/100, P.shape[1])) for _ in range(12)])
            r = D.bootstrap_f2(R, T, seed=st.session_state.get("seed", 42))
            c = st.columns(3)
            c[0].metric("f2 point estimate", f"{r['f2_point']:.1f}")
            c[1].metric("Lower 90% bound", f"{r['f2_lower_90CI']:.1f}")
            c[2].metric("Upper 90% bound", f"{r['f2_upper_90CI']:.1f}")
            (st.success if r["similar_by_lower_bound"] else st.error)(r["verdict"])
