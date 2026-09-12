import os, sys
import numpy as np, pandas as pd, streamlit as st
import plotly.express as px, plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fi_ivivc as IV
import fi_be_simulation as BE

st.title("🧬 IVIVC & Virtual Bioequivalence")
st.caption("Deconvolute → Level A → convolve → Monte Carlo the study. "
           "A probability with a mechanistic chain behind it.")

t1, t2, t3, t4 = st.tabs(["🔄 Reference drift check", "📉 Deconvolution",
                          "🔗 Level A", "🎲 Virtual BE"])

with t1:
    st.subheader("Are your BE studies comparable?")
    st.error("""
Comparing test products **across separate BE studies** is confounded with the
reference arm, the subject panel and the RLD lot. If one study's reference ran
high, that study's GMR is biased low — and a formulation conclusion drawn from
the comparison is wrong. Check this before any dose-response reading.
""")
    st.markdown("Paste or edit study results:")
    default = pd.DataFrame({"study": ["A", "B", "C"], "level": [25.0, 28.0, 30.0],
                            "Cmax_T": [4128.08, 3958.52, 3120.43],
                            "Cmax_R": [3291.65, 3300.65, 3785.74]})
    ed = st.data_editor(default, num_rows="dynamic", width="stretch")
    if st.button("Check reference drift", type="primary"):
        r = BE.normalise_across_studies(ed)
        st.dataframe(r.round(4), width="stretch", hide_index=True)
        ref = r.attrs["common_reference"]
        st.metric("Common reference used (median)", f"{ref:.1f}")
        sus = r[r["reference_suspect"]]
        if len(sus):
            st.error(f"**{len(sus)} study/studies have a reference arm more than "
                     f"10% from the median.** Re-check the RLD lot number, subject "
                     f"panel and clinical site before reading a dose-response.")
        else:
            st.success("Reference arms are consistent across studies.")
        fig = go.Figure()
        fig.add_hrect(y0=0.80, y1=1.25, fillcolor="green", opacity=0.10, line_width=0)
        fig.add_trace(go.Scatter(x=r["level"], y=r["GMR_reported"],
                                 mode="lines+markers", name="as reported", marker=dict(size=12)))
        fig.add_trace(go.Scatter(x=r["level"], y=r["GMR_common_ref"],
                                 mode="lines+markers", name="vs common reference", marker=dict(size=12)))
        fig.update_layout(height=400, xaxis_title="level", yaxis_title="Cmax GMR")
        st.plotly_chart(fig, width="stretch")

with t2:
    st.subheader("In vivo fraction absorbed")
    method = st.radio("Method", ["Wagner-Nelson (1-compartment)",
                                 "Loo-Riegelman (2-compartment)"], horizontal=True)
    up = st.file_uploader("Plasma profile CSV (time, Cp)", type=["csv"], key="pk_up")
    if up is not None:
        d = pd.read_csv(up); t = d.iloc[:, 0].to_numpy(float); cp = d.iloc[:, 1].to_numpy(float)
    else:
        st.caption("No file — using a simulated one-compartment profile.")
        t = np.linspace(0, 36, 145)
        fa = np.clip(1 - np.exp(-(t/12)**1.2), 0, 1)
        cp = IV.convolve_to_pk(t, fa, 100, 0.85, 22, 0.14).Cp.to_numpy()

    if "Wagner" in method:
        ke = st.number_input("ke (1/h)", 0.001, 5.0, 0.14, 0.01, format="%.4f")
        fa_est = IV.wagner_nelson(t, cp, ke)
    else:
        c = st.columns(3)
        k10 = c[0].number_input("k10", 0.001, 5.0, 0.20, 0.01, format="%.4f")
        k12 = c[1].number_input("k12", 0.001, 5.0, 0.30, 0.01, format="%.4f")
        k21 = c[2].number_input("k21", 0.001, 5.0, 0.15, 0.01, format="%.4f")
        fa_est = IV.loo_riegelman(t, cp, k10, k12, k21)
    st.session_state.fa_est = fa_est; st.session_state.pk_t = t

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(x=t, y=cp, labels={"x":"time (h)","y":"Cp"})
                        .update_layout(title="Plasma profile", height=360), width="stretch")
    with c2:
        st.plotly_chart(px.line(x=t, y=fa_est, labels={"x":"time (h)","y":"fraction absorbed"})
                        .update_layout(title="Deconvoluted absorption", height=360), width="stretch")
    m = IV.pk_metrics(t, cp)
    c = st.columns(4)
    for i, k in enumerate(["Cmax","Tmax","AUC_0_t","AUC_0_inf"]):
        c[i].metric(k, f"{m[k]:.3f}")

with t3:
    st.subheader("Level A correlation")
    st.caption("Fraction absorbed vs fraction dissolved. Slope near 1 and "
               "intercept near 0 is the target; a slope far from 1 means the in "
               "vitro method runs at the wrong rate and needs time scaling.")
    n_pts = st.slider("Paired points", 3, 20, 8)
    c1, c2 = st.columns(2)
    with c1:
        fd = st.text_area("Fraction dissolved (comma separated)",
                          ", ".join(f"{v:.3f}" for v in np.linspace(0.05, 0.95, n_pts)))
    with c2:
        fa = st.text_area("Fraction absorbed (comma separated)",
                          ", ".join(f"{v:.3f}" for v in np.linspace(0.04, 0.93, n_pts)))
    if st.button("Fit Level A", type="primary"):
        try:
            X = np.array([float(v) for v in fd.split(",")])
            Y = np.array([float(v) for v in fa.split(",")])
            r = IV.level_a(X, Y, time_scale=True)
            if "error" in r: st.error(r["error"])
            else:
                c = st.columns(4)
                c[0].metric("Slope", f"{r['slope']:.3f}")
                c[1].metric("Intercept", f"{r['intercept']:.3f}")
                c[2].metric("R²", f"{r['r2']:.4f}")
                c[3].metric("RMSE", f"{r['rmse']:.4f}")
                (st.success if r["linear_acceptable"] else st.warning)(r["verdict"])
                fig = px.scatter(x=X, y=Y, labels={"x":"fraction dissolved","y":"fraction absorbed"})
                xs = np.linspace(0, 1, 50)
                fig.add_trace(go.Scatter(x=xs, y=r["intercept"]+r["slope"]*xs,
                                         mode="lines", name="fit"))
                fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="identity",
                                         line=dict(dash="dash", color="grey")))
                fig.update_traces(marker=dict(size=11), selector=dict(mode="markers"))
                st.plotly_chart(fig.update_layout(height=420), width="stretch")
        except Exception as e:
            st.error(f"Could not parse: {e}")

with t4:
    st.subheader("Monte Carlo the crossover study")
    c = st.columns(4)
    gmr = c[0].number_input("Predicted true GMR", 0.50, 2.00, 0.95, 0.01)
    cv  = c[1].number_input("Intra-subject CV (%)", 5.0, 80.0, 25.0, 1.0)
    n   = c[2].number_input("Subjects", 6, 300, 36, 2)
    des = c[3].selectbox("Design", ["2x2x2", "replicate"])
    gsd = st.slider("Model uncertainty in the GMR (SD)", 0.0, 0.30, 0.05, 0.01,
                    help="0 = treat the GMR as exactly known. Any model-derived "
                         "GMR has uncertainty; propagating it is the honest option.")

    if st.button("Simulate", type="primary"):
        r = BE.simulate_be(gmr, cv, int(n), n_sim=6000, design=des,
                           seed=st.session_state.get("seed", 42))
        c = st.columns(4)
        c[0].metric("P(pass BE)", f"{r['p_pass']:.1%}")
        c[1].metric("Median 90% CI low", f"{r['median_CI_lower']:.3f}")
        c[2].metric("Median 90% CI high", f"{r['median_CI_upper']:.3f}")
        c[3].metric("P(fail low / high)", f"{r['p_fail_low']:.0%} / {r['p_fail_high']:.0%}")
        (st.success if r["p_pass"] >= 0.80 else st.error)(r["verdict"])

        if gsd > 0:
            pr = BE.propagate_prediction_uncertainty(gmr, gsd, cv, int(n),
                                                     seed=st.session_state.get("seed", 42))
            st.divider(); st.subheader("With model uncertainty propagated")
            c = st.columns(3)
            c[0].metric("Mean P(pass)", f"{pr['p_pass_mean']:.1%}")
            c[1].metric("90% range low", f"{pr['p_pass_lo90']:.1%}")
            c[2].metric("90% range high", f"{pr['p_pass_hi90']:.1%}")
            st.info(pr["note"])

        st.divider(); st.subheader("Operating window")
        w = BE.gmr_operating_window(cv, int(n), n_sim=1200,
                                    seed=st.session_state.get("seed", 42))
        st.warning(w["note"])
        fig = go.Figure(go.Scatter(x=w["curve"]["true_gmr"], y=w["curve"]["power"], mode="lines"))
        fig.add_hline(y=0.80, line_dash="dash", line_color="red", annotation_text="80% power")
        if w["gmr_low"]:
            fig.add_vrect(x0=w["gmr_low"], x1=w["gmr_high"], fillcolor="green",
                          opacity=0.15, line_width=0)
        fig.update_layout(height=400, xaxis_title="true GMR", yaxis_title="power")
        st.plotly_chart(fig, width="stretch")
        st.caption("This green band — not 0.80–1.25 — is your real formulation target.")

        ss = BE.sample_size_for_power(gmr, cv, 0.80, n_sim=1500,
                                      seed=st.session_state.get("seed", 42))
        st.info(ss["note"])
