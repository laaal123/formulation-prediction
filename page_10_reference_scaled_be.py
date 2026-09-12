import os, sys
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fi_rsabe import (DESIGNS, CV_SCALING_THRESHOLD, CV_CAP_EMA, THETA_FDA,
                        ema_expanded_limits, simulate_rsabe, compare_frameworks,
                        rsabe_sample_size, type_i_error_check)
from fi_be_simulation import cv_to_sigma

st.title("📈 Reference-Scaled BE for highly variable drugs")
st.caption("FDA RSABE and EMA ABEL. Both need a replicate design.")

st.info("""
Unscaled ABE is close to unpassable once the within-subject CV of the reference
exceeds about 30% — the study needs enormous n even when the products are
genuinely identical. Both regulators allow the acceptance range to be scaled to
the reference variability, but **only with a replicate design**, because CVwR
has to be estimated from repeated administrations of the reference.
""")

t1, t2, t3, t4 = st.tabs(["📏 Acceptance limits", "🎲 Simulate a study",
                          "⚖️ Framework comparison", "🔬 Type I error"])

with t1:
    st.subheader("How the limits widen with CVwR")
    cv = st.slider("Within-subject CV of the reference (%)", 10.0, 70.0, 45.0, 1.0)
    L, U, note = ema_expanded_limits(cv)
    c = st.columns(4)
    c[0].metric("EMA lower limit", f"{L:.4f}")
    c[1].metric("EMA upper limit", f"{U:.4f}")
    c[2].metric("swR", f"{cv_to_sigma(cv):.4f}")
    c[3].metric("FDA θ", f"{THETA_FDA:.4f}")
    st.caption(f"Status: {note}. EMA applies widening to Cmax only, and caps "
               f"it at CVwR = {CV_CAP_EMA:.0f}% (0.6984–1.4319). FDA scales "
               f"the criterion rather than the limits, from CVwR ≥ "
               f"{CV_SCALING_THRESHOLD:.0f}%.")
    grid = np.linspace(10, 70, 200)
    lims = np.array([ema_expanded_limits(v)[:2] for v in grid])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=lims[:, 1], mode="lines", name="upper limit"))
    fig.add_trace(go.Scatter(x=grid, y=lims[:, 0], mode="lines", name="lower limit",
                             fill="tonexty", fillcolor="rgba(76,175,137,0.15)"))
    fig.add_vline(x=CV_SCALING_THRESHOLD, line_dash="dash",
                  annotation_text="scaling starts")
    fig.add_vline(x=CV_CAP_EMA, line_dash="dash", annotation_text="EMA cap")
    fig.add_vline(x=cv, line_color="orange")
    fig.update_layout(height=430, xaxis_title="CVwR (%)", yaxis_title="GMR limit")
    st.plotly_chart(fig, width="stretch")

with t2:
    st.subheader("Monte Carlo a replicate-design study")
    c = st.columns(4)
    gmr = c[0].number_input("True GMR", 0.60, 1.70, 1.00, 0.01)
    cvwr = c[1].number_input("True CVwR (%)", 10.0, 80.0, 45.0, 1.0)
    n = c[2].number_input("Subjects", 12, 200, 36, 6)
    method = c[3].selectbox("Framework", ["FDA", "EMA", "ABE"])
    c = st.columns(3)
    design = c[0].selectbox("Design", list(DESIGNS),
                            format_func=lambda k: DESIGNS[k]["label"])
    cvwt = c[1].number_input("CVwT (%) — blank uses CVwR", 0.0, 80.0, 0.0, 1.0)
    nsim = c[2].select_slider("Simulations", [1000, 3000, 5000, 10000], 3000)

    if st.button("Simulate", type="primary"):
        if DESIGNS[design]["n_R"] < 2:
            st.error("Reference-scaling needs at least two reference periods.")
        else:
            with st.spinner("Simulating…"):
                r = simulate_rsabe(gmr, cvwr, int(n),
                                   cv_wt_pct=cvwt if cvwt > 0 else None,
                                   design=design, method=method, n_sim=int(nsim),
                                   seed=st.session_state.get("seed", 42))
            c = st.columns(4)
            c[0].metric("P(pass)", f"{r['p_pass']:.1%}")
            c[1].metric("Scaling applied", f"{r['p_scaling_applied']:.0%}")
            c[2].metric("Point-estimate constraint met", f"{r['p_point_estimate_ok']:.0%}")
            c[3].metric("Median observed CVwR", f"{r['median_observed_CVwR']:.1f}%")
            (st.success if r["p_pass"] >= 0.80 else st.error)(r["verdict"])
            if r["p_scaling_applied"] < 0.95 and cvwr > CV_SCALING_THRESHOLD:
                st.warning(
                    f"Scaling was only permitted in {r['p_scaling_applied']:.0%} "
                    f"of simulated studies. With a true CVwR this close to the "
                    f"30% threshold, some studies will observe a CVwR below it "
                    f"and lose the widened limits entirely — a real and often "
                    f"overlooked failure mode.")
            ss = rsabe_sample_size(gmr, cvwr, 0.80, method=method,
                                   design=design, n_sim=1200,
                                   seed=st.session_state.get("seed", 42))
            st.info(ss["note"])

with t3:
    st.subheader("Same data, three decision rules")
    c = st.columns(3)
    g2 = c[0].number_input("True GMR ", 0.60, 1.70, 1.00, 0.01, key="g2")
    cv2 = c[1].number_input("True CVwR (%) ", 10.0, 80.0, 45.0, 1.0, key="cv2")
    n2 = c[2].number_input("Subjects ", 12, 200, 36, 6, key="n2")
    if st.button("Compare frameworks", type="primary"):
        with st.spinner("Simulating three frameworks…"):
            cmp = compare_frameworks(g2, cv2, int(n2), n_sim=3000,
                                     seed=st.session_state.get("seed", 42))
        st.dataframe(cmp, width="stretch", hide_index=True)
        fig = go.Figure(go.Bar(x=cmp["framework"], y=cmp["p_pass"]))
        fig.add_hline(y=0.80, line_dash="dash", line_color="red",
                      annotation_text="80% power")
        fig.update_layout(height=380, yaxis_title="P(pass)", yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
        st.caption("ABE here is evaluated on the **same replicate data**, so "
                   "this isolates the decision rule. It is not the power of a "
                   "2×2×2 crossover, which would be lower again.")

with t4:
    st.subheader("Is the procedure correctly calibrated?")
    st.markdown("""
The GMR is placed exactly on the scaled boundary, where a correctly built
procedure rejects about 5% of the time. This is the check that catches a
mis-specified confidence bound — it shows up here and nowhere else.
""")
    c = st.columns(3)
    cv3 = c[0].number_input("CVwR (%)  ", 31.0, 80.0, 45.0, 1.0, key="cv3")
    n3 = c[1].number_input("Subjects  ", 12, 200, 48, 6, key="n3")
    pe = c[2].checkbox("Apply point-estimate constraint", value=False,
                       help="Off isolates the confidence bound. A real study "
                            "always has it on, but with it on the constraint "
                            "binds first and the observed rate falls far "
                            "below nominal.")
    if st.button("Run calibration check", type="primary"):
        with st.spinner("Running 20,000 simulations…"):
            rows = [type_i_error_check(cv3, int(n3), method=m, n_sim=20000,
                                       seed=st.session_state.get("seed", 42),
                                       apply_pe_constraint=pe)
                    for m in ("FDA", "EMA")]
        st.dataframe(pd.DataFrame([
            {"method": r["method"], "boundary GMR": round(r["boundary_gmr"], 4),
             "empirical α": round(r["empirical_type_I"], 4),
             "nominal": 0.05, "controlled": r["controlled"]} for r in rows]),
            width="stretch", hide_index=True)
        for r in rows:
            st.caption(f"**{r['method']}** — {r['note']}")
