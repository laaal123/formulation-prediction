import os, sys
import numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fi_percolation as P

st.title("⚡ Percolation & Threshold Analysis")
st.caption("Mechanistic mode for small n. Output is an envelope, never a point prediction.")

st.info("""
With 3–6 batches and a monotonic driver you cannot fit a response surface,
but you **can** locate a threshold and bound the response between tested
levels. Near a critical volume fraction, transport follows k ~ (ε−εc)^μ —
either side of εc the behaviour is qualitatively different, so interpolating
across it with a smooth model is the classic small-n error.
""")

df = st.session_state.get("df")
if df is None:
    st.warning("Load data on the main page first."); st.stop()

num = df.select_dtypes(include=[np.number]).columns.tolist()
c1, c2 = st.columns(2)
with c1: xcol = st.selectbox("Driver (e.g. polymer %)", num)
with c2: ycol = st.selectbox("Response (e.g. Cmax GMR)", num,
                             index=min(len(num)-1, 1))

x = df[xcol].to_numpy(float); y = df[ycol].to_numpy(float)
if len(x) < 3:
    st.error("Need at least 3 levels."); st.stop()

st.divider(); st.subheader("Log-slope between consecutive levels")
th = P.detect_threshold(x, y)
st.dataframe(th["slope_table"].round(5), width="stretch", hide_index=True)
(st.error if th["threshold_detected"] else st.success)(th["verdict"])
st.caption("A smooth dose-response gives similar slopes. A large jump is a "
           "threshold — the single most informative statistic available at n=3.")

o = np.argsort(x)
fig = go.Figure()
fig.add_hrect(y0=0.80, y1=1.25, fillcolor="green", opacity=0.10,
              annotation_text="BE window", line_width=0)
if th["threshold_detected"]:
    fig.add_vrect(x0=th["interval_low"], x1=th["interval_high"],
                  fillcolor="red", opacity=0.13, line_width=0,
                  annotation_text="threshold interval")
fig.add_trace(go.Scatter(x=x[o], y=y[o], mode="lines+markers+text",
                         text=[f"{v:.3f}" for v in y[o]], textposition="top center",
                         marker=dict(size=13), name="observed"))
fig.update_layout(height=430, xaxis_title=xcol, yaxis_title=ycol)
st.plotly_chart(fig, width="stretch")

st.divider(); st.subheader("Percolation fit (μ fixed)")
mu = st.slider("Exponent μ", 1.5, 2.2, 1.9, 0.05)
pf = P.percolation_fit(x, y, mu)
if "error" in pf:
    st.warning(pf["error"])
else:
    c = st.columns(4)
    c[0].metric("Critical value εc", f"{pf['x_critical']:.3f}")
    c[1].metric("εc std. error", f"{pf['x_critical_se']:.3f}" if np.isfinite(pf['x_critical_se']) else "n/a")
    c[2].metric("R²", f"{pf['r2']:.4f}")
    c[3].metric("Residual d.o.f.", pf["residual_dof"])
    st.caption(pf["note"])

st.divider(); st.subheader("Prediction envelope at an untested level")
xt = st.slider("Target level", float(np.min(x)), float(np.max(x)),
               float(np.mean(x)), step=float((np.max(x)-np.min(x))/100))
env = P.prediction_envelope(x, y, xt)
if env.get("extrapolation"):
    st.error(env["note"])
else:
    c = st.columns(4)
    c[0].metric("Smooth interpolation", f"{env['smooth_estimate']:.3f}")
    c[1].metric("Envelope low", f"{env['envelope_low']:.3f}")
    c[2].metric("Envelope high", f"{env['envelope_high']:.3f}")
    c[3].metric("Width", f"{env['envelope_width']:.3f}")
    st.warning(env["note"])
    risk = P.be_risk_from_envelope(env)
    (st.success if risk["fully_inside_BE"] and not risk["too_wide"] else st.error)(risk["decision"])
    fig = go.Figure()
    fig.add_hrect(y0=0.80, y1=1.25, fillcolor="green", opacity=0.10, line_width=0)
    fig.add_trace(go.Scatter(x=x[o], y=y[o], mode="lines+markers",
                             marker=dict(size=12), name="observed"))
    fig.add_trace(go.Scatter(x=[xt, xt], y=[env["envelope_low"], env["envelope_high"]],
                             mode="lines+markers", line=dict(width=9, color="orange"),
                             name="honest envelope"))
    fig.add_trace(go.Scatter(x=[xt], y=[env["smooth_estimate"]], mode="markers",
                             marker=dict(size=16, symbol="x", color="red"),
                             name="smooth estimate (misleading)"))
    fig.update_layout(height=430, xaxis_title=xcol, yaxis_title=ycol)
    st.plotly_chart(fig, width="stretch")
