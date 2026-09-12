import os, sys, json, datetime, platform
import numpy as np, pandas as pd, streamlit as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fi_design_diagnostics import diagnose

st.title("📄 Reproducible Report")
st.caption("Seeds, package versions and every gate decision. If this ever "
           "supports a filing, reproducibility is the first question asked.")

df = st.session_state.get("df"); X_cols = st.session_state.get("X_cols") or []
y_col = st.session_state.get("y_col"); a = st.session_state.get("arena")
if df is None:
    st.warning("Load data first."); st.stop()

import sklearn, scipy
env = {"generated": datetime.datetime.now().isoformat(timespec="seconds"),
       "python": platform.python_version(), "numpy": np.__version__,
       "pandas": pd.__version__, "scipy": scipy.__version__,
       "scikit-learn": sklearn.__version__,
       "seed": st.session_state.get("seed", 42)}

lines = ["# Formulation Intelligence — Analysis Report", "",
         "## Environment", ""]
lines += [f"- **{k}**: {v}" for k, v in env.items()]
lines += ["", "## Dataset", "",
          f"- Rows: {df.shape[0]}", f"- Response: {y_col}",
          f"- Factors: {', '.join(X_cols) if X_cols else 'none selected'}", ""]

if X_cols and y_col:
    v = diagnose(df[X_cols], df[y_col])
    lines += ["## Design gate", "", "```", v.summary(), "```", ""]

if a is not None:
    lines += ["## Model selection", "", "```", a.explain_selection(), "```", "",
              f"- Regime: {a.regime}", f"- CV scheme: {a.cv_scheme}", ""]
    if not a.leaderboard.empty:
        lines += ["### Leaderboard", "", a.leaderboard.to_markdown(index=False), ""]
    if a.messages:
        lines += ["### Run notes", ""] + [f"- {m}" for m in a.messages] + [""]

fit = st.session_state.get("lmm")
if fit is not None:
    lines += ["## Mixed-effects analysis", "", "```", fit.summary(), "```", ""]
    lrt = st.session_state.get("lmm_lrt")
    if lrt:
        lines += [f"- LRT for the random effect: statistic="
                  f"{lrt['lr_statistic']:.3f}, p={lrt['p_value']:.4f} "
                  f"({lrt['reference']})", f"- {lrt['verdict']}", ""]

lines += ["## Interpretation rules applied", "",
          "- Models ranked on cross-validated Q², never training R².",
          "- LOOCV below n=30; repeated 5-fold ×10 above.",
          "- 1-SE rule: simplest model within one standard error of the best Q².",
          "- Y-randomisation: models not separating from their permuted null "
          "are excluded regardless of Q².",
          "- Applicability domain (leverage + Hotelling T²) flags every "
          "prediction outside the training hull.",
          "- Mixture-constrained components modelled with Scheffé canonical "
          "polynomials without an intercept.",
          "- Batch/site structure modelled as a random intercept (REML); the "
          "design gate tests the effective sample size, not the row count.",
          "- Reference-scaled BE (FDA RSABE via Howe's bound, EMA ABEL) "
          "available for highly variable drugs; both retain the 80-125% "
          "point-estimate constraint.", ""]

md = "\n".join(lines)
st.markdown(md)
c1, c2 = st.columns(2)
c1.download_button("Download report (Markdown)", md.encode(),
                   "formulation_report.md", "text/markdown")
c2.download_button("Download environment (JSON)",
                   json.dumps(env, indent=2).encode(),
                   "environment.json", "application/json")
