"""
Formulation Intelligence - Streamlit entry point.

Run:  streamlit run app.py

Flat layout: every module sits in this one directory, so the whole repo
can be uploaded to GitHub in a single drag. Streamlit's `pages/`
convention needs a subfolder, so navigation is built explicitly with
st.navigation instead - same result, no directories.
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Formulation Intelligence",
                   page_icon="⚗️", layout="wide")

for _k, _v in {
    "df": None, "X_cols": [], "y_col": None, "arena": None,
    "profiles": None, "profile_times": None, "seed": 42, "group_col": None,
}.items():
    st.session_state.setdefault(_k, _v)

PAGES = [
    st.Page("page_0_load_data.py", title="Load data", icon="📁", default=True),
    st.Page("page_1_design_diagnostics.py", title="Design Diagnostics",
            icon="🔬"),
    st.Page("page_2_model_arena.py", title="Model Arena", icon="🏆"),
    st.Page("page_3_factor_identification.py", title="Factor Identification",
            icon="🎯"),
    st.Page("page_4_dissolution_modeling.py", title="Dissolution Modelling",
            icon="💊"),
    st.Page("page_5_percolation_threshold.py", title="Percolation & Threshold",
            icon="⚡"),
    st.Page("page_6_ivivc_and_virtual_be.py", title="IVIVC & Virtual BE",
            icon="🧬"),
    st.Page("page_7_predict_and_optimize.py", title="Predict & Optimise",
            icon="🔮"),
    st.Page("page_9_mixed_effects.py", title="Mixed Effects", icon="🏭"),
    st.Page("page_10_reference_scaled_be.py", title="Reference-Scaled BE",
            icon="📈"),
    st.Page("page_8_report.py", title="Report", icon="📄"),
]

with st.sidebar:
    st.header("Session")
    st.session_state.seed = st.number_input("Random seed", 0, 10**6,
                                            st.session_state.seed)
    st.caption("Every result in this app is reproducible from this seed.")
    if st.session_state.df is not None:
        st.success(f"Loaded: {st.session_state.df.shape[0]} rows x "
                   f"{st.session_state.df.shape[1]} cols")
        if st.session_state.y_col:
            st.info(f"Response: **{st.session_state.y_col}**")
            st.info(f"Factors: {len(st.session_state.X_cols)}")
        if st.session_state.group_col:
            st.info(f"Grouping: **{st.session_state.group_col}**")

st.navigation(PAGES).run()
