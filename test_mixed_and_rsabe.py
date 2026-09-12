"""Unit tests for mixed-effects models and reference-scaled BE."""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pytest
from fi_mixed_models import (fit_random_intercept, lrt_random_effect,
                               compare_pooled_vs_mixed, effective_sample_size)
from fi_rsabe import (ema_expanded_limits, fda_rsabe_bound, simulate_rsabe,
                        type_i_error_check, THETA_FDA, CV_CAP_EMA)
from fi_be_simulation import cv_to_sigma
from fi_design_diagnostics import diagnose
import pandas as pd


def _grouped(s2u, s2e, q=30, m=6, seed=0):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(q), m)
    u = rng.normal(0, np.sqrt(s2u), q) if s2u > 0 else np.zeros(q)
    X = rng.normal(size=(q * m, 2))
    y = 1 + 2 * X[:, 0] - X[:, 1] + u[groups] + rng.normal(0, np.sqrt(s2e), q * m)
    return X, y, groups


# ---------------- mixed models ----------------
def test_icc_recovered():
    X, y, g = _grouped(0.6, 0.4, q=50)
    assert fit_random_intercept(X, y, g).icc == pytest.approx(0.6, abs=0.10)


def test_icc_zero_when_no_structure():
    X, y, g = _grouped(0.0, 1.0, q=50)
    assert fit_random_intercept(X, y, g).icc < 0.08


def test_fixed_effects_recovered():
    X, y, g = _grouped(0.6, 0.4, q=50)
    b = fit_random_intercept(X, y, g, ["a", "b"]).beta
    assert b[1] == pytest.approx(2.0, abs=0.15)
    assert b[2] == pytest.approx(-1.0, abs=0.15)


def test_design_effect_and_effective_n():
    X, y, g = _grouped(0.6, 0.4, q=50)
    f = fit_random_intercept(X, y, g)
    assert f.design_effect > 1.0
    assert f.effective_n < f.n_obs


def test_effective_n_equals_rows_when_icc_zero():
    X, y, g = _grouped(0.0, 1.0, q=50)
    f = fit_random_intercept(X, y, g)
    assert f.effective_n > 0.75 * f.n_obs


def test_lrt_detects_and_rejects():
    X, y, g = _grouped(0.6, 0.4, q=40)
    assert lrt_random_effect(X, y, g)["p_value"] < 0.01
    X0, y0, g0 = _grouped(0.0, 1.0, q=40)
    assert lrt_random_effect(X0, y0, g0)["p_value"] > 0.05


def test_lrt_uses_boundary_mixture():
    X, y, g = _grouped(0.4, 0.6, q=30)
    assert "mixture" in lrt_random_effect(X, y, g)["reference"]


def test_pooled_vs_mixed_returns_ratio():
    X, y, g = _grouped(0.6, 0.4, q=40)
    c = compare_pooled_vs_mixed(X, y, g, ["a", "b"])
    assert "se_ratio_LMM_over_OLS" in c["table"].columns
    assert np.isfinite(c["max_se_inflation"])


def test_effective_sample_size_conditioning_matters():
    X, y, g = _grouped(0.6, 0.4, q=40)
    uncond = effective_sample_size(y, g)["icc"]
    cond = effective_sample_size(y, g, X=X)["icc"]
    assert cond > uncond          # covariates absorb within-group variance


def test_gate_uses_effective_n():
    rng = np.random.default_rng(1)
    q, m = 6, 8
    g = np.repeat(np.arange(q), m)
    u = rng.normal(0, 1.6, q)
    X = pd.DataFrame(rng.normal(size=(q * m, 3)), columns=list("abc"))
    y = pd.Series(2 * X["a"] + u[g] + rng.normal(0, 0.5, q * m))
    assert diagnose(X, y).admissible                 # pooled: looks fine
    v = diagnose(X, y, groups=g)
    assert not v.admissible                          # grouped: blocked
    assert v.effective_n < v.n_samples


def test_single_observation_groups_warn():
    X, y, g = _grouped(0.5, 0.5, q=20, m=1)
    assert fit_random_intercept(X, y, g).warnings


# ---------------- reference-scaled BE ----------------
def test_fda_theta_constant():
    assert THETA_FDA == pytest.approx((np.log(1.25) / 0.25) ** 2)
    assert THETA_FDA == pytest.approx(0.7967, abs=1e-3)


def test_ema_no_scaling_below_threshold():
    assert ema_expanded_limits(25.0)[:2] == (0.80, 1.25)
    assert ema_expanded_limits(30.0)[:2] == (0.80, 1.25)


def test_ema_limits_at_cap():
    L, U, _ = ema_expanded_limits(CV_CAP_EMA)
    assert U == pytest.approx(1.4319, abs=1e-3)
    assert L == pytest.approx(0.6984, abs=1e-3)


def test_ema_limits_capped_above_50():
    assert ema_expanded_limits(70.0)[1] == pytest.approx(
        ema_expanded_limits(50.0)[1])


def test_ema_limits_monotone():
    u = [ema_expanded_limits(c)[1] for c in (30, 35, 40, 45, 50)]
    assert all(b >= a for a, b in zip(u, u[1:]))


def test_fda_bound_exceeds_point_criterion():
    b = fda_rsabe_bound(0.05, 0.03, 30, 0.16, 30)
    assert b["criterion_upper95"] > b["criterion_point"]


def test_fda_bound_passes_identical_products():
    b = fda_rsabe_bound(0.0, 0.02, 40, 0.18, 40)
    assert b["criterion_upper95"] < 0


def test_rsabe_needs_replicate_reference():
    with pytest.raises(ValueError):
        simulate_rsabe(1.0, 45.0, 36, design="bogus")


def test_rsabe_beats_abe_for_hvd():
    kw = dict(cv_wr_pct=45.0, n_subjects=36, n_sim=1500, seed=0)
    abe = simulate_rsabe(1.0, method="ABE", **kw)["p_pass"]
    fda = simulate_rsabe(1.0, method="FDA", **kw)["p_pass"]
    ema = simulate_rsabe(1.0, method="EMA", **kw)["p_pass"]
    assert fda > abe and ema > abe


def test_rsabe_rejects_real_difference():
    assert simulate_rsabe(1.45, 45.0, 36, method="FDA",
                          n_sim=1500, seed=0)["p_pass"] < 0.30


def test_rsabe_no_scaling_at_low_cv():
    r = simulate_rsabe(1.0, 20.0, 36, method="FDA", n_sim=1000, seed=0)
    assert r["p_scaling_applied"] < 0.10


def test_point_estimate_constraint_binds():
    on = simulate_rsabe(1.30, 50.0, 36, method="FDA", n_sim=1500, seed=0)
    off = simulate_rsabe(1.30, 50.0, 36, method="FDA", n_sim=1500, seed=0,
                         apply_pe_constraint=False)
    assert off["p_pass"] >= on["p_pass"]


@pytest.mark.parametrize("cv", [35.0, 45.0, 60.0])
def test_fda_type_i_error_calibrated(cv):
    r = type_i_error_check(cv, 48, method="FDA", n_sim=6000, seed=0,
                           apply_pe_constraint=False)
    assert 0.02 <= r["empirical_type_I"] <= 0.08


# ---------------- crossed effects, random slopes, Satterthwaite ----------
from fi_mixed_models import (RandomTerm, fit_crossed, fit_lmm,
                               fit_random_slope, lrt_terms)
from fi_validation import (cross_validate_model, make_cv,
                             make_permutation_cv, q2_score)
from sklearn.ensemble import RandomForestRegressor


def _crossed(seed=0, ns=10, nc=8, reps=2, vs=1.5, vc=0.5, ve=0.8):
    rng = np.random.default_rng(seed)
    site = np.tile(np.repeat(np.arange(ns), nc), reps)
    camp = np.tile(np.tile(np.arange(nc), ns), reps)
    n = len(site)
    us = rng.normal(0, np.sqrt(vs), ns)
    uc = rng.normal(0, np.sqrt(vc), nc)
    x = rng.normal(size=n)
    y = 2 * x + us[site] + uc[camp] + rng.normal(0, np.sqrt(ve), n)
    return x.reshape(-1, 1), y, site, camp


def test_crossed_separates_two_variance_components():
    X, y, site, camp = _crossed(seed=2)
    f = fit_crossed(X, y, site, camp, ["x"], satterthwaite=False)
    assert set(f.variance_components) == {"site", "campaign"}
    assert f.variance_components["site"] > f.variance_components["campaign"]


def test_crossed_components_unbiased():
    s, c, r = [], [], []
    for rep in range(12):
        X, y, site, camp = _crossed(seed=rep, ns=12, nc=10)
        f = fit_crossed(X, y, site, camp, ["x"], satterthwaite=False)
        s.append(f.variance_components["site"])
        c.append(f.variance_components["campaign"])
        r.append(f.sigma2_e)
    assert np.mean(s) == pytest.approx(1.5, abs=0.6)
    assert np.mean(c) == pytest.approx(0.5, abs=0.3)
    assert np.mean(r) == pytest.approx(0.8, abs=0.1)


def test_crossed_fixed_effect_recovered():
    X, y, site, camp = _crossed(seed=4, ns=12, nc=10)
    f = fit_crossed(X, y, site, camp, ["x"], satterthwaite=False)
    assert f.beta[1] == pytest.approx(2.0, abs=0.2)


def test_lrt_terms_flags_both_crossed_factors():
    X, y, site, camp = _crossed(seed=6, ns=12, nc=10)
    t = lrt_terms(X, y, [RandomTerm(site, "site"), RandomTerm(camp, "campaign")])
    assert len(t) == 2 and t["significant"].all()


def test_multi_term_fit_warns_about_design_effect():
    X, y, site, camp = _crossed(seed=8)
    f = fit_crossed(X, y, site, camp, ["x"], satterthwaite=False)
    assert any("dominant grouping factor" in w for w in f.warnings)


def _slope_data(seed=0, q=40, m=10, vi=1.0, vs=0.6, ve=0.25):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(q), m)
    x = rng.normal(size=q * m)
    ui = rng.normal(0, np.sqrt(vi), q)
    us = rng.normal(0, np.sqrt(vs), q)
    y = 1 + 2 * x + ui[g] + us[g] * x + rng.normal(0, np.sqrt(ve), q * m)
    return x.reshape(-1, 1), y, g


def test_random_slope_variance_recovered():
    est_i, est_s, est_e = [], [], []
    for rep in range(10):
        X, y, g = _slope_data(seed=rep)
        f = fit_random_slope(X, y, g, 0, ["x"], satterthwaite=False)
        est_i.append(f.variance_components["group"])
        est_s.append(f.variance_components["x | group"])
        est_e.append(f.sigma2_e)
    assert np.mean(est_i) == pytest.approx(1.0, abs=0.3)
    assert np.mean(est_s) == pytest.approx(0.6, abs=0.2)
    assert np.mean(est_e) == pytest.approx(0.25, abs=0.05)


def test_random_slope_labels_term():
    X, y, g = _slope_data(seed=1)
    f = fit_random_slope(X, y, g, 0, ["x"], satterthwaite=False)
    assert "x | group" in f.variance_components


def _between_within(seed=5, q=8, m=10):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(q), m)
    lot = rng.normal(size=q)
    u = rng.normal(0, 1.0, q)
    within = rng.normal(size=q * m)
    y = 1.5 * lot[g] + 0.8 * within + u[g] + rng.normal(0, 0.5, q * m)
    return np.column_stack([lot[g], within]), y, g, q


def test_satterthwaite_between_group_df_near_group_count():
    X, y, g, q = _between_within()
    f = fit_random_intercept(X, y, g, ["between", "within"])
    df = f.table().set_index("term")["df_satterthwaite"]
    assert df["between"] == pytest.approx(q - 2, abs=2.0)


def test_satterthwaite_within_group_df_near_row_count():
    X, y, g, q = _between_within()
    f = fit_random_intercept(X, y, g, ["between", "within"])
    df = f.table().set_index("term")["df_satterthwaite"]
    assert df["within"] > 0.7 * (len(y) - q)


def test_satterthwaite_orders_the_two_correctly():
    X, y, g, q = _between_within()
    df = fit_random_intercept(X, y, g, ["between", "within"]).table(
        ).set_index("term")["df_satterthwaite"]
    assert df["within"] > 5 * df["between"]


def test_satterthwaite_can_be_disabled():
    X, y, g, _ = _between_within()
    assert fit_random_intercept(X, y, g, ["a", "b"],
                                satterthwaite=False).dof_satterthwaite is None


# ---------------- group-aware cross-validation ----------------
def test_make_cv_switches_to_group_folds():
    g = np.repeat(np.arange(6), 8)
    assert "group" in make_cv(48, groups=g)[1].lower()
    assert "group" not in make_cv(48)[1].lower()


def test_make_cv_group_kfold_for_many_groups():
    g = np.repeat(np.arange(20), 6)
    assert "Group 5-fold" in make_cv(120, groups=g)[1]


def test_permutation_cv_respects_groups():
    g = np.repeat(np.arange(6), 8)
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    assert isinstance(make_permutation_cv(48, groups=g),
                      (GroupKFold, LeaveOneGroupOut))


def _banded(seed=3, q=8, m=10):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(q), m)
    centres = np.linspace(20, 36, q)
    poly = centres[g] + rng.normal(0, 0.8, q * m)
    hard = rng.uniform(8, 12, q * m)
    off = rng.normal(0, 4.0, q)
    y = 55 - 0.6 * (poly - 28) + off[g] + rng.normal(0, 1.0, q * m)
    return np.column_stack([poly, hard]), y, g


def test_group_cv_exposes_leakage():
    X, y, g = _banded()
    est = RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                random_state=0, n_jobs=-1)
    yt, yp, _ = cross_validate_model(est, X, y, cv=make_cv(len(y), 0)[0])
    q2_rand = q2_score(yt, yp)
    yt2, yp2, _ = cross_validate_model(est, X, y,
                                       cv=make_cv(len(y), 0, groups=g)[0],
                                       groups=g)
    q2_grp = q2_score(yt2, yp2)
    assert q2_rand > q2_grp + 0.5
    assert q2_grp < 0.3


def test_group_cv_no_penalty_when_groups_are_meaningless():
    rng = np.random.default_rng(0)
    n = 80
    X = rng.normal(size=(n, 2))
    y = 2 * X[:, 0] - X[:, 1] + rng.normal(0, 0.3, n)
    g = rng.integers(0, 8, n)          # random labels, no real structure
    from sklearn.linear_model import LinearRegression
    est = LinearRegression()
    q2_rand = q2_score(*cross_validate_model(est, X, y,
                                             cv=make_cv(n, 0)[0])[:2])
    q2_grp = q2_score(*cross_validate_model(est, X, y,
                                            cv=make_cv(n, 0, groups=g)[0],
                                            groups=g)[:2])
    assert abs(q2_rand - q2_grp) < 0.05
