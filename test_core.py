"""Unit tests. Run: python -m pytest tests/ -q"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, pytest
from fi_design_diagnostics import diagnose, detect_mixture, variance_inflation
from fi_validation import q2_score, make_cv, apply_one_se_rule, CVResult
from fi_models import ScheffeMixture, lenth_effects
import fi_dissolution as D
import fi_ivivc as IV
import fi_be_simulation as BE
import fi_percolation as P


def test_q2_perfect_and_useless():
    y = np.array([1.0, 2, 3, 4, 5])
    assert q2_score(y, y) == pytest.approx(1.0)
    assert q2_score(y, np.full_like(y, y.mean())) == pytest.approx(0.0)
    assert q2_score(y, y[::-1]) < 0            # must be able to go negative


def test_cv_scheme_switches_at_30():
    assert make_cv(10)[1] == "LOOCV"
    assert "fold" in make_cv(50)[1]


def test_mixture_detection_true_and_false():
    x = np.random.default_rng(0).dirichlet([2, 2, 2], 20)
    det, cols, cv = detect_mixture(pd.DataFrame(x, columns=list("abc")))
    assert det and set(cols) == set("abc") and cv < 1e-6
    z = pd.DataFrame(np.random.default_rng(1).normal(size=(20, 3)), columns=list("abc"))
    assert detect_mixture(z)[0] is False


def test_gate_blocks_rank_deficient():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [2, 4, 6], "c": [1, 1, 2], "d": [3, 1, 4]})
    v = diagnose(X, pd.Series([1.0, 2, 3]))
    assert not v.admissible and v.rank_centered < v.n_features


def test_gate_admits_valid_design():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    assert diagnose(X, pd.Series(rng.normal(size=40))).admissible


def test_vif_detects_perfect_collinearity():
    X = pd.DataFrame({"a": [1., 2, 3, 4, 5], "b": [2., 4, 6, 8, 10], "c": [1., 3, 2, 5, 4]})
    assert np.isinf(variance_inflation(X)["a"])


def test_scheffe_has_no_intercept():
    rng = np.random.default_rng(0)
    x = rng.dirichlet([2, 2, 2], 30)
    y = 10 * x[:, 0] + 20 * x[:, 1] + 5 * x[:, 2]
    m = ScheffeMixture(1).fit(x, y)
    assert len(m.coef_) == 3                    # no intercept term
    assert np.corrcoef(m.predict(x), y)[0, 1] > 0.99


def test_one_se_rule_prefers_simpler():
    a = CVResult("complex", 40, 0.905, 1, 1, .01, .01, .95); a.signal = True
    b = CVResult("simple", 10, 0.900, 1, 1, .01, .01, .91); b.signal = True
    assert apply_one_se_rule([a, b]).name == "simple"


def test_one_se_rule_excludes_no_signal():
    a = CVResult("x", 10, 0.9, 1, 1, .01, .01, .9); a.signal = False
    assert apply_one_se_rule([a]) is None


def test_lenth_flags_large_effects():
    r = lenth_effects([8.0, .1, -7.5, .2, .1, -.15, .05], list("abcdefg"))
    assert {d["factor"] for d in r if d["significant_ME"]} == {"a", "c"}


def test_weibull_roundtrip():
    t = np.array([0, 1, 2, 4, 8, 12, 24.])
    f = D.weibull(t, 25.0, 0.9)
    fit = D.fit_profile(t, f, "Weibull")
    assert fit.params["alpha"] == pytest.approx(25.0, rel=.02)
    assert fit.r2 > 0.999


def test_f2_identical_profiles_is_100():
    p = np.array([10., 30, 50, 70, 90])
    assert D.f2_similarity(p, p) == pytest.approx(100.0)


def test_f2_bootstrap_bounds_bracket_point():
    rng = np.random.default_rng(0)
    R = np.array([[10., 30, 50, 70, 90] + rng.normal(0, 2, 5) for _ in range(8)])
    T = np.array([[12., 33, 53, 72, 91] + rng.normal(0, 2, 5) for _ in range(8)])
    r = D.bootstrap_f2(R, T, n_boot=800)
    assert r["f2_lower_90CI"] <= r["f2_point"] <= r["f2_upper_90CI"]


def test_wagner_nelson_monotone_and_bounded():
    t = np.linspace(0, 24, 100)
    fa = np.clip(1 - np.exp(-(t / 8) ** 1.1), 0, 1)
    cp = IV.convolve_to_pk(t, fa, 100, 1, 20, .2).Cp.to_numpy()
    r = IV.wagner_nelson(t, cp, .2)
    assert r.min() >= 0 and r.max() <= 1.0001
    assert np.all(np.diff(r) >= -1e-6)


def test_cv_sigma_roundtrip():
    assert BE.sigma_to_cv(BE.cv_to_sigma(30.0)) == pytest.approx(30.0)


def test_be_power_monotone_in_n():
    lo = BE.simulate_be(1.05, 25, 16, n_sim=3000, seed=0)["p_pass"]
    hi = BE.simulate_be(1.05, 25, 64, n_sim=3000, seed=0)["p_pass"]
    assert hi > lo


def test_be_extreme_gmr_fails():
    assert BE.simulate_be(1.5, 20, 40, n_sim=3000, seed=0)["p_pass"] < 0.01


def test_reference_drift_flagged():
    d = pd.DataFrame({"Cmax_T": [100., 100, 100], "Cmax_R": [100., 100, 130]})
    r = BE.normalise_across_studies(d)
    assert r["reference_suspect"].sum() == 1


def test_threshold_detected_and_absent():
    x = np.array([20., 25, 30, 35])
    assert P.detect_threshold(x, np.array([1.3, 1.25, .9, .5]))["threshold_detected"]
    assert not P.detect_threshold(x, np.array([1.3, 1.2, 1.1, 1.0]))["threshold_detected"]


def test_envelope_rejects_extrapolation():
    assert P.prediction_envelope(np.array([25., 28, 30]),
                                 np.array([1.25, 1.2, .82]), 35.0)["extrapolation"]


def test_wide_envelope_flagged_uninformative():
    env = P.prediction_envelope(np.array([25., 28, 30]), np.array([1.25, 1.2, .82]), 29.)
    assert P.be_risk_from_envelope(env)["too_wide"]
