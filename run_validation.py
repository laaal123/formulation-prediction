"""
Validation suite.

Every check has a KNOWN ground truth. If the app cannot recover a driver
it planted itself, or fails to refuse a design it should refuse, it is
not fit for use on real data.

Run:  python validation/run_validation.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import fi_be_simulation as BE  # noqa: E402
import fi_dissolution as D  # noqa: E402
import fi_ivivc as IV  # noqa: E402
import fi_mixed_models as MM  # noqa: E402
import fi_percolation as P  # noqa: E402
import fi_rsabe as RS  # noqa: E402
from fi_design_diagnostics import diagnose  # noqa: E402
from fi_factor_id import consensus  # noqa: E402
from fi_model_arena import run_arena  # noqa: E402
import simulate_data as S  # noqa: E402

RESULTS: list[dict] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    RESULTS.append({"check": name, "pass": bool(passed), "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
          + (f"  — {detail}" if detail else ""), flush=True)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


# ======================================================================
def v1_design_gate():
    section("1. DESIGN GATE — must refuse inadmissible designs")

    df, truth = S.spurious_correlation_trap()
    Xc = ["Ludipress_pct", "Eudragit_pct", "MCC_pct", "Hardness_kP"]
    v = diagnose(df[Xc], df["Cmax_GMR"])
    check("n=3 rank-deficient design is blocked", not v.admissible,
          f"rank={v.rank_centered} < p={v.n_features}")
    check("mixture constraint detected", v.mixture_detected,
          f"{v.mixture_columns}, CV={v.mixture_sum_cv:.2e}")
    check("routed to mechanistic mode", v.mode == "mechanistic")

    corr = df[Xc].corrwith(df["Cmax_GMR"])
    decoy_tops = corr.abs().idxmax() == truth["decoy"]
    check("decoy tops the naive correlation ranking (the trap the gate "
          "exists to stop)", decoy_tops,
          f"{corr.abs().idxmax()} r={corr[corr.abs().idxmax()]:+.4f}")

    doe, _ = S.matrix_tablet_doe(n=48)
    v2 = diagnose(doe[["Polymer_pct", "Filler_pct", "Lubricant_pct",
                       "Hardness_kP"]], doe["Q12h"])
    check("valid n=48 design is admitted", v2.admissible, f"mode={v2.mode}")


def v2_no_signal():
    section("2. NO-SIGNAL DETECTION — noise must not produce a model")

    df, _ = S.pure_noise(n=30, p=5)
    Xc = [c for c in df.columns if c != "Response"]
    a = run_arena(df[Xc], df["Response"], perm_budget_s=8, seed=42)
    check("pure noise reports NO SIGNAL", a.no_signal,
          f"best Q²={a.leaderboard['Q2'].max():.3f}"
          if not a.leaderboard.empty else "")

    if not a.leaderboard.empty:
        check("no noise model reaches positive Q²",
              a.leaderboard["Q2"].max() < 0.2,
              f"max Q²={a.leaderboard['Q2'].max():.3f}")


def v3_signal_recovery():
    section("3. SIGNAL RECOVERY — planted drivers must be found")

    doe, truth = S.matrix_tablet_doe(n=48, seed=3)
    Xc = ["Polymer_pct", "Filler_pct", "Lubricant_pct", "Hardness_kP"]
    a = run_arena(doe[Xc], doe["Q12h"], perm_budget_s=10, seed=42)

    check("signal detected", not a.no_signal,
          f"{a.selected.name} Q²={a.selected.q2:.3f}" if a.selected else "")
    if a.selected:
        check("selected model has real predictive power", a.selected.q2 > 0.5,
              f"Q²={a.selected.q2:.3f}")
        check("permutation test passed", a.selected.perm_p < 0.05,
              f"p={a.selected.perm_p:.4f}")
        check("no severe overfit in the selection",
              (a.selected.r2_train - a.selected.q2) < 0.30,
              f"gap={a.selected.r2_train - a.selected.q2:.3f}")

    fid = consensus(doe[Xc], doe["Q12h"], a.selected_estimator,
                    seed=42, run_permutation=False)
    top = fid.index[0]
    check("dominant driver ranked first", top == truth["dominant"],
          f"top={top}, expected {truth['dominant']}")
    decoy_votes = int(fid.loc[truth["decoys"][0], "votes"])
    driver_votes = int(fid.loc[truth["dominant"], "votes"])
    check("decoy scores below the true driver", decoy_votes < driver_votes,
          f"{truth['decoys'][0]}={decoy_votes} votes, "
          f"{truth['dominant']}={driver_votes} votes")


def v4_mixture():
    section("4. MIXTURE DESIGNS — Scheffé must be offered and competitive")

    mix, truth = S.mixture_design(n=30)
    Xc = truth["mixture_columns"] + truth["process_columns"]
    a = run_arena(mix[Xc], mix["Q12h"], perm_budget_s=8, seed=42)
    names = a.leaderboard["model"].tolist() if not a.leaderboard.empty else []
    check("Scheffé models present in the candidate set",
          any("Scheffe" in n for n in names), f"{len(names)} candidates")
    if not a.leaderboard.empty:
        sch = a.leaderboard[a.leaderboard["model"].str.contains("Scheffe")]
        check("a Scheffé model achieves positive Q²",
              len(sch) > 0 and sch["Q2"].max() > 0,
              f"best Scheffé Q²={sch['Q2'].max():.3f}" if len(sch) else "none")


def v5_dissolution():
    section("5. DISSOLUTION — parameter recovery and discrimination logic")

    t = np.array([0, 1, 2, 4, 6, 8, 12, 16, 24.])
    prof = D.weibull(t, alpha=30.0, beta=0.85)
    fit = D.fit_profile(t, prof, "Weibull")
    check("Weibull parameters recovered",
          abs(fit.params["alpha"] - 30) < 0.5
          and abs(fit.params["beta"] - 0.85) < 0.02,
          f"α={fit.params['alpha']:.2f} β={fit.params['beta']:.3f} "
          f"R²={fit.r2:.4f}")
    check("correct model ranked first by AIC",
          D.best_model(t, prof)["model"].iloc[0] == "Weibull")

    rng = np.random.default_rng(0)

    def units(alpha):
        u = np.array([D.weibull(t, alpha, 0.85) + rng.normal(0, 2, len(t))
                      for _ in range(12)])
        return np.maximum.accumulate(np.clip(u, 0, 100), axis=1)

    same = {b: units(30.0) for b in "ABC"}
    r1 = D.discrimination_power(same, list("ABC"), t)
    check("identical profiles flagged NOT discriminating",
          not r1["discriminating"], f"ρ={r1['spearman_rho_vs_invivo']:.2f}")

    diff = {"A": units(18.0), "B": units(30.0), "C": units(55.0)}
    r2 = D.discrimination_power(diff, ["A", "B", "C"], t)
    check("separated profiles flagged discriminating", r2["discriminating"],
          f"ρ={r2['spearman_rho_vs_invivo']:.2f}, "
          f"f2(extremes)={r2['f2_extremes']:.1f}")

    f2 = D.bootstrap_f2(units(30.0), units(30.5))
    check("bootstrap f2 lower bound below the point estimate",
          f2["f2_lower_90CI"] < f2["f2_point"],
          f"point={f2['f2_point']:.1f}, lower90={f2['f2_lower_90CI']:.1f}")


def v6_ivivc():
    section("6. IVIVC — deconvolution round trip")

    ke = 0.15
    t = np.linspace(0, 36, 200)
    fa_true = np.clip(1 - np.exp(-(t / 12) ** 1.2), 0, 1)
    cp = IV.convolve_to_pk(t, fa_true, 100, 1.0, 20, ke).Cp.to_numpy()
    fa_rec = IV.wagner_nelson(t, cp, ke)
    err = float(np.max(np.abs(fa_rec - fa_true)))
    check("Wagner-Nelson recovers the absorption profile", err < 0.05,
          f"max abs error={err:.4f}")

    la = IV.level_a(fa_true, fa_rec)
    check("Level A established on the round trip", la["linear_acceptable"],
          f"slope={la['slope']:.3f}, r²={la['r2']:.4f}")

    pe = IV.prediction_error(np.array([100.0, 95.0, 88.0]),
                             np.array([103.0, 92.0, 91.0]))
    check("prediction error criterion applied", "acceptable" in pe,
          f"mean |%PE|={pe['mean_abs_PE_pct']:.1f}%")


def v7_virtual_be():
    section("7. VIRTUAL BE — simulation must be calibrated")

    r = BE.simulate_be(1.00, 10.0, 60, n_sim=8000, seed=1)
    check("GMR=1.0, low CV, large n → near-certain pass", r["p_pass"] > 0.95,
          f"p_pass={r['p_pass']:.3f}")

    r = BE.simulate_be(1.30, 25.0, 24, n_sim=8000, seed=1)
    check("GMR outside the window → near-certain fail", r["p_pass"] < 0.05,
          f"p_pass={r['p_pass']:.3f}")

    r = BE.simulate_be(1.00, 25.0, 24, n_sim=8000, seed=1)
    check("GMR=1.0, CV=25%, n=24 → power in the expected 75–92% range",
          0.75 < r["p_pass"] < 0.92, f"p_pass={r['p_pass']:.3f}")

    small = BE.simulate_be(1.10, 30.0, 20, n_sim=6000, seed=1)["p_pass"]
    large = BE.simulate_be(1.10, 30.0, 60, n_sim=6000, seed=1)["p_pass"]
    check("power increases with sample size", large > small,
          f"n=20 → {small:.3f}; n=60 → {large:.3f}")

    lowcv = BE.simulate_be(1.10, 15.0, 30, n_sim=6000, seed=1)["p_pass"]
    hicv = BE.simulate_be(1.10, 40.0, 30, n_sim=6000, seed=1)["p_pass"]
    check("power decreases as intra-subject CV rises", lowcv > hicv,
          f"CV=15% → {lowcv:.3f}; CV=40% → {hicv:.3f}")

    w = BE.gmr_operating_window(25.0, 24, n_sim=1200, seed=1)
    check("operating window is narrower than 0.80–1.25",
          w["width"] is not None and w["width"] < 0.45,
          w["note"][:80])


def v8_percolation():
    section("8. PERCOLATION — threshold must be located, not smoothed over")

    df, truth = S.percolation_series()
    x = df["Polymer_pct"].to_numpy()
    y = df["Cmax_GMR"].to_numpy()
    th = P.detect_threshold(x, y)
    lo, hi = truth["expect_interval"]
    found = (th["threshold_detected"]
             and th["interval_low"] >= lo - 1 and th["interval_high"] <= hi + 1)
    check("threshold interval located", found,
          f"found [{th.get('interval_low')}, {th.get('interval_high')}], "
          f"true εc={truth['x_critical']}")

    fit = P.percolation_fit(x, y)
    check("percolation fit retains a residual degree of freedom",
          fit.get("residual_dof", 0) >= 1,
          f"dof={fit.get('residual_dof')}")

    # The real case: envelope at 29% must be flagged as uninformative
    xr = np.array([25.0, 28.0, 30.0])
    yr = np.array([1.2541, 1.1993, 0.8243])
    env = P.prediction_envelope(xr, yr, 29.0)
    risk = P.be_risk_from_envelope(env)
    check("wide envelope inside BE limits is flagged UNINFORMATIVE, "
          "not ACCEPTABLE", risk["too_wide"],
          f"envelope {env['envelope_low']:.3f}–{env['envelope_high']:.3f} "
          f"= {risk['width_pct_of_window']:.0f}% of the window")
    check("smooth interpolation is visibly more optimistic than the envelope",
          env["smooth_estimate"] > env["envelope_low"],
          f"smooth={env['smooth_estimate']:.3f} vs "
          f"low={env['envelope_low']:.3f}")


def v9_reference_drift():
    section("9. REFERENCE DRIFT — cross-study confounding must be caught")

    df, truth = S.spurious_correlation_trap()
    nz = BE.normalise_across_studies(df)
    sus = nz[nz["reference_suspect"]]
    check("drifting reference arm detected", len(sus) == 1,
          f"{len(sus)} suspect study/studies")
    if len(sus) == 1:
        check("the correct study is flagged",
              sus["batch"].iloc[0] == truth["reference_drift_batch"],
              f"flagged {sus['batch'].iloc[0]}, "
              f"deviation {sus['ref_deviation_pct'].iloc[0]:+.1f}%")
        shift = float(sus["GMR_shift"].iloc[0])
        check("normalisation materially changes that study's GMR",
              abs(shift) > 0.05,
              f"GMR moves {sus['GMR_reported'].iloc[0]:.3f} → "
              f"{sus['GMR_common_ref'].iloc[0]:.3f}")


def v10_applicability():
    section("10. APPLICABILITY DOMAIN — extrapolation must be refused")

    doe, _ = S.matrix_tablet_doe(n=48, seed=9)
    Xc = ["Polymer_pct", "Filler_pct", "Lubricant_pct", "Hardness_kP"]
    a = run_arena(doe[Xc], doe["Q12h"], perm_budget_s=8, seed=42)
    if a.applicability is None:
        check("applicability domain fitted", False, "no model selected")
        return
    ad = a.applicability

    inside = doe[Xc].median().to_numpy().reshape(1, -1)
    check("a central formulation is inside the domain",
          ad.check(inside)[0]["inside"])

    far = doe[Xc].max().to_numpy().reshape(1, -1) * 1.8
    r = ad.check(far)[0]
    check("a far-outside formulation is flagged as extrapolation",
          not r["inside"], r["verdict"][:70])


def v11_mixed_effects():
    section("11. MIXED EFFECTS — batch structure must be measured, not pooled")

    df, truth = S.multi_campaign_doe()
    Xc = ["Polymer_pct", "Hardness_kP", "Lubricant_pct"]
    X = df[Xc].to_numpy(float)
    y = df["Q12h"].to_numpy(float)
    g = df["campaign"].to_numpy()

    fit = MM.fit_random_intercept(X, y, g, Xc)
    check("ICC recovered near the planted value",
          abs(fit.icc - truth["true_icc"]) < 0.20,
          f"estimated {fit.icc:.3f}, planted {truth['true_icc']:.3f}")
    check("effective n is below the row count",
          fit.effective_n < fit.n_obs,
          f"{fit.effective_n:.1f} effective vs {fit.n_obs} rows")

    lrt = MM.lrt_random_effect(X, y, g)
    check("random effect detected", lrt["significant"],
          f"p={lrt['p_value']:.5f}")
    check("boundary mixture used as the reference distribution",
          "mixture" in lrt["reference"])

    # unbiasedness across replicates
    for s2u, s2e in ((0.6, 0.4), (0.0, 1.0)):
        est = []
        for rep in range(25):
            rng = np.random.default_rng(rep)
            q, m = 40, 6
            gg = np.repeat(np.arange(q), m)
            u = rng.normal(0, np.sqrt(s2u), q) if s2u else np.zeros(q)
            Xr = rng.normal(size=(q * m, 2))
            yr = (1 + 2 * Xr[:, 0] - Xr[:, 1] + u[gg]
                  + rng.normal(0, np.sqrt(s2e), q * m))
            est.append(MM.fit_random_intercept(Xr, yr, gg).icc)
        true_icc = s2u / (s2u + s2e)
        check(f"ICC estimator unbiased at true ICC={true_icc:.2f}",
              abs(np.mean(est) - true_icc) < 0.05,
              f"mean estimate {np.mean(est):.3f} (sd {np.std(est):.3f})")

    # the gate must change its answer once grouping is known
    v_pooled = diagnose(df[Xc], df["Q12h"])
    v_grouped = diagnose(df[Xc], df["Q12h"], groups=g)
    check("gate admits the pooled view but blocks the grouped view",
          v_pooled.admissible and not v_grouped.admissible,
          f"pooled n={v_pooled.n_samples} admissible; grouped effective n="
          f"{v_grouped.effective_n:.1f} blocked")

    cmp = MM.compare_pooled_vs_mixed(X, y, g, Xc)
    check("pooled vs mixed standard errors compared",
          np.isfinite(cmp["max_se_inflation"]),
          f"slope SE ratio {cmp['min_se_ratio']:.3f}-"
          f"{cmp['max_se_inflation']:.3f}")
    # Direction depends on whether a factor varies within or between
    # groups. Here every factor varies WITHIN campaign, so the mixed model
    # gains precision by moving the campaign offset out of the residual.
    check("within-group factors gain precision under the mixed model",
          cmp["min_se_ratio"] < 1.0,
          f"slope SE ratio {cmp['min_se_ratio']:.3f}")

    # The guaranteed case: a BETWEEN-group covariate (constant within a
    # campaign, e.g. a raw-material lot property) has an effective sample
    # size equal to the NUMBER OF CAMPAIGNS, not the number of rows.
    # Pooled OLS cannot see that and always reports false precision.
    rng = np.random.default_rng(5)
    q, m = 8, 8
    gg = np.repeat(np.arange(q), m)
    lot = rng.normal(size=q)                 # one value per campaign
    u = rng.normal(0, 1.0, q)
    within = rng.normal(size=q * m)
    yb = 1.5 * lot[gg] + 0.8 * within + u[gg] + rng.normal(0, 0.5, q * m)
    Xb = np.column_stack([lot[gg], within])
    cb = MM.compare_pooled_vs_mixed(Xb, yb, gg, ["lot_property", "within_var"])
    r_between = float(cb["table"].loc[
        cb["table"]["term"] == "lot_property", "se_ratio_LMM_over_OLS"].iloc[0])
    r_within = float(cb["table"].loc[
        cb["table"]["term"] == "within_var", "se_ratio_LMM_over_OLS"].iloc[0])
    check("pooled OLS reports false precision for a between-group covariate",
          r_between > 1.5,
          f"lot_property SE inflates {r_between:.2f}x under the mixed model")
    check("the same fit leaves a within-group covariate essentially unchanged",
          r_within < 1.15,
          f"within_var SE ratio {r_within:.2f}")


def v12_reference_scaled_be():
    section("12. REFERENCE-SCALED BE — limits, calibration and behaviour")

    check("FDA theta constant correct",
          abs(RS.THETA_FDA - (np.log(1.25) / 0.25) ** 2) < 1e-12,
          f"theta={RS.THETA_FDA:.4f}")

    L, U, _ = RS.ema_expanded_limits(50.0)
    check("EMA limits at the CVwR=50% cap match the published values",
          abs(U - 1.4319) < 1e-3 and abs(L - 0.6984) < 1e-3,
          f"{L:.4f}-{U:.4f} (expected 0.6984-1.4319)")
    check("no widening at or below CVwR=30%",
          RS.ema_expanded_limits(30.0)[:2] == (0.80, 1.25))
    check("widening capped above CVwR=50%",
          abs(RS.ema_expanded_limits(70.0)[1]
              - RS.ema_expanded_limits(50.0)[1]) < 1e-9)

    # Type I error of the confidence bound, isolated from the PE constraint
    for cv in (35.0, 45.0, 60.0):
        r = RS.type_i_error_check(cv, 48, method="FDA", n_sim=12000, seed=0,
                                  apply_pe_constraint=False)
        check(f"FDA bound calibrated at CVwR={cv:.0f}% "
              f"(Howe upper 95% bound)",
              0.03 <= r["empirical_type_I"] <= 0.07,
              f"empirical alpha={r['empirical_type_I']:.4f} "
              f"at boundary GMR {r['boundary_gmr']:.4f}")

    kw = dict(cv_wr_pct=45.0, n_subjects=36, n_sim=3000, seed=0)
    abe = RS.simulate_rsabe(1.0, method="ABE", **kw)["p_pass"]
    fda = RS.simulate_rsabe(1.0, method="FDA", **kw)["p_pass"]
    ema = RS.simulate_rsabe(1.0, method="EMA", **kw)["p_pass"]
    check("scaling helps an equivalent highly variable product",
          fda > abe and ema > abe,
          f"ABE={abe:.3f}, EMA={ema:.3f}, FDA={fda:.3f}")

    bad = RS.simulate_rsabe(1.45, 45.0, 36, method="FDA",
                            n_sim=3000, seed=0)["p_pass"]
    check("a genuinely inequivalent product still fails", bad < 0.30,
          f"p_pass={bad:.3f}")

    low = RS.simulate_rsabe(1.0, 20.0, 36, method="FDA",
                            n_sim=2000, seed=0)
    check("scaling withheld when CVwR is low",
          low["p_scaling_applied"] < 0.10,
          f"scaling applied in {low['p_scaling_applied']:.1%} of studies")

    on = RS.simulate_rsabe(1.30, 50.0, 36, method="FDA", n_sim=3000, seed=0)
    off = RS.simulate_rsabe(1.30, 50.0, 36, method="FDA", n_sim=3000, seed=0,
                            apply_pe_constraint=False)
    check("the 80-125% point-estimate constraint binds",
          off["p_pass"] > on["p_pass"],
          f"with constraint {on['p_pass']:.3f}, without {off['p_pass']:.3f}")

    borderline = RS.simulate_rsabe(1.0, 32.0, 36, method="EMA",
                                   n_sim=3000, seed=0)
    check("studies near the 30% threshold sometimes lose scaling",
          borderline["p_scaling_applied"] < 0.95,
          f"scaling permitted in only "
          f"{borderline['p_scaling_applied']:.1%} of simulated studies")


def v13_group_aware_cv():
    section("13. GROUP-AWARE CV — leakage must be exposed, not hidden")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    from fi_validation import (cross_validate_model, make_cv,
                                 make_permutation_cv, q2_score)

    df, _ = S.campaign_banded_doe()
    g = df["campaign"].to_numpy()
    X = df[["Polymer_pct", "Hardness_kP"]].to_numpy(float)
    y = df["Q12h"].to_numpy(float)

    check("group folds selected when a grouping is supplied",
          "group" in make_cv(len(y), 0, groups=g)[1].lower()
          and "group" not in make_cv(len(y), 0)[1].lower(),
          f"{make_cv(len(y), 0)[1]}  ->  {make_cv(len(y), 0, groups=g)[1]}")

    est = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                random_state=0, n_jobs=-1)
    q2_rand = q2_score(*cross_validate_model(
        est, X, y, cv=make_cv(len(y), 0)[0])[:2])
    q2_grp = q2_score(*cross_validate_model(
        est, X, y, cv=make_cv(len(y), 0, groups=g)[0], groups=g)[:2])
    check("random folds inflate Q2 when campaigns band the factor space",
          q2_rand - q2_grp > 0.5,
          f"random Q2={q2_rand:.3f} vs group Q2={q2_grp:.3f} "
          f"(inflation {q2_rand - q2_grp:.3f})")
    check("group folds reveal the model cannot predict an unseen campaign",
          q2_grp < 0.3, f"group Q2={q2_grp:.3f}")

    rng = np.random.default_rng(0)
    n = 90
    Xr = rng.normal(size=(n, 2))
    yr = 2 * Xr[:, 0] - Xr[:, 1] + rng.normal(0, 0.3, n)
    gr = rng.integers(0, 9, n)
    a = q2_score(*cross_validate_model(LinearRegression(), Xr, yr,
                                       cv=make_cv(n, 0)[0])[:2])
    b = q2_score(*cross_validate_model(LinearRegression(), Xr, yr,
                                       cv=make_cv(n, 0, groups=gr)[0],
                                       groups=gr)[:2])
    check("no penalty when the grouping carries no structure",
          abs(a - b) < 0.05, f"random Q2={a:.3f} vs group Q2={b:.3f}")

    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    check("the permutation null also uses group folds",
          isinstance(make_permutation_cv(len(y), groups=g),
                     (GroupKFold, LeaveOneGroupOut)))

    a2 = run_arena(df[["Polymer_pct", "Hardness_kP"]], df["Q12h"],
                   perm_budget_s=6, seed=42, groups=g, force=True)
    check("the arena reports the group CV scheme it used",
          "group" in a2.cv_scheme.lower(), a2.cv_scheme)


def v14_crossed_and_satterthwaite():
    section("14. CROSSED EFFECTS, RANDOM SLOPES, SATTERTHWAITE dof")

    df, truth = S.crossed_site_campaign(n_sites=12, n_campaigns=10, reps=2)
    X = ((df["Polymer_pct"].to_numpy(float) - 28) / 3).reshape(-1, 1)
    y = df["Q12h"].to_numpy(float)
    site, camp = df["site"].to_numpy(), df["campaign"].to_numpy()

    f = MM.fit_crossed(X, y, site, camp, ["x"], satterthwaite=False)
    check("crossed model separates two variance components",
          set(f.variance_components) == {"site", "campaign"},
          ", ".join(f"{k}={v:.3f}" for k, v in f.variance_components.items()))
    check("site variance exceeds campaign variance, as planted",
          f.variance_components["site"] > f.variance_components["campaign"],
          f"true site={truth['var_site']}, campaign={truth['var_campaign']}")

    tt = MM.lrt_terms(X, y, [MM.RandomTerm(site, "site"),
                             MM.RandomTerm(camp, "campaign")])
    check("both crossed terms significant by likelihood ratio",
          bool(tt["significant"].all()),
          ", ".join(f"{r['term']} p={r['p_value']:.2e}"
                    for _, r in tt.iterrows()))

    # unbiasedness over replicates
    s_, c_, r_ = [], [], []
    for rep in range(12):
        rng = np.random.default_rng(rep)
        ns, nc, reps = 12, 10, 2
        st_ = np.tile(np.repeat(np.arange(ns), nc), reps)
        cp = np.tile(np.tile(np.arange(nc), ns), reps)
        n = len(st_)
        us = rng.normal(0, np.sqrt(1.5), ns)
        uc = rng.normal(0, np.sqrt(0.5), nc)
        xx = rng.normal(size=n)
        yy = 2 * xx + us[st_] + uc[cp] + rng.normal(0, np.sqrt(0.8), n)
        ff = MM.fit_crossed(xx.reshape(-1, 1), yy, st_, cp, ["x"],
                            satterthwaite=False)
        s_.append(ff.variance_components["site"])
        c_.append(ff.variance_components["campaign"])
        r_.append(ff.sigma2_e)
    check("crossed variance components unbiased",
          abs(np.mean(s_) - 1.5) < 0.6 and abs(np.mean(c_) - 0.5) < 0.3
          and abs(np.mean(r_) - 0.8) < 0.1,
          f"site {np.mean(s_):.3f}/1.50, campaign {np.mean(c_):.3f}/0.50, "
          f"residual {np.mean(r_):.3f}/0.80")

    # random slope
    est_i, est_s, est_e = [], [], []
    for rep in range(10):
        rng = np.random.default_rng(100 + rep)
        q, m = 40, 10
        gg = np.repeat(np.arange(q), m)
        xx = rng.normal(size=q * m)
        ui = rng.normal(0, 1.0, q)
        us = rng.normal(0, np.sqrt(0.6), q)
        yy = (1 + 2 * xx + ui[gg] + us[gg] * xx
              + rng.normal(0, 0.5, q * m))
        ff = MM.fit_random_slope(xx.reshape(-1, 1), yy, gg, 0, ["x"],
                                 satterthwaite=False)
        est_i.append(ff.variance_components["group"])
        est_s.append(ff.variance_components["x | group"])
        est_e.append(ff.sigma2_e)
    check("random-slope variances unbiased",
          abs(np.mean(est_i) - 1.0) < 0.3 and abs(np.mean(est_s) - 0.6) < 0.2
          and abs(np.mean(est_e) - 0.25) < 0.05,
          f"intercept {np.mean(est_i):.3f}/1.00, slope "
          f"{np.mean(est_s):.3f}/0.60, residual {np.mean(est_e):.3f}/0.25")

    # Satterthwaite: the sharp check
    rng = np.random.default_rng(5)
    q, m = 8, 10
    gg = np.repeat(np.arange(q), m)
    lot = rng.normal(size=q)
    u = rng.normal(0, 1.0, q)
    within = rng.normal(size=q * m)
    yy = 1.5 * lot[gg] + 0.8 * within + u[gg] + rng.normal(0, 0.5, q * m)
    Xb = np.column_stack([lot[gg], within])
    tb = MM.fit_random_intercept(Xb, yy, gg, ["between", "within"]
                                 ).table().set_index("term")
    df_b = float(tb.loc["between", "df_satterthwaite"])
    df_w = float(tb.loc["within", "df_satterthwaite"])
    check("Satterthwaite dof for a between-group covariate is ~ the number "
          "of groups", abs(df_b - (q - 2)) < 2.0,
          f"df={df_b:.2f}, expected about {q - 2} (q={q} groups)")
    check("Satterthwaite dof for a within-group covariate is ~ the row count",
          df_w > 0.7 * (len(yy) - q),
          f"df={df_w:.2f}, expected about {len(yy) - q - 1} "
          f"(n={len(yy)} rows)")
    check("the two differ by an order of magnitude, as they must",
          df_w > 5 * df_b, f"{df_w:.1f} vs {df_b:.1f}")


def main():
    t0 = time.time()
    print("FORMULATION INTELLIGENCE — VALIDATION SUITE")
    print(f"Started {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for fn in (v1_design_gate, v2_no_signal, v3_signal_recovery, v4_mixture,
               v5_dissolution, v6_ivivc, v7_virtual_be, v8_percolation,
               v9_reference_drift, v10_applicability, v11_mixed_effects,
               v12_reference_scaled_be, v13_group_aware_cv,
               v14_crossed_and_satterthwaite):
        try:
            fn()
        except Exception as exc:  # keep going; a crash is itself a result
            check(f"{fn.__name__} raised", False, f"{type(exc).__name__}: {exc}")

    df = pd.DataFrame(RESULTS)
    n_pass = int(df["pass"].sum())
    n_tot = len(df)
    section("SUMMARY")
    print(f"  {n_pass}/{n_tot} checks passed  "
          f"({time.time() - t0:.0f}s)")
    if n_pass < n_tot:
        print("\n  FAILURES:")
        for _, r in df[~df["pass"]].iterrows():
            print(f"    - {r['check']}: {r['detail']}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "validation_results.csv")
    df.to_csv(out, index=False)
    print(f"\n  Written: {out}")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    sys.exit(main())
