"""
Reference-scaled average bioequivalence for highly variable drugs.

Standard ABE with fixed 80-125% limits is close to unpassable when the
within-subject CV of the reference exceeds about 30%: the study needs
enormous n even when the products are genuinely identical. Both major
regulators therefore allow the acceptance range to be scaled to the
reference variability, using a replicate design so that CVwR is
estimable.

Two frameworks, and they are NOT interchangeable:

  FDA RSABE   - linearised scaled criterion, upper 95% confidence bound
                by Howe's approximation, plus a point-estimate
                constraint of 80-125%. Scaling applies from CVwR >= 30%.

  EMA ABEL    - widened acceptance limits exp(+/- k.swR) with k = 0.760,
                applied to Cmax only, capped at CVwR = 50%
                (69.84-143.19%), plus the same point-estimate constraint.

Both require a replicate design. This module simulates the study, so it
reports the PROBABILITY of passing rather than a single verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fi_be_simulation import BE_LOWER, BE_UPPER, cv_to_sigma, sigma_to_cv

# ----------------------------------------------------------------------
SIGMA_W0 = 0.25                     # FDA regulatory constant
THETA_FDA = (np.log(1.25) / SIGMA_W0) ** 2          # ~3.1615
K_EMA = 0.760                       # EMA regulatory constant
CV_SCALING_THRESHOLD = 30.0         # % CVwR at which scaling switches on
CV_CAP_EMA = 50.0                   # % CVwR at which EMA stops widening

DESIGNS = {
    "full_replicate_4p": {"n_T": 2, "n_R": 2,
                          "label": "4-period full replicate (TRTR/RTRT)"},
    "partial_replicate_3p": {"n_T": 1, "n_R": 2,
                             "label": "3-period partial replicate (TRR/RTR/RRT)"},
    "full_replicate_3p": {"n_T": 1, "n_R": 2,
                          "label": "3-period full replicate (TRT/RTR)"},
}


# ----------------------------------------------------------------------
def ema_expanded_limits(cv_wr_pct: float) -> tuple[float, float, str]:
    """
    EMA widened acceptance limits from the observed CVwR.

    Below 30% no widening is permitted; above 50% the widening is capped,
    which is why a very highly variable product does not get unlimited
    latitude.
    """
    if cv_wr_pct <= CV_SCALING_THRESHOLD:
        return BE_LOWER, BE_UPPER, "no scaling (CVwR <= 30%)"
    cv_used = min(cv_wr_pct, CV_CAP_EMA)
    s_wr = cv_to_sigma(cv_used)
    upper = float(np.exp(K_EMA * s_wr))
    note = ("scaled" if cv_wr_pct <= CV_CAP_EMA
            else f"scaled, capped at CVwR={CV_CAP_EMA:.0f}%")
    return float(1.0 / upper), upper, note


def fda_rsabe_bound(point_diff: float, se_diff: float, dof_diff: int,
                    s2_wr: float, dof_wr: int, alpha: float = 0.05) -> dict:
    """
    Upper 95% confidence bound on the FDA linearised criterion

        E = (muT - muR)^2 - theta . s2wR   <= 0

    via Howe's approximation. The bound is built by pushing each term to
    its own one-sided limit and combining in quadrature: the point
    estimate to its upper bound, and s2wR to its LOWER bound (a smaller
    reference variance makes the criterion harder to satisfy, so that is
    the conservative direction).
    """
    t_crit = stats.t.ppf(1 - alpha, max(dof_diff, 1))
    chi2_hi = stats.chi2.ppf(1 - alpha, max(dof_wr, 1))

    e_hat = point_diff ** 2 - THETA_FDA * s2_wr

    upper_x = (abs(point_diff) + t_crit * se_diff) ** 2
    c_x = (upper_x - point_diff ** 2) ** 2

    s2_wr_low = dof_wr * s2_wr / chi2_hi
    c_s = (THETA_FDA * s2_wr - THETA_FDA * s2_wr_low) ** 2

    return {
        "criterion_point": float(e_hat),
        "criterion_upper95": float(e_hat + np.sqrt(c_x + c_s)),
        "theta": float(THETA_FDA),
    }


# ----------------------------------------------------------------------
def _simulate_replicate(rng, log_gmr, sigma_wt, sigma_wr, sigma_b,
                        n_subj, n_T, n_R):
    """
    One replicate-design study on the log scale.

    Returns (point estimate of T-R, its standard error, its dof,
             s2wR estimate, dof of s2wR).
    """
    subj = rng.normal(0.0, sigma_b, n_subj)

    T = subj[:, None] + log_gmr + rng.normal(0, sigma_wt, (n_subj, n_T))
    R = subj[:, None] + rng.normal(0, sigma_wr, (n_subj, n_R))

    # Subject-level contrast: robust to the subject random effect
    d = T.mean(axis=1) - R.mean(axis=1)
    point = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n_subj))
    dof_d = n_subj - 1

    # Within-subject reference variance from the replicated R periods
    dr = R[:, 0] - R[:, 1]
    s2_wr = float(np.sum((dr - dr.mean()) ** 2) / (2 * (n_subj - 1)))
    dof_wr = n_subj - 1

    return point, se, max(dof_d, 1), max(s2_wr, 1e-12), max(dof_wr, 1)


def simulate_rsabe(
    true_gmr: float,
    cv_wr_pct: float,
    n_subjects: int,
    cv_wt_pct: float | None = None,
    cv_between_pct: float = 40.0,
    design: str = "full_replicate_4p",
    method: str = "FDA",
    n_sim: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
    apply_pe_constraint: bool = True,
) -> dict:
    """
    Monte Carlo a reference-scaled BE study.

    `method` is 'FDA' (linearised criterion), 'EMA' (widened limits) or
    'ABE' (unscaled 80-125%, for comparison). Scaling decisions are made
    from the SIMULATED CVwR in each replicate, exactly as they would be
    in a real study - not from the true value. That matters: a study can
    fail simply because its observed CVwR landed below 30% and scaling
    was therefore not permitted.

    Note on the ABE comparison: it is evaluated on the SAME replicate
    data, so it isolates the effect of the decision rule. It is not the
    power of a 2x2x2 crossover, which would be lower again because the
    treatment contrast is not averaged over replicate periods.

    `apply_pe_constraint` exists for validation only. Both frameworks
    require the point estimate to sit within 80-125%, and in practice
    that constraint - not the confidence bound - is what binds at the
    scaled boundary. Switching it off isolates the statistical bound so
    its type I error can be checked on its own; a real study always has
    it on.
    """
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {list(DESIGNS)}")
    d = DESIGNS[design]
    if d["n_R"] < 2:
        raise ValueError("reference-scaling requires >= 2 reference periods")

    rng = np.random.default_rng(seed)
    sigma_wr = cv_to_sigma(cv_wr_pct)
    sigma_wt = cv_to_sigma(cv_wt_pct if cv_wt_pct is not None else cv_wr_pct)
    sigma_b = cv_to_sigma(cv_between_pct)
    log_gmr = np.log(true_gmr)

    passed = np.zeros(n_sim, dtype=bool)
    scaled = np.zeros(n_sim, dtype=bool)
    pe_ok = np.zeros(n_sim, dtype=bool)
    cvwr_obs = np.zeros(n_sim)
    gmr_obs = np.zeros(n_sim)
    lo_obs = np.full(n_sim, np.nan)
    hi_obs = np.full(n_sim, np.nan)

    for i in range(n_sim):
        point, se, dof_d, s2_wr, dof_wr = _simulate_replicate(
            rng, log_gmr, sigma_wt, sigma_wr, sigma_b,
            n_subjects, d["n_T"], d["n_R"])

        cv_hat = sigma_to_cv(np.sqrt(s2_wr))
        cvwr_obs[i] = cv_hat
        gmr_obs[i] = np.exp(point)
        pe_ok[i] = BE_LOWER <= np.exp(point) <= BE_UPPER

        if method == "ABE":
            t_c = stats.t.ppf(1 - alpha, dof_d)
            lo, hi = np.exp(point - t_c * se), np.exp(point + t_c * se)
            lo_obs[i], hi_obs[i] = lo, hi
            passed[i] = (lo >= BE_LOWER) and (hi <= BE_UPPER)

        elif method == "EMA":
            L, U, _ = ema_expanded_limits(cv_hat)
            scaled[i] = cv_hat > CV_SCALING_THRESHOLD
            t_c = stats.t.ppf(1 - alpha, dof_d)
            lo, hi = np.exp(point - t_c * se), np.exp(point + t_c * se)
            lo_obs[i], hi_obs[i] = lo, hi
            passed[i] = (lo >= L) and (hi <= U) and (
                pe_ok[i] or not apply_pe_constraint)

        elif method == "FDA":
            if cv_hat >= CV_SCALING_THRESHOLD:
                scaled[i] = True
                b = fda_rsabe_bound(point, se, dof_d, s2_wr, dof_wr, alpha)
                passed[i] = (b["criterion_upper95"] <= 0) and (
                    pe_ok[i] or not apply_pe_constraint)
            else:
                t_c = stats.t.ppf(1 - alpha, dof_d)
                lo, hi = np.exp(point - t_c * se), np.exp(point + t_c * se)
                lo_obs[i], hi_obs[i] = lo, hi
                passed[i] = (lo >= BE_LOWER) and (hi <= BE_UPPER)
        else:
            raise ValueError("method must be 'FDA', 'EMA' or 'ABE'")

    L_nom, U_nom, note = ema_expanded_limits(cv_wr_pct)
    return {
        "method": method,
        "design": d["label"],
        "true_gmr": float(true_gmr),
        "cv_wr_pct": float(cv_wr_pct),
        "n_subjects": int(n_subjects),
        "n_sim": int(n_sim),
        "p_pass": float(passed.mean()),
        "p_scaling_applied": float(scaled.mean()),
        "p_point_estimate_ok": float(pe_ok.mean()),
        "median_observed_CVwR": float(np.median(cvwr_obs)),
        "median_observed_GMR": float(np.median(gmr_obs)),
        "nominal_EMA_limits": (L_nom, U_nom),
        "limits_note": note,
        "verdict": _verdict_rsabe(float(passed.mean()), method),
    }


def _verdict_rsabe(p: float, method: str) -> str:
    tag = {"FDA": "FDA RSABE", "EMA": "EMA ABEL",
           "ABE": "unscaled ABE"}.get(method, method)
    if p >= 0.90:
        return f"GO - high probability of passing under {tag}"
    if p >= 0.80:
        return f"GO with caution under {tag}"
    if p >= 0.50:
        return f"MARGINAL under {tag} - coin flip"
    return f"NO-GO under {tag}"


# ----------------------------------------------------------------------
def compare_frameworks(true_gmr: float, cv_wr_pct: float, n_subjects: int,
                       design: str = "full_replicate_4p",
                       n_sim: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Side-by-side power under ABE, EMA ABEL and FDA RSABE."""
    rows = []
    for m in ("ABE", "EMA", "FDA"):
        r = simulate_rsabe(true_gmr, cv_wr_pct, n_subjects, design=design,
                           method=m, n_sim=n_sim, seed=seed)
        rows.append({
            "framework": {"ABE": "Unscaled ABE (80-125%)",
                          "EMA": "EMA ABEL (widened limits)",
                          "FDA": "FDA RSABE (scaled criterion)"}[m],
            "p_pass": round(r["p_pass"], 4),
            "scaling_applied": round(r["p_scaling_applied"], 3),
            "point_est_constraint_met": round(r["p_point_estimate_ok"], 3),
            "verdict": r["verdict"],
        })
    return pd.DataFrame(rows)


def rsabe_sample_size(true_gmr: float, cv_wr_pct: float,
                      target_power: float = 0.80, method: str = "FDA",
                      design: str = "full_replicate_4p",
                      n_sim: int = 2000, seed: int = 0,
                      n_max: int = 120) -> dict:
    """Smallest even n reaching the target power under the chosen framework."""
    curve, found = [], None
    for n in range(12, n_max + 1, 6):
        p = simulate_rsabe(true_gmr, cv_wr_pct, n, design=design,
                           method=method, n_sim=n_sim, seed=seed)["p_pass"]
        curve.append({"n_subjects": n, "power": p})
        if found is None and p >= target_power:
            found = n
            break
    return {
        "n_required": found,
        "method": method,
        "curve": pd.DataFrame(curve),
        "note": (f"n={found} subjects reaches {target_power:.0%} power under "
                 f"{method}" if found else
                 f"Target power not reached by n={n_max} under {method}."),
    }


def type_i_error_check(cv_wr_pct: float, n_subjects: int,
                       method: str = "FDA", design: str = "full_replicate_4p",
                       n_sim: int = 20000, seed: int = 0,
                       apply_pe_constraint: bool = True) -> dict:
    """
    Empirical type I error at the regulatory boundary.

    The GMR is placed exactly on the scaled limit, where a correctly
    constructed procedure rejects about 5% of the time. This is the one
    check that catches a mis-specified confidence bound: a bound that is
    too liberal shows up here as inflated error, and nowhere else.
    """
    s_wr = cv_to_sigma(cv_wr_pct)
    if method == "FDA":
        boundary = float(np.exp(np.sqrt(THETA_FDA) * s_wr))
    else:
        boundary = ema_expanded_limits(cv_wr_pct)[1]
    boundary = min(boundary, BE_UPPER) if cv_wr_pct <= CV_SCALING_THRESHOLD \
        else boundary

    r = simulate_rsabe(boundary, cv_wr_pct, n_subjects, design=design,
                       method=method, n_sim=n_sim, seed=seed,
                       apply_pe_constraint=apply_pe_constraint)
    return {
        "method": method,
        "boundary_gmr": boundary,
        "cv_wr_pct": cv_wr_pct,
        "empirical_type_I": r["p_pass"],
        "nominal": 0.05,
        "controlled": bool(r["p_pass"] <= 0.075),
        "pe_constraint_applied": bool(apply_pe_constraint),
        "note": (
            f"At the boundary GMR of {boundary:.4f} the procedure passes "
            f"{r['p_pass']:.1%} of the time (nominal 5%)"
            + (" with the point-estimate constraint applied, which is why "
               "the observed rate sits well below nominal - at a boundary "
               "GMR that far from 1.0 the constraint almost always binds "
               "first." if apply_pe_constraint else
               " with the confidence bound acting alone.")),
    }
