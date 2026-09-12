"""
Virtual bioequivalence.

The output is a PROBABILITY that the 90% CI of the GMR falls inside
80-125%, obtained by Monte Carlo simulation of the crossover study using
the known intra-subject CV. That is a defensible number because there is
a mechanistic chain behind it. A black-box pass/fail classifier trained
on a handful of BE studies is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BE_LOWER, BE_UPPER = 0.80, 1.25


def cv_to_sigma(cv_pct: float) -> float:
    """Intra-subject CV (%) -> SD on the log scale."""
    return float(np.sqrt(np.log(1.0 + (cv_pct / 100.0) ** 2)))


def sigma_to_cv(sigma: float) -> float:
    return float(100.0 * np.sqrt(np.exp(sigma ** 2) - 1.0))


def simulate_be(
    true_gmr: float,
    cv_intra_pct: float,
    n_subjects: int,
    n_sim: int = 5000,
    design: str = "2x2x2",
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """
    Monte Carlo a crossover BE study.

    design: '2x2x2' (two-period crossover) or 'replicate' (4-period full
    replicate, which halves the variance of the treatment contrast).
    """
    rng = np.random.default_rng(seed)
    sigma_w = cv_to_sigma(cv_intra_pct)
    log_gmr = np.log(true_gmr)

    if design == "replicate":
        dof = 3 * n_subjects - 4
        se_factor = np.sqrt(2.0 / (2 * n_subjects)) / np.sqrt(2)
    else:
        dof = n_subjects - 2
        se_factor = np.sqrt(2.0 / n_subjects)
    dof = max(dof, 1)
    tcrit = stats.t.ppf(1 - alpha, dof)

    # sample the observed point estimate and the observed within-subject SD
    est = rng.normal(log_gmr, sigma_w * se_factor, n_sim)
    s2 = sigma_w ** 2 * rng.chisquare(dof, n_sim) / dof
    se = np.sqrt(s2) * se_factor

    lo = np.exp(est - tcrit * se)
    hi = np.exp(est + tcrit * se)
    passed = (lo >= BE_LOWER) & (hi <= BE_UPPER)

    return {
        "true_gmr": float(true_gmr),
        "cv_intra_pct": float(cv_intra_pct),
        "n_subjects": int(n_subjects),
        "design": design,
        "n_sim": int(n_sim),
        "p_pass": float(passed.mean()),
        "median_gmr": float(np.median(np.exp(est))),
        "median_CI_lower": float(np.median(lo)),
        "median_CI_upper": float(np.median(hi)),
        "p_fail_low": float(((lo < BE_LOWER) & ~passed).mean()),
        "p_fail_high": float(((hi > BE_UPPER) & ~passed).mean()),
        "verdict": _verdict(float(passed.mean())),
    }


def _verdict(p: float) -> str:
    if p >= 0.90:
        return "GO - high probability of passing"
    if p >= 0.80:
        return "GO with caution - acceptable but not comfortable"
    if p >= 0.50:
        return "MARGINAL - coin flip; reformulate or increase n first"
    return "NO-GO - do not run this study as designed"


def sample_size_for_power(true_gmr: float, cv_intra_pct: float,
                          target_power: float = 0.80,
                          design: str = "2x2x2", n_sim: int = 3000,
                          seed: int = 0, n_max: int = 200) -> dict:
    """Smallest even n reaching the target power, by simulation."""
    curve = []
    found = None
    for n in range(6, n_max + 1, 2):
        r = simulate_be(true_gmr, cv_intra_pct, n, n_sim=n_sim,
                        design=design, seed=seed)
        curve.append({"n_subjects": n, "power": r["p_pass"]})
        if found is None and r["p_pass"] >= target_power:
            found = n
            if n >= 24:
                break
    return {
        "n_required": found,
        "target_power": target_power,
        "achievable": found is not None,
        "curve": pd.DataFrame(curve),
        "note": (f"n={found} subjects reaches {target_power:.0%} power"
                 if found else
                 f"Target power not reached by n={n_max}. The GMR is too "
                 f"far from 1.0 - reformulate rather than enrol more."),
    }


def gmr_operating_window(cv_intra_pct: float, n_subjects: int,
                         target_power: float = 0.80, n_sim: int = 2000,
                         seed: int = 0) -> dict:
    """
    The range of TRUE GMR values that would pass with the target power at
    this n and CV. This is the real acceptance target for formulation
    work - much narrower than 0.80-1.25.
    """
    grid = np.round(np.arange(0.80, 1.2601, 0.01), 4)
    rows = [{"true_gmr": float(g),
             "power": simulate_be(g, cv_intra_pct, n_subjects,
                                  n_sim=n_sim, seed=seed)["p_pass"]}
            for g in grid]
    df = pd.DataFrame(rows)
    ok = df[df["power"] >= target_power]
    return {
        "curve": df,
        "gmr_low": float(ok["true_gmr"].min()) if len(ok) else None,
        "gmr_high": float(ok["true_gmr"].max()) if len(ok) else None,
        "width": (float(ok["true_gmr"].max() - ok["true_gmr"].min())
                  if len(ok) else 0.0),
        "note": (
            f"With CV={cv_intra_pct:.0f}% and n={n_subjects}, only a true "
            f"GMR between {ok['true_gmr'].min():.2f} and "
            f"{ok['true_gmr'].max():.2f} passes with "
            f"{target_power:.0%} power."
            if len(ok) else
            f"No true GMR reaches {target_power:.0%} power at n="
            f"{n_subjects} with CV={cv_intra_pct:.0f}%. Increase n."
        ),
    }


def propagate_prediction_uncertainty(
    gmr_mean: float, gmr_sd: float, cv_intra_pct: float, n_subjects: int,
    n_outer: int = 400, n_inner: int = 800, seed: int = 0,
) -> dict:
    """
    Two-stage Monte Carlo: the model's uncertainty about the true GMR is
    propagated through the study simulation.

    Reporting a single p_pass from a point-estimate GMR hides the fact
    that the GMR itself came from a model. This does not.
    """
    rng = np.random.default_rng(seed)
    draws = rng.normal(gmr_mean, max(gmr_sd, 1e-9), n_outer)
    draws = np.clip(draws, 0.4, 2.5)
    ps = [simulate_be(float(g), cv_intra_pct, n_subjects,
                      n_sim=n_inner, seed=seed + i)["p_pass"]
          for i, g in enumerate(draws)]
    ps = np.array(ps)
    return {
        "p_pass_mean": float(ps.mean()),
        "p_pass_lo90": float(np.percentile(ps, 5)),
        "p_pass_hi90": float(np.percentile(ps, 95)),
        "gmr_draws": draws,
        "p_pass_draws": ps,
        "verdict": _verdict(float(ps.mean())),
        "note": (
            f"Accounting for model uncertainty in the GMR "
            f"(SD={gmr_sd:.3f}), the probability of passing is "
            f"{ps.mean():.0%} with a 90% range of "
            f"{np.percentile(ps, 5):.0%}-{np.percentile(ps, 95):.0%}."
        ),
    }


def normalise_across_studies(df: pd.DataFrame, ref_col: str = "Cmax_R",
                             test_col: str = "Cmax_T",
                             common_ref: float | None = None) -> pd.DataFrame:
    """
    Reference-arm drift check.

    Comparing test products ACROSS separate BE studies is confounded with
    the reference arm, the subject panel and the RLD lot. If one study's
    reference ran high, that study's GMR is biased low and a formulation
    conclusion drawn from the comparison is wrong. This recomputes every
    GMR against a common reference so the drift is visible.
    """
    out = df.copy()
    out["GMR_reported"] = out[test_col] / out[ref_col]
    ref = float(common_ref if common_ref is not None
                else np.median(out[ref_col]))
    out["GMR_common_ref"] = out[test_col] / ref
    out["ref_deviation_pct"] = 100 * (out[ref_col] - ref) / ref
    out["GMR_shift"] = out["GMR_common_ref"] - out["GMR_reported"]
    out["reference_suspect"] = out["ref_deviation_pct"].abs() > 10
    out.attrs["common_reference"] = ref
    return out
