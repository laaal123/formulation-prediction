"""
Percolation / threshold analysis - the mechanistic mode for small n.

When the design gate blocks statistical modelling, this is where the app
routes. With 3-6 batches and a monotonic driver you cannot fit a
response surface, but you CAN locate a threshold and bound the response
between tested levels. The output is an ENVELOPE, never a point
prediction dressed up as one.

Percolation theory: near a critical volume fraction the transport
property follows k ~ (eps - eps_c)^mu, with mu ~ 1.8-2.0 in 3-D. Either
side of eps_c the behaviour is qualitatively different, so interpolating
across it with a smooth model is the classic small-n error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

MU_3D = 1.9


def log_slope_table(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Piecewise d(ln y)/dx between consecutive levels.

    A smooth dose-response gives similar slopes. A large jump is a
    threshold, and it is the single most informative statistic available
    at n=3.
    """
    o = np.argsort(x)
    x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
    ln = np.log(np.maximum(y, 1e-12))
    rows = []
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        rows.append({
            "from": float(x[i]), "to": float(x[i + 1]),
            "y_from": float(y[i]), "y_to": float(y[i + 1]),
            "d_lny_dx": float((ln[i + 1] - ln[i]) / dx) if dx else np.nan,
        })
    df = pd.DataFrame(rows)
    if len(df) >= 2:
        s = df["d_lny_dx"].abs().to_numpy()
        df["steepness_vs_previous"] = np.concatenate(
            [[np.nan], s[1:] / np.maximum(s[:-1], 1e-12)])
    return df


def detect_threshold(x: np.ndarray, y: np.ndarray,
                     ratio_trigger: float = 3.0) -> dict:
    """Locate the interval containing a threshold, if any."""
    df = log_slope_table(x, y)
    if len(df) < 2:
        return {"threshold_detected": False,
                "reason": "need at least 3 levels"}

    r = df["steepness_vs_previous"].to_numpy()[1:]
    ratios = np.where(r >= 1, r, 1.0 / np.maximum(r, 1e-12))
    k = int(np.nanargmax(ratios))
    worst = float(ratios[k])
    detected = worst >= ratio_trigger
    row = df.iloc[k + 1]

    return {
        "threshold_detected": detected,
        "slope_ratio": worst,
        "interval_low": float(row["from"]),
        "interval_high": float(row["to"]),
        "slope_table": df,
        "monotonic": bool(np.all(np.diff(np.asarray(y, float)[
            np.argsort(x)]) <= 0) or np.all(np.diff(np.asarray(y, float)[
                np.argsort(x)]) >= 0)),
        "verdict": (
            f"THRESHOLD between {row['from']:.3g} and {row['to']:.3g} "
            f"({worst:.1f}x change in log-slope). Do not interpolate "
            f"across this interval - the response is not smooth here."
            if detected else
            "No threshold evident; the response looks smooth across the "
            "tested range, but with this few levels that is weak evidence."
        ),
    }


def percolation_fit(x: np.ndarray, y: np.ndarray, mu: float = MU_3D) -> dict:
    """
    Fit k = a * |x - xc|^mu with mu FIXED at the theoretical exponent.

    Fixing mu is what makes this admissible at n=3: two free parameters
    against three points leaves one residual degree of freedom, whereas
    a free exponent would interpolate exactly and prove nothing.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3:
        return {"error": "need at least 3 levels"}

    def model(xx, a, xc):
        return a * np.power(np.abs(xx - xc), mu)

    xc0 = float(np.mean([x.min(), x.max()]))
    try:
        popt, pcov = curve_fit(
            model, x, y, p0=[float(np.ptp(y)) or 1.0, xc0],
            bounds=([-1e6, x.min() - 2 * np.ptp(x)],
                    [1e6, x.max() + 2 * np.ptp(x)]),
            maxfev=50000,
        )
        pred = model(x, *popt)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else [np.nan] * 2
        return {
            "a": float(popt[0]),
            "x_critical": float(popt[1]),
            "x_critical_se": float(perr[1]),
            "mu_fixed": mu,
            "r2": 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan,
            "residual_dof": len(x) - 2,
            "note": ("mu is FIXED at the 3-D theoretical value so that a "
                     "residual degree of freedom remains. A free exponent "
                     "would fit 3 points exactly and demonstrate nothing."),
        }
    except Exception as exc:
        return {"error": str(exc)[:200]}


def prediction_envelope(x: np.ndarray, y: np.ndarray, x_target: float,
                        n_grid: int = 400) -> dict:
    """
    Bound the response at an untested level between two extreme
    interpolants: a smooth log-linear model and a sharp step whose
    location is swept across the bracketing interval.

    The width of this envelope IS the answer. A narrow envelope means the
    level is safe to try; an envelope spanning the acceptance window
    means the experiment would be a coin flip.
    """
    o = np.argsort(x)
    x, y = np.asarray(x, float)[o], np.asarray(y, float)[o]
    ln = np.log(np.maximum(y, 1e-12))

    if x_target < x.min() or x_target > x.max():
        return {"extrapolation": True,
                "note": (f"x={x_target:g} is outside the tested range "
                         f"[{x.min():g}, {x.max():g}]. No envelope is "
                         f"defensible - this is extrapolation.")}

    i = int(np.clip(np.searchsorted(x, x_target) - 1, 0, len(x) - 2))
    lo_x, hi_x = x[i], x[i + 1]
    smooth = float(np.exp(np.interp(x_target, x, ln)))

    sweep = np.linspace(lo_x + 1e-6, hi_x - 1e-6, n_grid)
    step = [y[i] if x_target < xc else y[i + 1] for xc in sweep]
    lo = float(min(min(step), smooth))
    hi = float(max(max(step), smooth))

    return {
        "extrapolation": False,
        "x_target": float(x_target),
        "smooth_estimate": smooth,
        "envelope_low": lo,
        "envelope_high": hi,
        "envelope_width": hi - lo,
        "bracketed_by": (float(lo_x), float(hi_x)),
        "note": (
            f"Smooth interpolation gives {smooth:.3f}, but the honest "
            f"range is {lo:.3f}-{hi:.3f}. The smooth number assumes away "
            f"the non-linearity the data show."
        ),
    }


def be_risk_from_envelope(env: dict, lower: float = 0.80,
                          upper: float = 1.25) -> dict:
    """Does the envelope at a proposed level sit entirely inside BE limits?"""
    if env.get("extrapolation"):
        return {"decision": "DO NOT RUN - extrapolation",
                "note": env.get("note", "")}
    lo, hi = env["envelope_low"], env["envelope_high"]
    width = hi - lo
    window = upper - lower
    inside = lo >= lower and hi <= upper
    straddles = lo < lower or hi > upper

    # Width matters as much as position. An envelope occupying most of the
    # BE window is uninformative even when it technically sits inside it:
    # it says the outcome is undetermined, not that the batch will pass.
    too_wide = width > 0.5 * window

    if straddles:
        decision = (
            "COIN FLIP - the envelope crosses a BE limit. Batch-to-batch "
            "variation in blend uniformity and particle size will decide "
            "the outcome. Move off the threshold instead of interpolating.")
    elif too_wide:
        decision = (
            f"UNINFORMATIVE - the envelope ({lo:.3f}-{hi:.3f}) spans "
            f"{100 * width / window:.0f}% of the BE window. It happens to "
            f"sit inside, but it does not predict a pass. Add an "
            f"intermediate batch or move off the threshold.")
    elif inside:
        decision = (
            f"ACCEPTABLE - the envelope ({lo:.3f}-{hi:.3f}) is inside the "
            f"BE window and narrow enough to be informative.")
    else:
        decision = "UNCERTAIN"

    return {
        "envelope": (lo, hi),
        "envelope_width": float(width),
        "width_pct_of_window": float(100 * width / window),
        "fully_inside_BE": bool(inside),
        "too_wide": bool(too_wide),
        "decision": decision,
    }
