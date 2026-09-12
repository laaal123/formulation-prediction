"""
IVIVC - in vitro / in vivo correlation.

Direction of travel for a BE question:
    in vivo Cp  --deconvolute-->  fraction absorbed
    fraction absorbed  <--Level A-->  fraction dissolved
    predicted dissolution  --convolve-->  predicted Cp  --> virtual BE

Deconvolution methods:
  * Wagner-Nelson      : one-compartment, needs ke only
  * Loo-Riegelman      : two-compartment, needs k10, k12, k21
  * Numerical (Wagner) : model-independent, needs a unit impulse response
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import cumulative_trapezoid


# ----------------------------------------------------------------------
def wagner_nelson(t: np.ndarray, cp: np.ndarray, ke: float,
                  normalise: bool = True) -> np.ndarray:
    """
    Fraction absorbed for a one-compartment drug.

        Fa(t) = (Cp(t) + ke * AUC(0-t)) / (ke * AUC(0-inf))
    """
    t = np.asarray(t, float)
    cp = np.asarray(cp, float)
    auc_t = np.concatenate([[0.0], cumulative_trapezoid(cp, t)])
    amount = cp + ke * auc_t
    auc_inf = auc_t[-1] + cp[-1] / ke
    denom = ke * auc_inf
    fa = amount / denom if denom > 1e-12 else amount * 0
    return np.clip(fa / fa[-1], 0, 1) if normalise and fa[-1] > 0 else np.clip(fa, 0, 1)


def loo_riegelman(t: np.ndarray, cp: np.ndarray, k10: float, k12: float,
                  k21: float, normalise: bool = True) -> np.ndarray:
    """
    Fraction absorbed for a two-compartment drug. The peripheral amount
    Cp_periph is built recursively from the exact solution over each
    interval, which is the standard formulation.
    """
    t = np.asarray(t, float)
    cp = np.asarray(cp, float)
    n = len(t)
    cper = np.zeros(n)

    for i in range(1, n):
        dt = t[i] - t[i - 1]
        e = np.exp(-k21 * dt)
        cper[i] = (cper[i - 1] * e
                   + k12 * cp[i - 1] * (1 - e) / k21
                   + k12 * (cp[i] - cp[i - 1]) * dt / 2.0)

    auc_t = np.concatenate([[0.0], cumulative_trapezoid(cp, t)])
    amount = cp + cper + k10 * auc_t
    auc_inf = auc_t[-1] + cp[-1] / k10
    denom = cp[-1] + cper[-1] + k10 * auc_inf
    fa = amount / denom if denom > 1e-12 else amount * 0
    return np.clip(fa / fa[-1], 0, 1) if normalise and fa[-1] > 0 else np.clip(fa, 0, 1)


def numerical_deconvolution(t: np.ndarray, cp: np.ndarray,
                            t_imp: np.ndarray, c_imp: np.ndarray,
                            normalise: bool = True) -> np.ndarray:
    """
    Model-independent (Wagner) point-area deconvolution against a unit
    impulse response - e.g. an IV bolus or an oral solution profile.
    """
    t = np.asarray(t, float)
    cp = np.asarray(cp, float)
    imp = np.interp(t, np.asarray(t_imp, float), np.asarray(c_imp, float))
    if imp[0] <= 1e-12:
        imp = np.maximum(imp, 1e-12)

    n = len(t)
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    inp = np.zeros(n)
    for i in range(n):
        conv = sum(inp[j] * imp[i - j] * dt[j] for j in range(i))
        inp[i] = max((cp[i] - conv) / (imp[0] * dt[i]), 0.0)

    cum = np.cumsum(inp * dt)
    return np.clip(cum / cum[-1], 0, 1) if normalise and cum[-1] > 0 else cum


# ----------------------------------------------------------------------
def level_a(fdiss: np.ndarray, fabs: np.ndarray,
            time_scale: bool = False) -> dict:
    """
    Level A correlation: fraction absorbed vs fraction dissolved.

    A slope near 1 and an intercept near 0 is the ideal. A slope far from
    1 means the in vitro method runs at the wrong rate and needs time
    scaling before it can be used predictively.
    """
    x = np.asarray(fdiss, float)
    y = np.asarray(fabs, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return {"error": "need at least 3 paired points"}

    lr = stats.linregress(x, y)
    pred = lr.intercept + lr.slope * x
    out = {
        "slope": float(lr.slope),
        "intercept": float(lr.intercept),
        "r2": float(lr.rvalue ** 2),
        "p_value": float(lr.pvalue),
        "stderr_slope": float(lr.stderr),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "n_points": int(len(x)),
    }
    out["linear_acceptable"] = bool(out["r2"] >= 0.95
                                    and 0.8 <= out["slope"] <= 1.25)
    if time_scale and lr.slope > 1e-9:
        out["time_scaling_factor"] = float(1.0 / lr.slope)
    out["verdict"] = (
        "Level A established" if out["linear_acceptable"] else
        f"Not acceptable (r2={out['r2']:.3f}, slope={out['slope']:.3f}). "
        f"Consider time scaling or a more discriminating method."
    )
    return out


def prediction_error(observed: np.ndarray, predicted: np.ndarray,
                     label: str = "internal") -> dict:
    """
    %PE per FDA IVIVC guidance.
    Internal: mean <=10%, none >15%. External: <=10% acceptable.
    """
    obs = np.asarray(observed, float)
    pre = np.asarray(predicted, float)
    pe = 100 * (obs - pre) / np.where(np.abs(obs) > 1e-12, obs, np.nan)
    mean_abs = float(np.nanmean(np.abs(pe)))
    max_abs = float(np.nanmax(np.abs(pe)))
    if label == "internal":
        ok = mean_abs <= 10 and max_abs <= 15
    else:
        ok = mean_abs <= 10
    return {
        "per_batch_PE_pct": pe.tolist(),
        "mean_abs_PE_pct": mean_abs,
        "max_abs_PE_pct": max_abs,
        "acceptable": bool(ok),
        "criterion": ("mean<=10% and max<=15%" if label == "internal"
                      else "mean<=10%"),
    }


# ----------------------------------------------------------------------
def convolve_to_pk(t: np.ndarray, frac_abs: np.ndarray, dose: float,
                   F: float, V: float, ke: float,
                   t_out: np.ndarray | None = None) -> pd.DataFrame:
    """
    Convolve an absorption profile with one-compartment disposition to a
    plasma concentration-time curve.
    """
    t = np.asarray(t, float)
    fa = np.clip(np.asarray(frac_abs, float), 0, 1)
    t_out = t if t_out is None else np.asarray(t_out, float)

    rate = np.gradient(fa, t) * dose * F
    rate = np.maximum(rate, 0)

    cp = np.zeros(len(t_out))
    for i, ti in enumerate(t_out):
        m = t <= ti
        if not m.any():
            continue
        cp[i] = np.trapezoid(rate[m] * np.exp(-ke * (ti - t[m])), t[m]) / V
    return pd.DataFrame({"time": t_out, "Cp": cp})


def pk_metrics(t: np.ndarray, cp: np.ndarray) -> dict:
    """Cmax, Tmax, AUC0-t and AUC0-inf by the trapezoidal rule."""
    t = np.asarray(t, float)
    cp = np.asarray(cp, float)
    i = int(np.argmax(cp))
    auc_t = float(np.trapezoid(cp, t))

    lam, auc_inf = np.nan, auc_t
    tail = cp[i:] > 0
    if tail.sum() >= 3:
        tt, cc = t[i:][tail], cp[i:][tail]
        try:
            sl = stats.linregress(tt[-max(3, len(tt) // 2):],
                                  np.log(cc[-max(3, len(cc) // 2):]))
            if sl.slope < 0:
                lam = -float(sl.slope)
                auc_inf = auc_t + float(cp[-1] / lam)
        except Exception:
            pass
    return {"Cmax": float(cp[i]), "Tmax": float(t[i]),
            "AUC_0_t": auc_t, "AUC_0_inf": float(auc_inf),
            "lambda_z": float(lam)}
