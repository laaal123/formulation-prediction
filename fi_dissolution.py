"""
Dissolution profile handling.

Never model each timepoint independently. Three routes, best first:
  1. Parameterise then predict - fit Weibull / Korsmeyer-Peppas per
     formulation, model the 2-3 parameters against the factors, then
     reconstruct the profile. Preserves monotonicity, needs far less data.
  2. Functional PCA on the profile matrix, model the first 2-3 scores.
  3. Multi-output PLS (PLS2) as a quick baseline.

Also here: bootstrap f2 (the point estimate is unstable and should never
be reported alone) and a DISCRIMINATION POWER test - the tool that tells
you whether a method can see a difference you know exists in vivo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


# ----------------------------------------------------------------------
# Model library
# ----------------------------------------------------------------------
def weibull(t, alpha, beta, fmax=100.0):
    """F = fmax * (1 - exp(-t^beta / alpha))"""
    t = np.maximum(np.asarray(t, float), 0)
    return fmax * (1.0 - np.exp(-np.power(t, beta) / max(alpha, 1e-9)))


def korsmeyer_peppas(t, k, n):
    """F = k * t^n. n indicates the release mechanism."""
    return k * np.power(np.maximum(np.asarray(t, float), 1e-12), n)


def first_order(t, k, fmax=100.0):
    return fmax * (1.0 - np.exp(-k * np.asarray(t, float)))


def higuchi(t, k):
    return k * np.sqrt(np.maximum(np.asarray(t, float), 0))


def hixson_crowell(t, k, fmax=100.0):
    x = np.clip(1.0 - k * np.asarray(t, float), 0, 1)
    return fmax * (1.0 - x ** 3)


def peppas_sahlin(t, k1, k2, m):
    """
    F = k1*t^m + k2*t^(2m). Separates Fickian diffusion (k1) from
    Case-II relaxation / erosion (k2) - the erosion term is often the
    real factor-response variable for a matrix tablet.
    """
    t = np.maximum(np.asarray(t, float), 1e-12)
    return k1 * t ** m + k2 * t ** (2 * m)


MODEL_LIB = {
    "Weibull": (weibull, ["alpha", "beta"], [50.0, 1.0],
                ([1e-6, 0.1], [1e6, 5.0])),
    "Korsmeyer-Peppas": (korsmeyer_peppas, ["k", "n"], [10.0, 0.5],
                         ([1e-6, 0.05], [1e4, 2.0])),
    "First-order": (first_order, ["k"], [0.05], ([1e-8], [10.0])),
    "Higuchi": (higuchi, ["k"], [10.0], ([1e-8], [1e4])),
    "Hixson-Crowell": (hixson_crowell, ["k"], [0.01], ([1e-9], [10.0])),
    "Peppas-Sahlin": (peppas_sahlin, ["k1", "k2", "m"], [5.0, 0.5, 0.5],
                      ([-1e3, -1e3, 0.05], [1e3, 1e3, 1.5])),
}


@dataclass
class ProfileFit:
    model: str
    params: dict
    r2: float
    adj_r2: float
    aic: float
    rmse: float
    n_points: int
    converged: bool


def fit_profile(t: np.ndarray, f: np.ndarray, model: str = "Weibull"
                ) -> ProfileFit:
    """Fit one release model to one profile."""
    fn, pnames, p0, bounds = MODEL_LIB[model]
    t = np.asarray(t, float)
    f = np.asarray(f, float)
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    k = len(pnames)

    try:
        popt, _ = curve_fit(fn, t, f, p0=p0, bounds=bounds, maxfev=20000)
        pred = fn(t, *popt)
        conv = True
    except Exception:
        popt = np.array(p0, float)
        pred = fn(t, *popt)
        conv = False

    resid = f - pred
    ss_res = float(resid @ resid)
    ss_tot = float(((f - f.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan
    n = len(t)
    adj = (1 - (1 - r2) * (n - 1) / (n - k - 1)
           if n > k + 1 and np.isfinite(r2) else np.nan)
    aic = n * np.log(max(ss_res / n, 1e-300)) + 2 * k
    return ProfileFit(model, dict(zip(pnames, map(float, popt))),
                      float(r2), float(adj), float(aic),
                      float(np.sqrt(ss_res / n)), n, conv)


def best_model(t, f, models: list | None = None) -> pd.DataFrame:
    """Rank release models by AIC for a single profile."""
    models = models or list(MODEL_LIB)
    rows = []
    for m in models:
        fit = fit_profile(t, f, m)
        rows.append({"model": m, "R2": fit.r2, "adj_R2": fit.adj_r2,
                     "AIC": fit.aic, "RMSE": fit.rmse,
                     "converged": fit.converged, **fit.params})
    return pd.DataFrame(rows).sort_values("AIC")


def parameterise_batch(profiles: pd.DataFrame, t: np.ndarray,
                       model: str = "Weibull") -> pd.DataFrame:
    """
    Route 1. profiles: rows = formulations, columns = timepoints.
    Returns one row of model parameters per formulation - these become
    the response variables for the model arena.
    """
    rows = []
    for idx, row in profiles.iterrows():
        fit = fit_profile(t, row.to_numpy(float), model)
        rows.append({"formulation": idx, **fit.params,
                     "R2": fit.r2, "converged": fit.converged})
    return pd.DataFrame(rows).set_index("formulation")


def functional_pca(profiles: pd.DataFrame, n_components: int = 3):
    """Route 2. Returns (scores, pca_object, explained_variance_ratio)."""
    M = profiles.to_numpy(float)
    k = int(np.clip(n_components, 1, min(M.shape) - 1 if min(M.shape) > 1 else 1))
    pca = PCA(n_components=k).fit(M)
    scores = pd.DataFrame(
        pca.transform(M),
        index=profiles.index,
        columns=[f"PC{i+1}" for i in range(k)],
    )
    return scores, pca, pca.explained_variance_ratio_


def reconstruct(params: dict, t: np.ndarray, model: str = "Weibull"
                ) -> np.ndarray:
    fn, pnames, *_ = MODEL_LIB[model]
    return fn(t, *[params[p] for p in pnames])


# ----------------------------------------------------------------------
# f2 with a confidence interval
# ----------------------------------------------------------------------
def f2_similarity(ref: np.ndarray, test: np.ndarray) -> float:
    ref, test = np.asarray(ref, float), np.asarray(test, float)
    d = float(np.mean((ref - test) ** 2))
    return float(50 * np.log10(100 / np.sqrt(1 + d)))


def bootstrap_f2(ref_units: np.ndarray, test_units: np.ndarray,
                 n_boot: int = 5000, seed: int = 0) -> dict:
    """
    Bootstrap f2 over individual vessels.

    The f2 point estimate is statistically unstable; regulators
    increasingly expect the lower bound of the 90% CI. Report that.
    """
    rng = np.random.default_rng(seed)
    R = np.atleast_2d(np.asarray(ref_units, float))
    T = np.atleast_2d(np.asarray(test_units, float))
    obs = f2_similarity(R.mean(axis=0), T.mean(axis=0))

    vals = []
    for _ in range(n_boot):
        rb = R[rng.integers(0, len(R), len(R))]
        tb = T[rng.integers(0, len(T), len(T))]
        vals.append(f2_similarity(rb.mean(axis=0), tb.mean(axis=0)))
    v = np.array(vals)
    lo = float(np.percentile(v, 5))
    return {
        "f2_point": obs,
        "f2_lower_90CI": lo,
        "f2_upper_90CI": float(np.percentile(v, 95)),
        "similar_by_point": obs >= 50,
        "similar_by_lower_bound": lo >= 50,
        "verdict": ("similar (lower bound >= 50)" if lo >= 50 else
                    "NOT demonstrated - point estimate passes but the "
                    "lower 90% bound does not" if obs >= 50 else
                    "not similar"),
    }


# ----------------------------------------------------------------------
# Discrimination power - the module for "my method cannot tell them apart"
# ----------------------------------------------------------------------
def discrimination_power(profiles: dict, known_rank: list,
                         timepoints: np.ndarray | None = None) -> dict:
    """
    Does this dissolution method reproduce a known in vivo rank order?

    profiles   : {batch_id: 2-D array (vessels x timepoints)}
    known_rank : batch ids ordered fastest -> slowest in vivo

    A method that cannot rank-order batches you KNOW differ in vivo is
    not discriminating, and no amount of modelling will rescue it.
    Returns Spearman rho against the known order plus the timepoint
    where separation is largest and the between/within variance ratio.
    """
    from scipy import stats

    ids = [b for b in known_rank if b in profiles]
    if len(ids) < 3:
        return {"error": "need at least 3 batches present in known_rank"}

    means = {b: np.atleast_2d(profiles[b]).mean(axis=0) for b in ids}
    nt = len(next(iter(means.values())))
    t = np.arange(nt) if timepoints is None else np.asarray(timepoints)

    # in vitro speed proxy: mean cumulative release (higher = faster)
    speed = np.array([means[b].mean() for b in ids])
    invivo = np.arange(len(ids), 0, -1)          # fastest gets highest
    rho, pval = stats.spearmanr(speed, invivo)

    # separation and signal-to-noise per timepoint
    per_tp = []
    for j in range(nt):
        vals = np.array([means[b][j] for b in ids])
        within = np.mean([
            np.atleast_2d(profiles[b])[:, j].std(ddof=1)
            if np.atleast_2d(profiles[b]).shape[0] > 1 else 0.0
            for b in ids
        ])
        spread = float(vals.max() - vals.min())
        per_tp.append({
            "timepoint": float(t[j]),
            "spread_pct": spread,
            "mean_within_SD": float(within),
            "signal_to_noise": float(spread / within) if within > 1e-9 else np.inf,
        })
    tp = pd.DataFrame(per_tp)
    best = tp.loc[tp["spread_pct"].idxmax()]

    f2_pairs = {
        f"{ids[i]} vs {ids[j]}": f2_similarity(means[ids[i]], means[ids[j]])
        for i in range(len(ids)) for j in range(i + 1, len(ids))
    }
    max_spread = float(tp["spread_pct"].max())
    # The extremes of the known in vivo order must be dissimilar. Requiring
    # every pair to differ is too strict: adjacent batches can legitimately
    # be similar while the method still ranks the series correctly.
    extreme_pair = f"{ids[0]} vs {ids[-1]}"
    f2_extremes = f2_pairs.get(extreme_pair, np.nan)
    discriminating = bool(rho >= 0.9 and max_spread >= 10 and
                          np.isfinite(f2_extremes) and f2_extremes < 50)

    return {
        "spearman_rho_vs_invivo": float(rho),
        "spearman_p": float(pval),
        "max_spread_pct": max_spread,
        "best_timepoint": float(best["timepoint"]),
        "best_timepoint_SN": float(best["signal_to_noise"]),
        "pairwise_f2": f2_pairs,
        "f2_extremes": float(f2_extremes),
        "extreme_pair": extreme_pair,
        "per_timepoint": tp,
        "discriminating": discriminating,
        "verdict": (
            "DISCRIMINATING - reproduces the in vivo rank order"
            if discriminating else
            "NOT DISCRIMINATING - this method cannot see the difference "
            "that exists in vivo. Change the method (ionic strength, "
            "early sampling, mechanical stress), not the model."
        ),
    }


def rsd_by_timepoint(profiles: dict) -> pd.DataFrame:
    """
    Vessel-to-vessel %RSD per timepoint, per batch.

    A matrix sitting near its percolation threshold has an unreliable
    pore network and should show ELEVATED scatter even when its mean
    profile is superimposable. f2 on mean profiles throws this away by
    construction, so check it separately.
    """
    rows = []
    for b, arr in profiles.items():
        A = np.atleast_2d(np.asarray(arr, float))
        if A.shape[0] < 2:
            continue
        m, s = A.mean(axis=0), A.std(axis=0, ddof=1)
        for j in range(A.shape[1]):
            rows.append({
                "batch": b, "timepoint_index": j,
                "mean": float(m[j]), "SD": float(s[j]),
                "RSD_pct": float(100 * s[j] / m[j]) if m[j] > 1e-9 else np.nan,
            })
    return pd.DataFrame(rows)
