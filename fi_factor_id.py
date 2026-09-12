"""
Factor identification layer.

No single importance method is trustworthy when factors are correlated -
and formulation factors always are. This module runs several and reports
their AGREEMENT. A factor that only one method likes is a candidate for
further experiment, not a conclusion.

Methods:
  * VIP scores from PLS (>1 = influential), the chemometrics standard
  * Bootstrap regression coefficients with percentile CIs
  * Permutation importance with confidence bands
  * Stability selection: bootstrap Elastic Net, report selection frequency
  * Grouped importance for correlated clusters
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

STABILITY_THRESHOLD = 0.60
VIP_THRESHOLD = 1.0


# ----------------------------------------------------------------------
def vip_scores(X: np.ndarray, y: np.ndarray, n_components: int = 2
               ) -> np.ndarray:
    """
    Variable Importance in Projection. VIP > 1 is the conventional cut.
    Scaling is applied internally; do not pre-scale.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    k = int(np.clip(n_components, 1, min(X.shape[0] - 1, X.shape[1])))

    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    pls = PLSRegression(n_components=k, scale=False).fit(Xs, y)

    T, W, Q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p = X.shape[1]
    ssy = np.array([
        float((Q[0, a] ** 2) * (T[:, a] @ T[:, a])) for a in range(k)
    ])
    total = ssy.sum()
    if total < 1e-15:
        return np.ones(p)

    vip = np.zeros(p)
    for j in range(p):
        wj = np.array([
            (W[j, a] / (np.linalg.norm(W[:, a]) + 1e-300)) ** 2
            for a in range(k)
        ])
        vip[j] = np.sqrt(p * float(ssy @ wj) / total)
    return vip


def bootstrap_coefficients(X: np.ndarray, y: np.ndarray, n_boot: int = 1000,
                           seed: int = 0) -> pd.DataFrame:
    """Percentile CIs for standardised linear coefficients."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    rng = np.random.default_rng(seed)
    n = len(y)
    coefs = []
    base = Pipeline([("sc", StandardScaler()), ("m", LinearRegression())])
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(y[idx]) < 1e-12:
            continue
        try:
            m = clone(base).fit(X[idx], y[idx])
            coefs.append(m.named_steps["m"].coef_.ravel())
        except Exception:
            continue
    if not coefs:
        return pd.DataFrame()
    C = np.vstack(coefs)
    return pd.DataFrame({
        "coef_mean": C.mean(axis=0),
        "coef_lo95": np.percentile(C, 2.5, axis=0),
        "coef_hi95": np.percentile(C, 97.5, axis=0),
        "sign_consistency": np.maximum((C > 0).mean(axis=0),
                                       (C < 0).mean(axis=0)),
    })


def permutation_importance_cv(estimator, X: np.ndarray, y: np.ndarray,
                              n_repeats: int = 30, seed: int = 0
                              ) -> pd.DataFrame:
    """
    Permutation importance measured as the drop in Q2 when a column is
    shuffled, with a bootstrap band across repeats.
    """
    from fi_validation import cross_validate_model, q2_score

    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    rng = np.random.default_rng(seed)

    yt, yhat, _ = cross_validate_model(estimator, X, y)
    base = q2_score(yt, yhat)

    rows = []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            try:
                yt2, yh2, _ = cross_validate_model(estimator, Xp, y)
                drops.append(base - q2_score(yt2, yh2))
            except Exception:
                continue
        d = np.array(drops) if drops else np.array([np.nan])
        rows.append({
            "importance": float(np.nanmean(d)),
            "imp_lo": float(np.nanpercentile(d, 5)) if drops else np.nan,
            "imp_hi": float(np.nanpercentile(d, 95)) if drops else np.nan,
        })
    return pd.DataFrame(rows)


def stability_selection(X: np.ndarray, y: np.ndarray, n_boot: int = 500,
                        l1_ratio: float = 0.8, alpha: float | None = None,
                        sample_frac: float = 0.75, seed: int = 0
                        ) -> np.ndarray:
    """
    Bootstrap Elastic Net; return per-factor selection frequency.

    Far more trustworthy than a single fit: a factor selected in 90% of
    subsamples is real, one selected in 40% is a coin flip regardless of
    how large its coefficient looks in the full-data fit.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, p = X.shape
    rng = np.random.default_rng(seed)

    Xs = StandardScaler().fit_transform(X)
    if alpha is None:
        alpha = float(np.max(np.abs(Xs.T @ (y - y.mean()))) / n) * 0.30

    m = max(int(sample_frac * n), 3)
    counts = np.zeros(p)
    used = 0
    for _ in range(n_boot):
        idx = rng.choice(n, m, replace=False)
        if np.std(y[idx]) < 1e-12:
            continue
        try:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                            max_iter=10000).fit(Xs[idx], y[idx])
            counts += (np.abs(en.coef_) > 1e-8).astype(float)
            used += 1
        except Exception:
            continue
    return counts / max(used, 1)


def correlation_clusters(X: pd.DataFrame, threshold: float = 0.8
                         ) -> dict:
    """
    Cluster factors by absolute Spearman correlation. Correlated factors
    (e.g. polymer grade and viscosity) must be reported as a GROUP -
    splitting their importance between them is meaningless.
    """
    if X.shape[1] < 2:
        return {c: 0 for c in X.columns}
    corr = np.array(X.corr(method="spearman").abs().fillna(0).to_numpy(),
                    dtype=float, copy=True)
    np.fill_diagonal(corr, 1.0)
    dist = np.clip(1.0 - corr, 0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    try:
        Z = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(Z, t=1 - threshold, criterion="distance")
    except Exception:
        labels = np.arange(1, X.shape[1] + 1)
    return dict(zip(X.columns, labels.astype(int)))


def consensus(X: pd.DataFrame, y: pd.Series, estimator=None,
              n_components: int = 2, seed: int = 0,
              run_permutation: bool = True) -> pd.DataFrame:
    """
    Run every method and return one table with a consensus vote.

    `votes` counts how many independent methods flag the factor. A factor
    with 3-4 votes is solid; 1 vote means design a better experiment.
    """
    Xn = X.select_dtypes(include=[np.number])
    names = list(Xn.columns)
    Xv = np.array(Xn.to_numpy(float), copy=True)
    yv = np.array(y, dtype=float).ravel()

    out = pd.DataFrame(index=names)
    out["cluster"] = pd.Series(correlation_clusters(Xn))

    try:
        out["VIP"] = vip_scores(Xv, yv, n_components)
    except Exception:
        out["VIP"] = np.nan

    bc = bootstrap_coefficients(Xv, yv, seed=seed)
    if not bc.empty:
        bc.index = names
        out = out.join(bc)

    try:
        out["stability"] = stability_selection(Xv, yv, seed=seed)
    except Exception:
        out["stability"] = np.nan

    if estimator is not None and run_permutation:
        try:
            pi = permutation_importance_cv(estimator, Xv, yv, seed=seed)
            pi.index = names
            out = out.join(pi)
        except Exception:
            pass

    if estimator is not None and HAS_SHAP:
        try:
            est = clone(estimator).fit(Xv, yv)
            bg = shap.sample(Xv, min(50, len(Xv)), random_state=seed)
            ex = shap.KernelExplainer(
                lambda z: np.asarray(est.predict(z)).ravel(), bg)
            sv = ex.shap_values(Xv, nsamples=100, silent=True)
            out["SHAP_mean_abs"] = np.abs(np.asarray(sv)).mean(axis=0)
        except Exception:
            pass

    votes = np.zeros(len(names))
    if "VIP" in out:
        votes += (out["VIP"].fillna(0) > VIP_THRESHOLD).to_numpy()
    if "stability" in out:
        votes += (out["stability"].fillna(0) > STABILITY_THRESHOLD).to_numpy()
    if {"coef_lo95", "coef_hi95"}.issubset(out.columns):
        votes += ((out["coef_lo95"] > 0) | (out["coef_hi95"] < 0)).to_numpy()
    if "imp_lo" in out:
        votes += (out["imp_lo"].fillna(-1) > 0).to_numpy()
    out["votes"] = votes.astype(int)

    n_methods = sum(c in out.columns for c in
                    ["VIP", "stability", "coef_lo95", "imp_lo"])
    out["verdict"] = [
        "DRIVER" if v >= max(n_methods - 1, 1) else
        ("candidate" if v >= 1 else "no evidence")
        for v in out["votes"]
    ]
    out["n_methods"] = n_methods
    return out.sort_values("votes", ascending=False)
