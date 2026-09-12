"""
Model zoo. Candidate sets are chosen by data regime, not by trying
everything:

  Structured DoE, n<20      -> OLS with 2FI / quadratic + Lenth's method
  n 20-60, collinear X      -> PLS regression (VIP for factor ID)
  n 20-80, nonlinear smooth -> Gaussian Process (Matern 5/2), gives PIs
  n 60-300                  -> Random Forest / gradient boosting
  Mixture components        -> Scheffe canonical, no intercept
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

try:  # optional
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


# ----------------------------------------------------------------------
# Scheffe canonical mixture models
# ----------------------------------------------------------------------
class ScheffeMixture(BaseEstimator, RegressorMixin):
    """
    Scheffe canonical polynomial for mixture components. No intercept:
    with sum(x)=1 the intercept is not identifiable and including one
    makes the ordinary regression invalid.

      degree 1 : sum bi*xi
      degree 2 : sum bi*xi + sum_{i<j} bij*xi*xj
      degree 3 : ... + sum_{i<j<k} bijk*xi*xj*xk  (special cubic)

    Process variables (e.g. hardness) may be passed via `process_idx`;
    they enter linearly and crossed with the linear blend terms.
    """

    def __init__(self, degree: int = 2, process_idx: tuple = (),
                 ridge: float = 1e-8):
        self.degree = degree
        self.process_idx = process_idx
        self.ridge = ridge

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        pidx = list(self.process_idx)
        midx = [j for j in range(X.shape[1]) if j not in pidx]
        M, P = X[:, midx], X[:, pidx]

        cols = [M]  # linear blend terms
        if self.degree >= 2:
            cols.append(
                np.column_stack(
                    [M[:, i] * M[:, j]
                     for i in range(M.shape[1])
                     for j in range(i + 1, M.shape[1])]
                ) if M.shape[1] >= 2 else np.empty((len(M), 0))
            )
        if self.degree >= 3 and M.shape[1] >= 3:
            cols.append(
                np.column_stack(
                    [M[:, i] * M[:, j] * M[:, k]
                     for i in range(M.shape[1])
                     for j in range(i + 1, M.shape[1])
                     for k in range(j + 1, M.shape[1])]
                )
            )
        if P.shape[1]:
            cols.append(P)
            cols.append(
                np.column_stack(
                    [M[:, i] * P[:, q]
                     for i in range(M.shape[1])
                     for q in range(P.shape[1])]
                )
            )
        return np.column_stack([c for c in cols if c.size or c.shape[0]])

    def fit(self, X, y):
        A = self._design(X)
        y = np.asarray(y, float).ravel()
        # small ridge for numerical stability only
        G = A.T @ A + self.ridge * np.eye(A.shape[1])
        self.coef_ = np.linalg.solve(G, A.T @ y)
        self.n_terms_ = A.shape[1]
        return self

    def predict(self, X):
        return self._design(X) @ self.coef_


# ----------------------------------------------------------------------
# Lenth's method for unreplicated factorial designs
# ----------------------------------------------------------------------
def lenth_effects(effects: np.ndarray, names: list, alpha: float = 0.05
                  ) -> list:
    """
    Lenth's pseudo standard error for unreplicated designs.
    Returns list of dicts with effect, PSE, margin of error and
    significance at the individual (ME) and simultaneous (SME) levels.
    """
    e = np.asarray(effects, float)
    m = len(e)
    s0 = 1.5 * np.median(np.abs(e))
    keep = np.abs(e) < 2.5 * s0
    pse = 1.5 * np.median(np.abs(e[keep])) if keep.any() else s0
    dof = max(m // 3, 1)

    from scipy import stats

    t_me = stats.t.ppf(1 - alpha / 2, dof)
    gamma = (1 + (1 - alpha) ** (1 / m)) / 2
    t_sme = stats.t.ppf(gamma, dof)
    me, sme = t_me * pse, t_sme * pse

    return [
        {
            "factor": n,
            "effect": float(v),
            "abs_effect": float(abs(v)),
            "PSE": float(pse),
            "ME": float(me),
            "SME": float(sme),
            "significant_ME": bool(abs(v) > me),
            "significant_SME": bool(abs(v) > sme),
        }
        for n, v in zip(names, e)
    ]


# ----------------------------------------------------------------------
# Candidate builders
# ----------------------------------------------------------------------
def _pls(n_comp: int) -> Pipeline:
    return Pipeline([
        ("sc", StandardScaler()),
        ("pls", PLSRegression(n_components=n_comp, scale=False)),
    ])


def _gpr(nu: float = 2.5) -> Pipeline:
    kernel = (ConstantKernel(1.0, (1e-3, 1e3))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e3), nu=nu)
              + WhiteKernel(1e-3, (1e-8, 1e1)))
    return Pipeline([
        ("sc", StandardScaler()),
        ("gp", GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                        n_restarts_optimizer=3,
                                        random_state=0)),
    ])


def build_candidates(
    n: int,
    p: int,
    mixture_idx: list | None = None,
    process_idx: tuple = (),
    seed: int = 0,
) -> list[tuple[str, object, int]]:
    """
    Return [(name, estimator, complexity), ...] appropriate to the regime.
    `complexity` drives the 1-SE rule: lower is simpler.
    """
    C: list[tuple[str, object, int]] = []

    # Always available baselines
    C.append(("OLS linear", Pipeline([("sc", StandardScaler()),
                                      ("m", LinearRegression())]), 10))
    C.append(("Ridge (CV alpha)",
              Pipeline([("sc", StandardScaler()),
                        ("m", RidgeCV(alphas=np.logspace(-4, 3, 40)))]), 9))

    if mixture_idx:
        C.append(("Scheffe linear", ScheffeMixture(1, process_idx), 8))
        if n >= 10:
            C.append(("Scheffe quadratic", ScheffeMixture(2, process_idx), 14))
        if n >= 20 and len(mixture_idx) >= 3:
            C.append(("Scheffe special cubic",
                      ScheffeMixture(3, process_idx), 20))

    if n < 20:
        C.append(("OLS + 2FI",
                  Pipeline([("pf", PolynomialFeatures(2, interaction_only=True,
                                                      include_bias=False)),
                            ("sc", StandardScaler()),
                            ("m", RidgeCV(alphas=np.logspace(-4, 3, 40)))]), 16))

    if n >= 12:
        C.append(("ElasticNet (CV)",
                  Pipeline([("sc", StandardScaler()),
                            ("m", ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                               cv=3, random_state=seed,
                                               max_iter=20000))]), 11))
    if n >= 12:
        for k in range(1, min(p, max(n // 4, 1), 6) + 1):
            C.append((f"PLS ({k} comp)", _pls(k), 10 + 2 * k))

    if 15 <= n <= 300:
        C.append(("GPR Matern 5/2", _gpr(2.5), 25))
        C.append(("GPR Matern 3/2", _gpr(1.5), 26))

    if n >= 40:
        C.append(("Random Forest",
                  RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                        random_state=seed, n_jobs=-1), 40))
        if HAS_XGB:
            C.append(("XGBoost",
                      XGBRegressor(n_estimators=400, max_depth=3,
                                   learning_rate=0.05, subsample=0.8,
                                   colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=seed, verbosity=0), 45))
    if n >= 20:
        C.append(("Quadratic + Ridge",
                  Pipeline([("pf", PolynomialFeatures(2, include_bias=False)),
                            ("sc", StandardScaler()),
                            ("m", RidgeCV(alphas=np.logspace(-4, 4, 50)))]), 22))
    return C


def regime_label(n: int, mixture: bool) -> str:
    if mixture:
        return "Mixture design - Scheffe canonical models take priority"
    if n < 20:
        return "Structured DoE regime (n<20): OLS + 2FI, Lenth's method"
    if n <= 60:
        return "Collinear regime (n 20-60): PLS is the workhorse"
    if n <= 80:
        return "Smooth-nonlinear regime: GPR gives prediction intervals"
    if n <= 300:
        return "Tree regime (n 60-300): thresholds and interactions"
    return "Large-n regime: trees and boosting"
