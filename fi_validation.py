"""
Validation engine.

Rules enforced here (non-negotiable for small-n formulation data):
  * LOOCV when n < 30, repeated 5-fold (10 repeats) otherwise.
  * Ranking on Q2 (predicted R2) and RMSECV. Training R2 is computed for
    display only and is never used to rank.
  * 1-SE rule: among models within one standard error of the best Q2,
    the simplest wins.
  * Y-randomization: if the real Q2 is not clearly above the permuted
    null, the model is declared unusable regardless of its Q2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (GroupKFold, KFold, LeaveOneGroupOut,
                                     LeaveOneOut, RepeatedKFold)

def _needs_groups(cv) -> bool:
    """True for splitters whose split() requires a groups argument."""
    return isinstance(cv, (GroupKFold, LeaveOneGroupOut))


LOOCV_THRESHOLD = 30
DEFAULT_REPEATS = 10
PERMUTATION_N = 200
PERMUTATION_ALPHA = 0.05


@dataclass
class CVResult:
    name: str
    complexity: int
    q2: float
    rmsecv: float
    mae_cv: float
    q2_fold_sd: float
    q2_se: float
    r2_train: float
    y_true: np.ndarray = field(repr=False, default=None)
    y_pred: np.ndarray = field(repr=False, default=None)
    perm_p: float = np.nan
    perm_null_mean: float = np.nan
    perm_null_q95: float = np.nan
    signal: bool = False
    failed: str = ""


def make_cv(n: int, seed: int = 0, repeats: int = DEFAULT_REPEATS,
            groups=None):
    """
    LOOCV below the threshold, repeated K-fold above it.

    When `groups` is supplied the folds are drawn at the GROUP level.
    Random folds split a batch across train and test, so the model sees
    the batch offset during training and is scored on rows sharing it -
    the offset is then credited as predictive skill. That inflates Q2 in
    proportion to the ICC, and the inflation is invisible unless the
    folds respect the grouping. A model that cannot predict a batch it
    has never seen is not a model of the formulation.
    """
    if groups is not None:
        g = np.asarray(groups)
        n_g = len(np.unique(g))
        if n_g < 2:
            pass
        elif n_g <= 8:
            return LeaveOneGroupOut(), f"Leave-one-group-out ({n_g} groups)"
        else:
            k = min(5, n_g)
            return GroupKFold(n_splits=k), f"Group {k}-fold ({n_g} groups)"

    if n < LOOCV_THRESHOLD:
        return LeaveOneOut(), "LOOCV"
    k = 5 if n >= 25 else 3
    return (
        RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=seed),
        f"Repeated {k}-fold x{repeats}",
    )


def q2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Predicted R-squared. Uses the total sum of squares of the FULL
    response, which is what makes Q2 go negative for a useless model -
    the property that matters here.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-15:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def cross_validate_model(
    estimator, X: np.ndarray, y: np.ndarray, cv=None, seed: int = 0,
    groups=None
) -> tuple[np.ndarray, np.ndarray, list]:
    """Return (y_true_ordered, y_pred_oof, per_fold_q2)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    if cv is None:
        cv, _ = make_cv(n, seed, groups=groups)

    preds: dict[int, list[float]] = {i: [] for i in range(n)}
    fold_q2: list[float] = []

    splitter = (cv.split(X, y, np.asarray(groups))
                if groups is not None and _needs_groups(cv) else cv.split(X))
    for tr, te in splitter:
        est = clone(estimator)
        est.fit(X[tr], y[tr])
        p = np.asarray(est.predict(X[te]), float).ravel()
        for idx, val in zip(te, p):
            preds[int(idx)].append(float(val))
        if len(te) > 2:
            fold_q2.append(q2_score(y[te], p))

    y_pred = np.array([np.mean(preds[i]) if preds[i] else np.nan
                       for i in range(n)])
    return y, y_pred, fold_q2


def make_permutation_cv(n: int, seed: int = 0, groups=None):
    """
    Cheaper CV used only to build the permuted null distribution.

    Repeated K-fold is the right scheme for *ranking* models, but the
    null only needs an unbiased Q2 per shuffle, so a single pass is
    sufficient and ~10x cheaper. This is what makes the permutation test
    affordable for expensive estimators.
    """
    if groups is not None:
        return make_cv(n, seed, groups=groups)[0]
    if n < LOOCV_THRESHOLD:
        return LeaveOneOut()
    return KFold(n_splits=5, shuffle=True, random_state=seed)


def y_randomization(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    observed_q2: float,
    n_perm: int = PERMUTATION_N,
    seed: int = 0,
    cv=None,
    groups=None,
) -> tuple[float, float, float]:
    """
    Permutation test on the response.

    Returns (p_value, null_mean, null_q95). p is the fraction of permuted
    datasets whose cross-validated Q2 matches or beats the observed one,
    with the standard +1/+1 correction.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if cv is None:
        cv = make_permutation_cv(n, seed, groups=groups)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        try:
            yt, yhat, _ = cross_validate_model(estimator, X, yp, cv=cv,
                                               groups=groups)
            null.append(q2_score(yt, yhat))
        except Exception:
            null.append(-np.inf)
    null_arr = np.array(null, float)
    finite = null_arr[np.isfinite(null_arr)]
    p = (np.sum(null_arr >= observed_q2) + 1) / (n_perm + 1)
    return (
        float(p),
        float(finite.mean()) if finite.size else float("nan"),
        float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
    )


def evaluate(
    name: str,
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    complexity: int,
    seed: int = 0,
    run_permutation: bool = True,
    n_perm: int = PERMUTATION_N,
    groups=None,
) -> CVResult:
    """Full evaluation of one candidate model."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n = len(y)
    cv, _ = make_cv(n, seed, groups=groups)

    try:
        yt, yhat, fold_q2 = cross_validate_model(estimator, X, y, cv=cv,
                                                 groups=groups)
    except Exception as exc:
        return CVResult(name, complexity, -np.inf, np.inf, np.inf,
                        np.nan, np.nan, np.nan, failed=str(exc)[:200])

    q2 = q2_score(yt, yhat)
    resid = yt - yhat
    rmsecv = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))

    # Standard error of Q2: across folds when available, else jackknife
    if len(fold_q2) >= 3:
        sd = float(np.std(fold_q2, ddof=1))
        se = sd / np.sqrt(len(fold_q2))
    else:
        jack = [q2_score(np.delete(yt, i), np.delete(yhat, i))
                for i in range(n)]
        jack = np.array([j for j in jack if np.isfinite(j)])
        sd = float(np.std(jack, ddof=1)) if jack.size > 1 else np.nan
        se = (sd * np.sqrt(max(len(jack) - 1, 1)) / np.sqrt(len(jack))
              if jack.size > 1 else np.nan)

    try:
        est_full = clone(estimator).fit(X, y)
        r2_train = q2_score(y, np.asarray(est_full.predict(X)).ravel())
    except Exception:
        r2_train = np.nan

    res = CVResult(name, complexity, q2, rmsecv, mae, sd, se, r2_train,
                   y_true=yt, y_pred=yhat)

    if run_permutation and np.isfinite(q2):
        p, nm, nq = y_randomization(estimator, X, y, q2,
                                    n_perm=n_perm, seed=seed, groups=groups)
        res.perm_p, res.perm_null_mean, res.perm_null_q95 = p, nm, nq
        res.signal = bool(p < PERMUTATION_ALPHA and q2 > 0)
    else:
        res.signal = bool(q2 > 0)
    return res


def apply_one_se_rule(results: list[CVResult]) -> CVResult | None:
    """
    Among models whose Q2 is within one standard error of the best, return
    the one with the lowest complexity. Models that failed the
    permutation test are excluded entirely.
    """
    usable = [r for r in results
              if r.signal and np.isfinite(r.q2) and not r.failed]
    if not usable:
        return None
    best = max(usable, key=lambda r: r.q2)
    se = best.q2_se if np.isfinite(best.q2_se) else 0.0
    within = [r for r in usable if r.q2 >= best.q2 - se]
    return min(within, key=lambda r: (r.complexity, -r.q2))
