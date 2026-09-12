"""
Applicability domain.

A prediction outside the training hull is an extrapolation, and at these
sample sizes extrapolation is where models fail silently. Every
prediction returned by this package carries an AD flag; points outside
the domain get a warning instead of a number treated as reliable.

Three complementary criteria:
  * Leverage (Williams plot h*) - classic QSAR/chemometrics rule
  * Hotelling's T2 on PCA scores - the PLS/multivariate standard
  * Range check - a blunt but honest per-factor bound
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ApplicabilityDomain:
    """Fit on the training design; query any candidate formulation."""

    def __init__(self, alpha: float = 0.05, n_components: int | None = None):
        self.alpha = alpha
        self.n_components = n_components

    def fit(self, X: np.ndarray, feature_names: list | None = None):
        X = np.asarray(X, float)
        self.n_, self.p_ = X.shape
        self.feature_names_ = feature_names or [f"x{i}" for i in range(self.p_)]

        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)

        # leverage threshold h* = 3p/n (Williams plot convention)
        self.h_star_ = 3.0 * self.p_ / self.n_
        try:
            self.XtX_inv_ = np.linalg.pinv(Z.T @ Z)
        except np.linalg.LinAlgError:
            self.XtX_inv_ = np.linalg.pinv(Z.T @ Z + 1e-8 * np.eye(self.p_))

        k = self.n_components or max(
            1, min(self.p_, max(self.n_ - 1, 1), 3)
        )
        k = min(k, min(Z.shape))
        self.pca_ = PCA(n_components=k).fit(Z)
        T = self.pca_.transform(Z)
        self.t_var_ = T.var(axis=0, ddof=1) + 1e-12
        self.k_ = k

        # Hotelling T2 critical value
        if self.n_ > k:
            f_crit = stats.f.ppf(1 - self.alpha, k, self.n_ - k)
            self.t2_crit_ = (k * (self.n_ - 1) / (self.n_ - k)) * f_crit
        else:
            self.t2_crit_ = np.inf

        self.min_, self.max_ = X.min(axis=0), X.max(axis=0)
        self.range_ = np.where((self.max_ - self.min_) > 0,
                               self.max_ - self.min_, 1.0)
        return self

    def check(self, X_new: np.ndarray) -> list[dict]:
        X_new = np.atleast_2d(np.asarray(X_new, float))
        Z = self.scaler_.transform(X_new)
        T = self.pca_.transform(Z)

        out = []
        for i in range(len(X_new)):
            z = Z[i]
            h = float(z @ self.XtX_inv_ @ z)
            t2 = float(np.sum(T[i] ** 2 / self.t_var_))

            below = X_new[i] < self.min_
            above = X_new[i] > self.max_
            oor = [
                {
                    "factor": self.feature_names_[j],
                    "value": float(X_new[i, j]),
                    "train_min": float(self.min_[j]),
                    "train_max": float(self.max_[j]),
                    "excess_pct": float(
                        100 * (X_new[i, j] - self.max_[j]) / self.range_[j]
                        if above[j] else
                        100 * (self.min_[j] - X_new[i, j]) / self.range_[j]
                    ),
                }
                for j in range(self.p_) if below[j] or above[j]
            ]

            fails = []
            if h > self.h_star_:
                fails.append(f"leverage {h:.3f} > h*={self.h_star_:.3f}")
            if t2 > self.t2_crit_:
                fails.append(f"Hotelling T2 {t2:.2f} > {self.t2_crit_:.2f}")
            if oor:
                fails.append(
                    "outside training range: "
                    + ", ".join(f"{d['factor']}" for d in oor)
                )

            out.append({
                "inside": not fails,
                "leverage": h,
                "h_star": self.h_star_,
                "T2": t2,
                "T2_crit": self.t2_crit_,
                "out_of_range": oor,
                "reasons": fails,
                "verdict": ("INSIDE domain" if not fails else
                            "EXTRAPOLATION - " + "; ".join(fails)),
            })
        return out


def gp_predictive_interval(estimator, X_new: np.ndarray, level: float = 0.95):
    """
    Prediction interval from a fitted GP pipeline. Returns (mean, lo, hi)
    or (mean, None, None) if the estimator cannot express uncertainty -
    in which case the caller must not display an interval.
    """
    X_new = np.atleast_2d(np.asarray(X_new, float))
    z = stats.norm.ppf(0.5 + level / 2)

    inner = estimator
    if hasattr(estimator, "named_steps"):
        for step in estimator.named_steps.values():
            if hasattr(step, "predict") and "return_std" in getattr(
                step.predict, "__doc__", "") or "":
                pass
        try:
            Xt = X_new
            steps = list(estimator.named_steps.items())
            for _, st in steps[:-1]:
                Xt = st.transform(Xt)
            inner, X_new = steps[-1][1], Xt
        except Exception:
            return np.asarray(estimator.predict(X_new)).ravel(), None, None

    try:
        mu, sd = inner.predict(X_new, return_std=True)
        mu = np.asarray(mu).ravel()
        sd = np.asarray(sd).ravel()
        return mu, mu - z * sd, mu + z * sd
    except (TypeError, AttributeError):
        return np.asarray(inner.predict(X_new)).ravel(), None, None
