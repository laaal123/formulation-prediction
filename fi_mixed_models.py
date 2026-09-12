"""
Mixed-effects models for batch / site / campaign structure.

Pooling data across manufacturing campaigns and treating it as one flat
dataset is the quiet error in formulation analytics. If batches differ
systematically, the observations within a batch are correlated, the
pooled standard errors are too small, and the EFFECTIVE sample size is
smaller than the row count.

General model:

    y = X.beta + sum_k Z_k.u_k + e,
    u_k ~ N(0, s2_k I),   e ~ N(0, s2e I)

Any number of random terms, so grouping factors may be CROSSED (site and
campaign, neither nested in the other) as well as nested, and a term may
carry a slope column instead of an intercept. The likelihood is profiled
over s2e and optimised over the variance ratios lambda_k = s2_k/s2e.
Woodbury keeps the cost tied to the number of random effects rather than
the number of rows:

    (I + Z D Z')^-1  = I - Z (D^-1 + Z'Z)^-1 Z'
    log|I + Z D Z'|  = log|D| + log|D^-1 + Z'Z|

Random effects are INDEPENDENT: a random intercept and a random slope on
the same grouping factor each get their own variance, but their
correlation is not estimated. This is lme4's `(1|g) + (0+x|g)`, not
`(x|g)`. Estimating the correlation needs more groups than formulation
work usually provides, and a badly estimated correlation does more
damage than omitting it.

Denominator degrees of freedom use Satterthwaite's approximation, which
matters here: a covariate that varies BETWEEN groups has roughly as many
degrees of freedom as there are groups, not as there are rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

HIGH_ICC = 0.10
MIN_GROUPS_FOR_LMM = 3
LOG_LAMBDA_BOUNDS = (-14.0, 9.0)


# ----------------------------------------------------------------------
@dataclass
class RandomTerm:
    """One random effect: a grouping factor, optionally with a slope."""

    groups: np.ndarray
    name: str = "group"
    slope: np.ndarray | None = None
    slope_name: str | None = None

    def build(self, n: int):
        g = np.asarray(self.groups)
        levels, codes = np.unique(g, return_inverse=True)
        q = len(levels)
        vals = (np.ones(n) if self.slope is None
                else np.asarray(self.slope, float).ravel())
        Z = np.zeros((n, q))
        Z[np.arange(n), codes] = vals
        return Z, q, levels, np.bincount(codes, minlength=q)

    @property
    def label(self) -> str:
        return (self.name if self.slope is None
                else f"{self.slope_name or 'slope'} | {self.name}")


@dataclass
class LMMFit:
    beta: np.ndarray
    se: np.ndarray
    feature_names: list
    sigma2_u: float
    sigma2_e: float
    icc: float
    lambda_: float
    n_obs: int
    n_groups: int
    group_sizes: np.ndarray
    dof: int
    reml_loglik: float
    converged: bool = True
    warnings: list = field(default_factory=list)
    variance_components: dict = field(default_factory=dict)
    dof_satterthwaite: np.ndarray | None = None
    term_labels: list = field(default_factory=list)

    @property
    def mean_group_size(self) -> float:
        return float(np.mean(self.group_sizes))

    @property
    def design_effect(self) -> float:
        m = np.asarray(self.group_sizes, float)
        m_eff = float((m ** 2).sum() / m.sum()) if m.sum() else 1.0
        return float(1.0 + (m_eff - 1.0) * self.icc)

    @property
    def effective_n(self) -> float:
        return float(self.n_obs / max(self.design_effect, 1e-9))

    def table(self) -> pd.DataFrame:
        dof = (self.dof_satterthwaite
               if self.dof_satterthwaite is not None
               else np.full(len(self.beta), float(self.dof)))
        dof = np.clip(np.nan_to_num(np.asarray(dof, float),
                                    nan=float(self.dof)), 1.0, 1e6)
        t = self.beta / np.where(self.se > 0, self.se, np.nan)
        p = 2 * stats.t.sf(np.abs(t), dof)
        crit = stats.t.ppf(0.975, dof)
        return pd.DataFrame({
            "term": self.feature_names,
            "estimate": self.beta,
            "std_error": self.se,
            "df_satterthwaite": dof,
            "t": t,
            "p_value": p,
            "ci_lo95": self.beta - crit * self.se,
            "ci_hi95": self.beta + crit * self.se,
        })

    def summary(self) -> str:
        lines = [
            f"Observations        : {self.n_obs}",
            f"Groups              : {self.n_groups} "
            f"(sizes {int(np.min(self.group_sizes))}-"
            f"{int(np.max(self.group_sizes))})",
        ]
        for k, v in self.variance_components.items():
            lines.append(f"  var[{k:<16}]: {v:.6g}")
        lines += [
            f"Residual var        : {self.sigma2_e:.6g}",
            f"Total random var    : {self.sigma2_u:.6g}",
            f"ICC                 : {self.icc:.4f}",
            f"Design effect       : {self.design_effect:.3f}",
            f"Effective n         : {self.effective_n:.1f}  "
            f"(vs {self.n_obs} rows)",
            f"REML log-likelihood : {self.reml_loglik:.4f}",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
def _core(loglam: np.ndarray, X: np.ndarray, y: np.ndarray,
          Zs: list, qs: list):
    """
    Everything that depends only on the variance ratios.

    Returns (beta, rHr, logdet_H, logdet_XtHiX, XtHiX) where H = V/s2e.
    """
    n, p = X.shape
    loglam = np.asarray(loglam, float)
    lam = np.exp(np.clip(loglam, *LOG_LAMBDA_BOUNDS))

    if not Zs:
        XtX = X.T @ X
        beta = np.linalg.lstsq(XtX, X.T @ y, rcond=None)[0]
        r = y - X @ beta
        _s, ld = np.linalg.slogdet(XtX)
        return beta, max(float(r @ r), 1e-300), 0.0, float(ld), XtX

    Z = np.hstack(Zs)
    dinv = np.concatenate([np.full(q, 1.0 / l) for q, l in zip(qs, lam)])
    M = np.diag(dinv) + Z.T @ Z
    try:
        Mi = np.linalg.inv(M)
        _sm, logdet_m = np.linalg.slogdet(M)
    except np.linalg.LinAlgError:
        Mi = np.linalg.pinv(M)
        logdet_m = float(np.log(max(abs(np.linalg.det(M)), 1e-300)))

    ZtX, Zty = Z.T @ X, Z.T @ y
    XtHiX = X.T @ X - ZtX.T @ Mi @ ZtX
    XtHiy = X.T @ y - ZtX.T @ Mi @ Zty
    try:
        beta = np.linalg.solve(XtHiX, XtHiy)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtHiX, XtHiy, rcond=None)[0]

    r = y - X @ beta
    Ztr = Z.T @ r
    rHr = float(r @ r - Ztr @ Mi @ Ztr)

    logdet_H = float(logdet_m + sum(q * np.log(l) for q, l in zip(qs, lam)))
    _sx, logdet_x = np.linalg.slogdet(XtHiX)
    if _sx <= 0:
        logdet_x = float(np.log(max(abs(np.linalg.det(XtHiX)), 1e-300)))
    return beta, max(rHr, 1e-300), logdet_H, float(logdet_x), XtHiX


def _m2ll_profiled(loglam, X, y, Zs, qs) -> float:
    n, p = X.shape
    _, rHr, ldH, ldX, _ = _core(loglam, X, y, Zs, qs)
    dof = max(n - p, 1)
    return float(dof * np.log(rHr / dof) + ldH + ldX)


def _m2ll_full(theta, X, y, Zs, qs) -> float:
    """-2 REML log-likelihood with s2e free, needed for the Hessian."""
    n, p = X.shape
    s2e = float(np.exp(theta[0]))
    _, rHr, ldH, ldX, _ = _core(np.asarray(theta[1:]), X, y, Zs, qs)
    return float((n - p) * np.log(s2e) + ldH + ldX + rHr / s2e)


def _numerical_hessian(f, x0, step: float = 1e-4):
    x0 = np.asarray(x0, float)
    k = len(x0)
    H = np.zeros((k, k))
    h = step * np.maximum(np.abs(x0), 1.0)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            H[i, j] = H[j, i] = (
                f(x0 + ei + ej) - f(x0 + ei - ej)
                - f(x0 - ei + ej) + f(x0 - ei - ej)
            ) / (4 * h[i] * h[j])
    return H


def _satterthwaite(theta_hat, X, y, Zs, qs) -> np.ndarray:
    """
    Satterthwaite denominator degrees of freedom per fixed effect.

        df_j = 2 . g_j^2 / ( grad(g_j)' . Cov(theta) . grad(g_j) )

    g_j is the sampling variance of beta_j. Cov(theta) comes from the
    observed REML information (2 x inverse Hessian of -2logL), and both
    it and the gradient are evaluated numerically. Parameters are on the
    log scale so the finite-difference steps are scale-free.
    """
    theta_hat = np.asarray(theta_hat, float)
    k = len(theta_hat)
    p = X.shape[1]

    def var_all(theta):
        s2e = float(np.exp(theta[0]))
        _, _, _, _, XtHiX = _core(np.asarray(theta[1:]), X, y, Zs, qs)
        try:
            return s2e * np.diag(np.linalg.inv(XtHiX))
        except np.linalg.LinAlgError:
            return s2e * np.diag(np.linalg.pinv(XtHiX))

    try:
        H = _numerical_hessian(lambda t: _m2ll_full(t, X, y, Zs, qs),
                               theta_hat)
        cov_theta = 2.0 * np.linalg.pinv(H)
    except Exception:
        return np.full(p, np.nan)

    g0 = var_all(theta_hat)
    grad = np.zeros((p, k))
    for i in range(k):
        h = 1e-4 * max(abs(theta_hat[i]), 1.0)
        e = np.zeros(k); e[i] = h
        grad[:, i] = (var_all(theta_hat + e) - var_all(theta_hat - e)) / (2 * h)

    df = np.full(p, np.nan)
    for j in range(p):
        denom = float(grad[j] @ cov_theta @ grad[j])
        if np.isfinite(denom) and denom > 1e-300:
            df[j] = 2.0 * g0[j] ** 2 / denom
    return df


# ----------------------------------------------------------------------
def fit_lmm(X, y, terms: list, feature_names: list | None = None,
            add_intercept: bool = True,
            satterthwaite: bool = True) -> LMMFit:
    """
    Fit a linear mixed model with any number of independent random terms.

    `terms` is a list of RandomTerm. Crossed factors are two terms with
    different grouping arrays; a random slope is a term carrying a
    `slope` column.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, float).ravel()
    n = len(y)

    names = list(feature_names) if feature_names else [
        f"x{i}" for i in range(X.shape[1])]
    if add_intercept:
        X = np.column_stack([np.ones(n), X])
        names = ["(Intercept)"] + names
    p = X.shape[1]

    Zs, qs, sizes_list, labels = [], [], [], []
    warns: list[str] = []
    for t in terms:
        Z, q, _lv, sizes = t.build(n)
        Zs.append(Z)
        qs.append(q)
        sizes_list.append(sizes)
        labels.append(t.label)
        if q < MIN_GROUPS_FOR_LMM:
            warns.append(
                f"Term '{t.label}' has only {q} level(s). Its variance is "
                f"barely identifiable below {MIN_GROUPS_FOR_LMM} levels; "
                f"treat it as indicative.")
        if t.slope is None and (sizes == 1).all():
            warns.append(
                f"Every level of '{t.label}' has one observation, so that "
                f"random effect is not identifiable and collapses to zero.")

    K = len(Zs)
    if K == 0:
        beta, rHr, _, _, XtX = _core(np.array([]), X, y, [], [])
        s2e = rHr / max(n - p, 1)
        se = np.sqrt(np.clip(np.diag(s2e * np.linalg.pinv(XtX)), 0, None))
        return LMMFit(beta, se, names, 0.0, float(s2e), 0.0, 0.0, n, 0,
                      np.array([n]), max(n - p, 1), 0.0, True,
                      warns + ["No random terms supplied; this is OLS."])

    best_x = np.full(K, LOG_LAMBDA_BOUNDS[0])
    best_f = _m2ll_profiled(best_x, X, y, Zs, qs)
    converged = True
    for start in (np.zeros(K), np.full(K, -2.0), np.full(K, 2.0)):
        try:
            res = minimize(_m2ll_profiled, start, args=(X, y, Zs, qs),
                           method="L-BFGS-B",
                           bounds=[LOG_LAMBDA_BOUNDS] * K,
                           options={"maxiter": 600, "ftol": 1e-12})
            if res.fun < best_f:
                best_f, best_x = float(res.fun), np.asarray(res.x, float)
        except Exception:
            converged = False

    beta, rHr, _ldH, _ldX, XtHiX = _core(best_x, X, y, Zs, qs)
    s2e = float(rHr / max(n - p, 1))
    lam = np.exp(np.clip(best_x, *LOG_LAMBDA_BOUNDS))
    comps = {lab: float(l * s2e) for lab, l in zip(labels, lam)}
    s2u = float(sum(comps.values()))
    icc = float(s2u / (s2u + s2e)) if (s2u + s2e) > 0 else 0.0

    try:
        se = np.sqrt(np.clip(np.diag(s2e * np.linalg.inv(XtHiX)), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
        warns.append("Covariance matrix is singular; standard errors are "
                     "unavailable.")

    df_satt = None
    if satterthwaite:
        try:
            df_satt = _satterthwaite(
                np.concatenate([[np.log(max(s2e, 1e-300))], best_x]),
                X, y, Zs, qs)
        except Exception:
            warns.append("Satterthwaite degrees of freedom unavailable; "
                         "falling back to the containment approximation.")

    intercept_terms = [i for i, t in enumerate(terms) if t.slope is None]
    pick = (max(intercept_terms, key=lambda i: comps[labels[i]])
            if intercept_terms else 0)
    sizes = np.asarray(sizes_list[pick])

    if icc > HIGH_ICC:
        warns.append(
            f"ICC = {icc:.3f}. Observations within a group are correlated, "
            f"so pooled regression will understate standard errors and "
            f"overstate the effective sample size.")
    if K > 1:
        warns.append(
            f"Design effect and effective n are computed from the dominant "
            f"grouping factor ('{labels[pick]}') against the TOTAL ICC. With "
            f"{K} random terms that is an approximation - for crossed "
            f"factors there is no single cluster size, so read the per-term "
            f"variances and the LRT table rather than the single ICC.")

    dof_cont = max(n - p - (sum(qs) - K), 1)
    return LMMFit(beta=beta, se=se, feature_names=names,
                  sigma2_u=s2u, sigma2_e=s2e, icc=icc,
                  lambda_=float(lam[pick]), n_obs=n, n_groups=int(qs[pick]),
                  group_sizes=sizes, dof=int(dof_cont),
                  reml_loglik=float(-0.5 * best_f), converged=converged,
                  warnings=warns, variance_components=comps,
                  dof_satterthwaite=df_satt, term_labels=labels)


# ----------------------------------------------------------------------
def fit_random_intercept(X, y, groups, feature_names=None,
                         add_intercept: bool = True,
                         satterthwaite: bool = True) -> LMMFit:
    """Single random intercept - the common case."""
    return fit_lmm(X, y, [RandomTerm(np.asarray(groups), "group")],
                   feature_names, add_intercept, satterthwaite)


def fit_crossed(X, y, group_a, group_b, feature_names=None,
                name_a: str = "site", name_b: str = "campaign",
                satterthwaite: bool = True) -> LMMFit:
    """
    Two crossed grouping factors, neither nested in the other.

    Site and campaign is the usual pharma case: every site runs every
    campaign, so neither is a sub-level of the other, and nesting them
    would charge site-to-site variation to campaigns.
    """
    return fit_lmm(X, y,
                   [RandomTerm(np.asarray(group_a), name_a),
                    RandomTerm(np.asarray(group_b), name_b)],
                   feature_names, True, satterthwaite)


def fit_random_slope(X, y, groups, slope_col: int, feature_names=None,
                     name: str = "group",
                     satterthwaite: bool = True) -> LMMFit:
    """
    Random intercept plus an independent random slope on one factor.

    Use when a factor's effect plausibly differs between sites - a
    polymer level behaving differently on different compression lines,
    say. Forcing one common slope treats that heterogeneity as noise and
    understates the uncertainty on the average effect.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    names = list(feature_names) if feature_names else [
        f"x{i}" for i in range(X.shape[1])]
    g = np.asarray(groups)
    return fit_lmm(X, y,
                   [RandomTerm(g, name),
                    RandomTerm(g, name, slope=X[:, slope_col],
                               slope_name=names[slope_col])],
                   names, True, satterthwaite)


# ----------------------------------------------------------------------
def lrt_random_effect(X, y, groups, add_intercept: bool = True) -> dict:
    """
    Likelihood-ratio test for a random effect.

    The null sits on the boundary of the parameter space (variance = 0),
    so the reference distribution is the 50:50 mixture of a point mass at
    zero and chi-square with one degree of freedom - not plain chi2_1.
    Using chi2_1 roughly doubles the p-value and hides real batch
    structure.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, float).ravel()
    Xd = np.column_stack([np.ones(len(X)), X]) if add_intercept else X

    m2ll_null = _m2ll_profiled(np.array([]), Xd, y, [], [])
    fit = fit_random_intercept(X, y, groups, add_intercept=add_intercept,
                               satterthwaite=False)
    stat = float(max(m2ll_null - (-2.0 * fit.reml_loglik), 0.0))
    p = float(0.5 * stats.chi2.sf(stat, 1)) if stat > 0 else 1.0
    return {
        "lr_statistic": stat,
        "p_value": p,
        "significant": bool(p < 0.05),
        "icc": fit.icc,
        "reference": "50:50 mixture of chi2_0 and chi2_1 (boundary test)",
        "verdict": (
            f"Batch/site structure is real (p={p:.4f}, ICC={fit.icc:.3f}). "
            f"Pooled regression is not appropriate."
            if p < 0.05 else
            f"No detectable group structure (p={p:.4f}). Pooling is "
            f"defensible for this response."),
    }


def lrt_terms(X, y, terms: list, add_intercept: bool = True) -> pd.DataFrame:
    """
    Drop each random term in turn and test it by likelihood ratio.

    With crossed factors this is the only way to see which one carries
    the structure: a single pooled ICC cannot separate site from campaign.
    """
    full = fit_lmm(X, y, terms, add_intercept=add_intercept,
                   satterthwaite=False)
    rows = []
    for i, t in enumerate(terms):
        reduced = [u for j, u in enumerate(terms) if j != i]
        red = fit_lmm(X, y, reduced, add_intercept=add_intercept,
                      satterthwaite=False)
        stat = float(max(-2 * red.reml_loglik + 2 * full.reml_loglik, 0.0))
        p = float(0.5 * stats.chi2.sf(stat, 1)) if stat > 0 else 1.0
        rows.append({
            "term": t.label,
            "variance": full.variance_components.get(t.label, np.nan),
            "lr_statistic": stat,
            "p_value": p,
            "significant": bool(p < 0.05),
        })
    return pd.DataFrame(rows)


def compare_pooled_vs_mixed(X, y, groups, feature_names=None) -> dict:
    """
    Side-by-side of pooled OLS and the mixed model.

    The number that matters is the standard-error ratio, and its
    direction depends on where a factor varies. A factor varying BETWEEN
    groups loses precision, because its effective sample size is the
    number of groups. A factor varying WITHIN groups gains precision,
    because the group offset moves out of the residual.
    """
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, float).ravel()
    names = list(feature_names) if feature_names else [
        f"x{i}" for i in range(X.shape[1])]

    Xd = np.column_stack([np.ones(len(X)), X])
    beta_ols, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta_ols
    n, p = Xd.shape
    s2 = float(resid @ resid) / max(n - p, 1)
    try:
        se_ols = np.sqrt(np.clip(np.diag(s2 * np.linalg.inv(Xd.T @ Xd)),
                                 0, None))
    except np.linalg.LinAlgError:
        se_ols = np.full(p, np.nan)

    fit = fit_random_intercept(X, y, groups, names)
    ratio = fit.se / np.where(se_ols > 0, se_ols, np.nan)
    dof = (fit.dof_satterthwaite if fit.dof_satterthwaite is not None
           else np.full(p, float(fit.dof)))

    tbl = pd.DataFrame({
        "term": ["(Intercept)"] + names,
        "OLS_estimate": beta_ols,
        "OLS_se": se_ols,
        "OLS_df": float(max(n - p, 1)),
        "LMM_estimate": fit.beta,
        "LMM_se": fit.se,
        "LMM_df_satterthwaite": dof,
        "se_ratio_LMM_over_OLS": ratio,
    })
    slopes = ratio[1:] if len(ratio) > 1 else np.array([np.nan])
    worst, lowest = float(np.nanmax(slopes)), float(np.nanmin(slopes))
    icept = float(ratio[0]) if len(ratio) else float("nan")

    if np.isfinite(worst) and worst > 1.15:
        verdict = (
            f"Pooled OLS understates the standard error of at least one "
            f"factor by a factor of {worst:.2f}. Significance claims from "
            f"the pooled fit are not reliable.")
    elif np.isfinite(lowest) and lowest < 0.85:
        verdict = (
            f"The mixed model is MORE precise on the factors (SE ratio down "
            f"to {lowest:.2f}) because these vary within group, so removing "
            f"the group offset shrinks the residual instead of leaving it "
            f"in the error term. The cost lands on the intercept "
            f"(ratio {icept:.2f}) and on anything measured between groups.")
    else:
        verdict = ("Pooled and mixed standard errors agree closely; the "
                   "grouping is not distorting inference for these factors.")

    return {"table": tbl, "fit": fit, "max_se_inflation": worst,
            "min_se_ratio": lowest, "intercept_se_ratio": icept,
            "lrt": lrt_random_effect(X, y, groups), "verdict": verdict}


def effective_sample_size(y, groups, X=None) -> dict:
    """
    Effective n for a response. This is what the design gate should read
    rather than the raw row count.

    Pass `X` whenever covariates are part of the eventual model. The ICC
    is conditional on whatever is in the mean structure: with no
    covariates, factor-driven variation is charged to the residual and
    the ICC comes out far too low.
    """
    y = np.asarray(y, float).ravel()
    if X is None:
        Xd, names = np.zeros((len(y), 0)), []
        note_cov = "unconditional (no covariates supplied)"
    else:
        Xd = np.asarray(X, float)
        if Xd.ndim == 1:
            Xd = Xd.reshape(-1, 1)
        names = [f"x{i}" for i in range(Xd.shape[1])]
        note_cov = f"conditional on {Xd.shape[1]} covariate(s)"

    fit = fit_random_intercept(Xd, y, groups, feature_names=names,
                               add_intercept=True, satterthwaite=False)
    return {
        "n_rows": int(len(y)),
        "n_groups": int(fit.n_groups),
        "icc": fit.icc,
        "design_effect": fit.design_effect,
        "effective_n": fit.effective_n,
        "shrinkage_pct": float(100 * (1 - fit.effective_n / max(len(y), 1))),
        "conditioning": note_cov,
        "note": (f"{len(y)} rows across {fit.n_groups} groups with ICC="
                 f"{fit.icc:.3f} ({note_cov}) behave like "
                 f"{fit.effective_n:.1f} independent observations."),
    }
