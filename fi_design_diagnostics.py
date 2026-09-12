"""
Design diagnostics: run BEFORE any model is fitted.

The single most important module in this package. It decides whether a
regression problem is admissible at all. In small-n formulation work the
usual failure is not a bad model, it is a model that should never have
been fitted: rank-deficient design, mixture-constrained excipients, more
features than informative directions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Thresholds (single source of truth)
# ----------------------------------------------------------------------
MIN_N_FOR_STATISTICAL_MODELLING = 8
MIN_N_PER_FEATURE = 3.0
MIXTURE_CV_TOLERANCE = 0.02      # CV of row sums below this => constrained
RANK_TOL = 1e-8
HIGH_VIF = 10.0


@dataclass
class DesignVerdict:
    """Outcome of the admissibility gate."""

    admissible: bool
    mode: str                      # 'statistical' | 'mechanistic' | 'blocked'
    n_samples: int
    n_features: int
    rank_centered: int
    effective_dof: int
    mixture_detected: bool
    mixture_columns: list = field(default_factory=list)
    mixture_sum_cv: float = np.nan
    constant_columns: list = field(default_factory=list)
    duplicate_rows: int = 0
    vif: dict = field(default_factory=dict)
    condition_number: float = np.nan
    n_groups: int = 0
    icc: float = np.nan
    design_effect: float = np.nan
    effective_n: float = np.nan
    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Mode              : {self.mode.upper()}",
            f"Samples           : {self.n_samples}",
            f"Features          : {self.n_features}",
            f"Rank (centred X)  : {self.rank_centered}",
            f"Effective d.o.f.  : {self.effective_dof}",
            f"Mixture detected  : {self.mixture_detected}",
        ]
        if self.n_groups:
            lines += [
                f"Groups (batch/site): {self.n_groups}",
                f"ICC               : {self.icc:.4f}",
                f"Design effect     : {self.design_effect:.3f}",
                f"Effective n       : {self.effective_n:.1f} "
                f"(vs {self.n_samples} rows)",
            ]
        if self.blockers:
            lines.append("BLOCKERS:")
            lines += [f"  - {b}" for b in self.blockers]
        if self.warnings:
            lines.append("WARNINGS:")
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


# ----------------------------------------------------------------------
def detect_mixture(
    X: pd.DataFrame, tol: float = MIXTURE_CV_TOLERANCE
) -> tuple[bool, list, float]:
    """
    Detect a constant-sum (mixture) constraint among numeric columns.

    Tries the full set first, then every leave-one-out subset, then
    subsets of decreasing size, returning the largest group whose row
    sums have coefficient of variation below `tol`.

    Returns (detected, columns, cv_of_row_sums).
    """
    cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if len(cols) < 2 or len(X) < 2:
        return False, [], float("nan")

    from itertools import combinations

    # Exhaustive subset search is 2^p; cap it. For wide designs test only
    # the full set and the leave-one-out / leave-two-out subsets.
    MAX_EXHAUSTIVE = 12
    if len(cols) <= MAX_EXHAUSTIVE:
        sizes = range(len(cols), 1, -1)
        candidates = (
            combo for size in sizes for combo in combinations(cols, size)
        )
    else:
        candidates = (
            combo
            for size in (len(cols), len(cols) - 1, len(cols) - 2)
            if size > 1
            for combo in combinations(cols, size)
        )

    best: tuple[list, float] | None = None
    for combo in candidates:
        s = X.loc[:, list(combo)].sum(axis=1).to_numpy(float)
        m = np.abs(s.mean())
        if m < 1e-12:
            continue
        cv = float(s.std(ddof=0) / m)
        if cv >= tol:
            continue
        # Prefer the tightest constraint; break ties toward the larger group
        # so a genuine full-formula constraint is not split.
        if best is None or cv < best[1] * 0.5 or (
            abs(cv - best[1]) <= best[1] * 0.5 and len(combo) > len(best[0])
        ):
            best = (list(combo), cv)

    if best is None:
        return False, [], float("nan")
    return True, best[0], best[1]


def variance_inflation(X: pd.DataFrame) -> dict:
    """VIF per column. Returns inf where a column is a perfect combination."""
    out: dict = {}
    cols = list(X.columns)
    if len(cols) < 2:
        return {c: 1.0 for c in cols}

    Xv = np.array(X.to_numpy(float), copy=True)
    for j, c in enumerate(cols):
        y = Xv[:, j]
        others = np.delete(Xv, j, axis=1)
        if others.shape[0] <= others.shape[1]:
            out[c] = float("inf")
            continue
        A = np.column_stack([np.ones(len(others)), others])
        try:
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = y - A @ beta
            ss_res = float(resid @ resid)
            ss_tot = float(((y - y.mean()) ** 2).sum())
            if ss_tot < 1e-15:
                out[c] = float("inf")
                continue
            r2 = 1.0 - ss_res / ss_tot
            out[c] = float("inf") if r2 > 1 - 1e-10 else 1.0 / (1.0 - r2)
        except np.linalg.LinAlgError:
            out[c] = float("inf")
    return out


def diagnose(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray | None = None,
    min_n: int = MIN_N_FOR_STATISTICAL_MODELLING,
    groups=None,
) -> DesignVerdict:
    """
    Full admissibility gate. Call this before touching model_arena.

    Pass `groups` (batch, site or campaign labels) whenever the rows come
    from more than one manufacturing campaign. Correlated observations
    within a group mean the row count overstates the information
    available, and the gate then tests the EFFECTIVE sample size instead.
    A 48-row dataset from six campaigns with a high ICC can be a genuine
    n of twelve, and admitting it on the row count is how an
    inadmissible design slips through.
    """
    X = X.select_dtypes(include=[np.number]).copy()
    n, p = X.shape

    blockers: list[str] = []
    warnings: list[str] = []

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if constant_cols:
        warnings.append(
            f"Constant column(s) carry no information: {constant_cols}"
        )

    dup = int(len(X) - len(X.drop_duplicates()))
    if dup:
        warnings.append(f"{dup} duplicate row(s) in the design matrix")

    Xv = np.array(X.to_numpy(float), copy=True)
    Xc = Xv - Xv.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(Xc, tol=RANK_TOL)) if n > 1 else 0

    mixture, mix_cols, mix_cv = detect_mixture(X)
    # a constant-sum group costs one further direction
    eff_dof = max(rank - (1 if mixture else 0), 0)

    try:
        cond = float(np.linalg.cond(Xc)) if rank > 0 else float("inf")
    except np.linalg.LinAlgError:
        cond = float("inf")

    vif = variance_inflation(X) if n > p else {c: float("inf") for c in X.columns}

    # ---------------- grouping / effective n ----------------
    n_groups, icc, deff, eff_n = 0, np.nan, np.nan, float(n)
    if groups is not None:
        try:
            from fi_mixed_models import effective_sample_size

            g = np.asarray(groups)
            if len(g) == n and len(np.unique(g)) > 1:
                es = effective_sample_size(
                    np.asarray(y, float) if y is not None else np.zeros(n),
                    g, X=Xv)
                n_groups = int(es["n_groups"])
                icc = float(es["icc"])
                deff = float(es["design_effect"])
                eff_n = float(es["effective_n"])
                if icc > 0.10:
                    warnings.append(
                        f"Batch/site ICC = {icc:.3f}. The {n} rows carry "
                        f"about {eff_n:.1f} independent observations "
                        f"(design effect {deff:.2f}). Pooled regression "
                        f"will report standard errors that are too small; "
                        f"use the Mixed Effects page."
                    )
        except Exception as exc:
            warnings.append(f"Grouping ignored ({type(exc).__name__}).")

    n_for_gate = eff_n if n_groups else float(n)

    # ---------------- blocking rules ----------------
    if n_for_gate < min_n:
        blockers.append(
            f"{'Effective n' if n_groups else 'n'}={n_for_gate:.1f} is below "
            f"the minimum of {min_n} for statistical modelling"
            + (f" ({n} rows discounted for batch structure)"
               if n_groups else "")
        )
    if rank < p:
        blockers.append(
            f"Design matrix is rank deficient: rank(centred X)={rank} < "
            f"p={p}. Infinitely many exact fits exist; any coefficient "
            f"vector returned would be chosen by the penalty, not the data"
        )
    if n_for_gate < MIN_N_PER_FEATURE * p:
        blockers.append(
            f"{'Effective n' if n_groups else 'n'}={n_for_gate:.1f} gives "
            f"{n_for_gate / max(p, 1):.1f} samples per feature "
            f"(minimum {MIN_N_PER_FEATURE:.0f})"
        )
    if eff_dof < 2:
        blockers.append(
            f"Only {eff_dof} effective degree(s) of freedom after centring"
            + (" and the mixture constraint" if mixture else "")
        )

    if mixture:
        warnings.append(
            f"Mixture constraint on {mix_cols} (row-sum CV={mix_cv:.2e}). "
            f"Ordinary regression with an intercept is invalid here - use "
            f"a Scheffe canonical model."
        )
    hi_vif = {c: v for c, v in vif.items() if v > HIGH_VIF}
    if hi_vif:
        warnings.append(
            "High collinearity (VIF>10): "
            + ", ".join(
                f"{c}={'inf' if np.isinf(v) else f'{v:.1f}'}"
                for c, v in hi_vif.items()
            )
        )
    if y is not None:
        yv = np.asarray(y, float)
        if np.nanstd(yv) < 1e-12:
            blockers.append("Response has zero variance")

    mode = "statistical" if not blockers else (
        "mechanistic" if n >= 3 else "blocked"
    )

    return DesignVerdict(
        admissible=not blockers,
        mode=mode,
        n_samples=n,
        n_features=p,
        rank_centered=rank,
        effective_dof=eff_dof,
        mixture_detected=mixture,
        mixture_columns=mix_cols,
        mixture_sum_cv=mix_cv,
        constant_columns=constant_cols,
        duplicate_rows=dup,
        vif=vif,
        condition_number=cond,
        n_groups=n_groups,
        icc=icc,
        design_effect=deff,
        effective_n=eff_n,
        blockers=blockers,
        warnings=warnings,
    )


def spurious_correlation_risk(n: int, n_features: int, n_sim: int = 4000,
                              seed: int = 0) -> dict:
    """
    Probability that at least one of `n_features` pure-noise variables
    reaches |r| >= 0.9 / 0.95 / 0.99 against a noise response at this n.

    Used to put a number on why a correlation bar chart cannot be read
    at small n.
    """
    rng = np.random.default_rng(seed)
    hits = {0.90: 0, 0.95: 0, 0.99: 0}
    for _ in range(n_sim):
        y = rng.normal(size=n)
        Xr = rng.normal(size=(n, max(n_features, 1)))
        yc = y - y.mean()
        Xc = Xr - Xr.mean(axis=0)
        num = Xc.T @ yc
        den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum()) + 1e-300
        rmax = float(np.max(np.abs(num / den)))
        for t in hits:
            if rmax >= t:
                hits[t] += 1
    return {f"P(max|r|>={t:.2f})": hits[t] / n_sim for t in hits}
