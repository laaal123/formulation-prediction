"""
Model arena - the leaderboard engine.

    detect task -> gate -> candidate list -> nested CV ->
    rank by Q2 -> 1-SE rule -> y-randomization -> applicability domain

Cost control: the permutation test is the expensive step, so it runs
only on contenders (top-K by Q2) with a per-model budget derived from
the measured cross-validation time. A cheap model gets the full 200
permutations; an expensive one gets fewer, and the number actually used
is reported so the p-value resolution is never hidden.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fi_applicability import ApplicabilityDomain
from fi_design_diagnostics import DesignVerdict, diagnose
from fi_models import build_candidates, regime_label
from fi_validation import (PERMUTATION_ALPHA, CVResult, apply_one_se_rule,
                         cross_validate_model, evaluate, make_cv,
                         make_permutation_cv, y_randomization)

DEFAULT_PERM_BUDGET_S = 25.0
MIN_PERM = 40
MAX_PERM = 200
DEFAULT_TOP_K = 4


def cv_name_preview(n, seed, groups):
    return make_cv(n, seed, groups=groups)[1]


@dataclass
class ArenaResult:
    verdict: DesignVerdict
    regime: str
    cv_scheme: str
    leaderboard: pd.DataFrame
    results: list = field(default_factory=list)
    selected: CVResult | None = None
    selected_estimator: object = None
    applicability: ApplicabilityDomain | None = None
    feature_names: list = field(default_factory=list)
    no_signal: bool = False
    messages: list = field(default_factory=list)

    def explain_selection(self) -> str:
        if self.no_signal or self.selected is None:
            return (
                "NO MODEL SELECTED. No candidate beat its own permuted "
                "null at alpha=%.2f. Any apparent correlation in this "
                "dataset is consistent with chance." % PERMUTATION_ALPHA
            )
        s = self.selected
        best = max((r for r in self.results if r.signal),
                   key=lambda r: r.q2, default=s)
        txt = [
            f"Selected: {s.name}",
            f"  Q2 = {s.q2:.3f}   RMSECV = {s.rmsecv:.4g}   "
            f"complexity = {s.complexity}",
            f"  permutation p = {s.perm_p:.4f} "
            f"(null mean Q2 = {s.perm_null_mean:.3f})",
        ]
        if best.name != s.name:
            txt.append(
                f"  Note: {best.name} scored higher (Q2={best.q2:.3f}) but "
                f"lies within one standard error ({best.q2_se:.3f}) of the "
                f"selection, so the simpler model wins under the 1-SE rule."
            )
        if s.r2_train - s.q2 > 0.25:
            txt.append(
                f"  Overfit gap: training R2={s.r2_train:.3f} vs "
                f"Q2={s.q2:.3f}. Trust Q2."
            )
        return "\n".join(txt)


def run_arena(
    X: pd.DataFrame,
    y: pd.Series,
    mixture_idx: list | None = None,
    process_idx: tuple = (),
    seed: int = 0,
    top_k: int = DEFAULT_TOP_K,
    perm_budget_s: float = DEFAULT_PERM_BUDGET_S,
    force: bool = False,
    groups=None,
) -> ArenaResult:
    """
    Full leaderboard run. If the design gate blocks and force=False the
    arena refuses to fit and returns the verdict only - that refusal is
    the feature, not a limitation.
    """
    Xn = X.select_dtypes(include=[np.number])
    names = list(Xn.columns)
    Xv = np.array(Xn.to_numpy(float), copy=True)
    yv = np.asarray(y, float).ravel()
    n, p = Xv.shape

    verdict = diagnose(Xn, yv, groups=groups)
    msgs: list[str] = []
    if verdict.n_groups and verdict.icc > 0.10:
        msgs.append(
            f"Batch/site ICC = {verdict.icc:.3f}. Folds are drawn at the "
            f"GROUP level ({cv_name_preview(n, seed, groups)}) so no batch "
            f"appears in both training and test. Q2 below is therefore the "
            f"honest out-of-batch figure, not the inflated within-batch one.")

    if mixture_idx is None and verdict.mixture_detected:
        mixture_idx = [names.index(c) for c in verdict.mixture_columns
                       if c in names]
        msgs.append(
            f"Mixture constraint auto-detected on {verdict.mixture_columns}; "
            f"Scheffe canonical models added and intercept suppressed."
        )
    if mixture_idx and not process_idx:
        process_idx = tuple(j for j in range(p) if j not in mixture_idx)

    cv, cv_name = make_cv(n, seed, groups=groups)
    regime = regime_label(n, bool(mixture_idx))

    if not verdict.admissible and not force:
        msgs.append(
            "Design gate BLOCKED statistical modelling. Routed to "
            "mechanistic mode. Use the Percolation / IVIVC pages instead."
        )
        return ArenaResult(verdict, regime, cv_name,
                           pd.DataFrame(), messages=msgs,
                           feature_names=names, no_signal=True)
    if not verdict.admissible and force:
        msgs.append(
            "WARNING: gate overridden. Results below are not defensible "
            "and must not be used for a regulatory decision."
        )

    # ---------- pass 1: cheap screen, no permutation ----------
    candidates = build_candidates(n, p, mixture_idx, process_idx, seed)
    results: list[CVResult] = []
    timings: dict[str, float] = {}
    for nm, est, cx in candidates:
        t0 = time.time()
        r = evaluate(nm, est, Xv, yv, cx, seed=seed, run_permutation=False,
                     groups=groups)
        timings[nm] = max(time.time() - t0, 1e-3)
        r._estimator = est          # noqa: SLF001  (kept for refit)
        results.append(r)

    valid = [r for r in results if np.isfinite(r.q2) and not r.failed]
    if not valid:
        msgs.append("Every candidate failed to fit.")
        return ArenaResult(verdict, regime, cv_name, pd.DataFrame(),
                           results, messages=msgs, feature_names=names,
                           no_signal=True)

    # ---------- pass 2: permutation on contenders only ----------
    contenders = sorted(valid, key=lambda r: -r.q2)[:top_k]
    simplest = min(valid, key=lambda r: r.complexity)
    if simplest not in contenders:
        contenders.append(simplest)

    perm_cv = make_permutation_cv(n, seed, groups=groups)
    for r in contenders:
        # Measure one permuted fit on the CHEAP cv to price the test
        t0 = time.time()
        try:
            cross_validate_model(r._estimator, Xv,
                                 np.random.default_rng(seed).permutation(yv),
                                 cv=perm_cv, groups=groups)
        except Exception:
            pass
        unit = max(time.time() - t0, 1e-3)
        affordable = int(perm_budget_s / unit)

        if affordable < MIN_PERM:
            r.signal = False
            r._n_perm = 0                      # noqa: SLF001
            msgs.append(
                f"{r.name}: permutation test unaffordable "
                f"({unit:.2f}s per shuffle; {MIN_PERM} needed = "
                f"{unit * MIN_PERM:.0f}s > budget {perm_budget_s:.0f}s). "
                f"Excluded from selection - an unvalidated model is not a "
                f"candidate. Raise perm_budget_s to include it."
            )
            continue

        n_perm = int(np.clip(affordable, MIN_PERM, MAX_PERM))
        pval, nmean, nq = y_randomization(r._estimator, Xv, yv, r.q2,
                                          n_perm=n_perm, seed=seed,
                                          cv=perm_cv, groups=groups)
        r.perm_p, r.perm_null_mean, r.perm_null_q95 = pval, nmean, nq
        r.signal = bool(pval < PERMUTATION_ALPHA and r.q2 > 0)
        r._n_perm = n_perm                     # noqa: SLF001
        if n_perm < MAX_PERM:
            msgs.append(
                f"{r.name}: {n_perm} shuffles (cost-limited); "
                f"minimum resolvable p = {1 / (n_perm + 1):.3f}"
            )

    selected = apply_one_se_rule(contenders)
    no_signal = selected is None
    if no_signal:
        msgs.append(
            "NO SIGNAL. No contender separated from its permuted null. "
            "Reporting a model here would be reporting noise."
        )

    est_fitted, ad = None, None
    if selected is not None:
        from sklearn.base import clone
        est_fitted = clone(selected._estimator).fit(Xv, yv)
        ad = ApplicabilityDomain().fit(Xv, names)

    lb = pd.DataFrame([
        {
            "model": r.name,
            "Q2": round(r.q2, 4),
            "RMSECV": round(r.rmsecv, 5),
            "MAE_CV": round(r.mae_cv, 5),
            "R2_train": round(r.r2_train, 4) if np.isfinite(r.r2_train) else np.nan,
            "overfit_gap": (round(r.r2_train - r.q2, 4)
                            if np.isfinite(r.r2_train) else np.nan),
            "complexity": r.complexity,
            "perm_p": round(r.perm_p, 4) if np.isfinite(r.perm_p) else None,
            "signal": r.signal,
            "tested": r in contenders,
        }
        for r in sorted(valid, key=lambda r: -r.q2)
    ])

    return ArenaResult(verdict, regime, cv_name, lb, results, selected,
                       est_fitted, ad, names, no_signal, msgs)
