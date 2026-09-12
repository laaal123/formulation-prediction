"""
Synthetic formulation data with KNOWN ground truth.

Every generator returns (dataframe, truth_dict). The truth dict is what
the validation suite checks the app against: if the app cannot recover a
driver it planted itself, it cannot be trusted on real data.

Scenarios deliberately include the failure modes:
  * pure noise            -> the app must report NO SIGNAL
  * n=3 rank deficient    -> the app must REFUSE to model
  * spurious correlation  -> the app must not promote the decoy
  * mixture constraint    -> Scheffe must beat ordinary regression
  * percolation threshold -> the app must find the interval, not a slope
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fi_dissolution as D


def matrix_tablet_doe(n: int = 48, seed: int = 0, noise: float = 0.03
                      ) -> tuple[pd.DataFrame, dict]:
    """
    A realistic sustained-release matrix DoE.

    Truth: polymer dominates, hardness is weak, a polymer x filler
    interaction exists, and lubricant is pure decoy.
    """
    rng = np.random.default_rng(seed)
    polymer = rng.uniform(15, 40, n)
    hardness = rng.uniform(6, 14, n)
    lubricant = rng.uniform(0.5, 2.0, n)          # decoy
    filler = 100 - polymer - lubricant - rng.uniform(18, 24, n)

    z = (-0.070 * (polymer - 27)
         - 0.015 * (hardness - 10)
         + 0.0025 * (polymer - 27) * (filler - 50)
         + rng.normal(0, noise, n))
    k_release = np.exp(z) * 0.06

    t = np.array([0, 1, 2, 4, 6, 8, 12, 16, 20, 24.])
    alpha = 1.0 / np.maximum(k_release, 1e-6) * 0.9
    beta = 0.80 + 0.004 * (hardness - 10)
    prof = np.array([D.weibull(t, a, b) for a, b in zip(alpha, beta)])
    prof += rng.normal(0, 1.5, prof.shape)
    prof = np.clip(np.maximum.accumulate(prof, axis=1), 0, 100)

    df = pd.DataFrame({
        "batch": [f"B{i+1:03d}" for i in range(n)],
        "Polymer_pct": polymer.round(2),
        "Filler_pct": filler.round(2),
        "Lubricant_pct": lubricant.round(3),
        "Hardness_kP": hardness.round(2),
        "Q1h": prof[:, 1].round(2),
        "Q6h": prof[:, 4].round(2),
        "Q12h": prof[:, 6].round(2),
        "Q24h": prof[:, 9].round(2),
        "Weibull_alpha": alpha.round(3),
        "Weibull_beta": beta.round(4),
    })
    truth = {
        "drivers": ["Polymer_pct", "Hardness_kP"],
        "dominant": "Polymer_pct",
        "decoys": ["Lubricant_pct"],
        "interaction": ("Polymer_pct", "Filler_pct"),
        "timepoints": t.tolist(),
        "profiles": prof,
    }
    return df, truth


def pure_noise(n: int = 30, p: int = 5, seed: int = 7
               ) -> tuple[pd.DataFrame, dict]:
    """No relationship at all. The app must report NO SIGNAL."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, p)),
                      columns=[f"Factor_{i+1}" for i in range(p)])
    df["Response"] = rng.normal(size=n)
    return df, {"drivers": [], "expect": "NO SIGNAL"}


def spurious_correlation_trap(n: int = 3, seed: int = 3
                              ) -> tuple[pd.DataFrame, dict]:
    """
    The real n=3 Eudragit case: mixture-constrained excipients, 4 factors,
    3 batches. A filler correlates at r~1.0 with the response purely by
    geometry. The app must refuse to model and must not name the decoy.
    """
    df = pd.DataFrame({
        "batch": ["F25", "F30", "F28"],
        "Ludipress_pct": [21.535, 19.035, 19.035],
        "Eudragit_pct": [25.0, 30.0, 28.0],
        "MCC_pct": [21.230, 18.928, 20.928],
        "Hardness_kP": [11.0, 10.0, 9.3],
        "Cmax_T": [4128.0752, 3120.4289, 3958.5169],
        "Cmax_R": [3291.6545, 3785.7429, 3300.6470],
    })
    df["Cmax_GMR"] = (df.Cmax_T / df.Cmax_R).round(4)
    return df, {
        "expect": "BLOCKED - rank deficient, mixture constrained",
        "true_driver": "Eudragit_pct",
        "decoy": "MCC_pct",
        "threshold_interval": (28.0, 30.0),
        "reference_drift_batch": "F30",
    }


def mixture_design(n: int = 30, seed: int = 5) -> tuple[pd.DataFrame, dict]:
    """Constrained mixture; Scheffe should beat ordinary regression."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet([2.0, 3.0, 2.0], n)
    lo = np.array([0.15, 0.30, 0.10])
    x = lo + raw * (1 - lo.sum())
    hardness = rng.uniform(7, 13, n)

    y = (40 * x[:, 0] + 95 * x[:, 1] + 25 * x[:, 2]
         - 120 * x[:, 0] * x[:, 1] + 0.4 * hardness
         + rng.normal(0, 0.8, n))

    df = pd.DataFrame({
        "Polymer": x[:, 0].round(4),
        "Filler": x[:, 1].round(4),
        "Disintegrant": x[:, 2].round(4),
        "Hardness_kP": hardness.round(2),
        "Q12h": y.round(3),
    })
    return df, {
        "mixture_columns": ["Polymer", "Filler", "Disintegrant"],
        "process_columns": ["Hardness_kP"],
        "expect": "Scheffe model selected or competitive",
    }


def percolation_series(seed: int = 11) -> tuple[pd.DataFrame, dict]:
    """Six polymer levels with a sharp threshold at 28.5%."""
    rng = np.random.default_rng(seed)
    poly = np.array([22.0, 25.0, 27.0, 28.0, 30.0, 33.0])
    xc = 28.5
    gmr = np.where(poly < xc,
                   1.30 - 0.012 * (poly - 22),
                   0.95 - 0.045 * (poly - xc))
    gmr = gmr + rng.normal(0, 0.008, len(poly))
    df = pd.DataFrame({"Polymer_pct": poly, "Cmax_GMR": gmr.round(4)})
    return df, {"x_critical": xc, "expect_interval": (28.0, 30.0)}


def ivivc_dataset(n_batches: int = 6, seed: int = 13
                  ) -> tuple[dict, dict]:
    """
    Paired in vitro / in vivo data with a genuine Level A relationship.
    Returns (data_dict, truth).
    """
    import fi_ivivc as IV

    rng = np.random.default_rng(seed)
    t_vitro = np.array([0, 0.25, 0.5, 1, 2, 4, 6, 8, 12, 16, 24.])
    t_vivo = np.linspace(0, 36, 145)
    ke, V, dose, F = 0.14, 22.0, 100.0, 0.85

    alphas = np.linspace(12, 46, n_batches)
    vitro, vivo, meta = {}, {}, []

    for i, a in enumerate(alphas):
        bid = f"IV{i+1:02d}"
        units = np.array([
            np.clip(D.weibull(t_vitro, a, 0.85) + rng.normal(0, 1.6, len(t_vitro)), 0, 100)
            for _ in range(12)
        ])
        units = np.maximum.accumulate(units, axis=1)
        vitro[bid] = units

        fd = np.clip(units.mean(axis=0) / 100.0, 0, 1)
        fa = np.clip(0.02 + 0.96 * np.interp(t_vivo, t_vitro, fd), 0, 1)
        pk = IV.convolve_to_pk(t_vivo, fa, dose, F, V, ke)
        cp = pk.Cp.to_numpy() * (1 + rng.normal(0, 0.02, len(t_vivo)))
        vivo[bid] = np.clip(cp, 0, None)
        m = IV.pk_metrics(t_vivo, cp)
        meta.append({"batch": bid, "Weibull_alpha": a, **m})

    return (
        {"t_vitro": t_vitro, "t_vivo": t_vivo,
         "vitro": vitro, "vivo": vivo, "pk": pd.DataFrame(meta)},
        {"ke": ke, "V": V, "dose": dose, "F": F,
         "expect": "Level A with slope near 1"},
    )


def multi_campaign_doe(n_per_campaign: int = 8, n_campaigns: int = 6,
                       icc_target: float = 0.55, seed: int = 21
                       ) -> tuple[pd.DataFrame, dict]:
    """
    The same matrix DoE run across several manufacturing campaigns, with a
    genuine campaign offset.

    Ground truth: the row count is 48 but the effective sample size is far
    lower. An analysis that pools these campaigns will report standard
    errors that are too small and will pass the design gate it should
    fail.
    """
    rng = np.random.default_rng(seed)
    n = n_per_campaign * n_campaigns
    campaign = np.repeat([f"C{i+1}" for i in range(n_campaigns)],
                         n_per_campaign)

    polymer = rng.uniform(18, 38, n)
    hardness = rng.uniform(7, 13, n)
    lubricant = rng.uniform(0.5, 2.0, n)

    s2e = 1.0
    s2u = icc_target / (1 - icc_target) * s2e
    offset = rng.normal(0, np.sqrt(s2u), n_campaigns)
    idx = np.repeat(np.arange(n_campaigns), n_per_campaign)

    q12 = (55.0 - 0.85 * (polymer - 28) - 0.30 * (hardness - 10)
           + offset[idx] + rng.normal(0, np.sqrt(s2e), n))

    df = pd.DataFrame({
        "batch": [f"M{i+1:03d}" for i in range(n)],
        "campaign": campaign,
        "Polymer_pct": polymer.round(2),
        "Hardness_kP": hardness.round(2),
        "Lubricant_pct": lubricant.round(3),
        "Q12h": q12.round(3),
    })
    return df, {
        "group_column": "campaign",
        "true_icc": float(s2u / (s2u + s2e)),
        "drivers": ["Polymer_pct", "Hardness_kP"],
        "decoys": ["Lubricant_pct"],
        "expect": "effective n far below the row count",
    }


def campaign_banded_doe(n_campaigns: int = 8, per_campaign: int = 10,
                        seed: int = 41) -> tuple[pd.DataFrame, dict]:
    """
    Each campaign runs its own polymer setpoint band, plus a campaign
    offset - the normal situation when a DoE is spread over time.

    Ground truth: the model can locate the campaign from the factor values
    and memorise its offset, so RANDOM cross-validation folds leak badly.
    Group-level folds expose that the model cannot predict a campaign it
    has never seen.
    """
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_campaigns), per_campaign)
    centres = np.linspace(20, 36, n_campaigns)
    polymer = centres[g] + rng.normal(0, 0.8, len(g))
    hardness = rng.uniform(8, 12, len(g))
    offset = rng.normal(0, 4.0, n_campaigns)
    q12 = 55 - 0.6 * (polymer - 28) + offset[g] + rng.normal(0, 1.0, len(g))

    df = pd.DataFrame({
        "batch": [f"K{i+1:03d}" for i in range(len(g))],
        "campaign": [f"C{i+1}" for i in g],
        "Polymer_pct": polymer.round(2),
        "Hardness_kP": hardness.round(2),
        "Q12h": q12.round(3),
    })
    return df, {
        "group_column": "campaign",
        "expect": "random-fold Q2 far above group-fold Q2",
    }


def crossed_site_campaign(n_sites: int = 5, n_campaigns: int = 6,
                          reps: int = 4, seed: int = 43
                          ) -> tuple[pd.DataFrame, dict]:
    """
    Site and campaign fully crossed - every site runs every campaign.

    Ground truth: site variance 1.5, campaign variance 0.5, residual 0.8.
    Nesting campaign inside site would charge site-to-site variation to
    campaigns and give the wrong answer about where the variability lives.
    """
    rng = np.random.default_rng(seed)
    site = np.tile(np.repeat(np.arange(n_sites), n_campaigns), reps)
    camp = np.tile(np.tile(np.arange(n_campaigns), n_sites), reps)
    n = len(site)
    us = rng.normal(0, np.sqrt(1.5), n_sites)
    uc = rng.normal(0, np.sqrt(0.5), n_campaigns)
    x = rng.normal(size=n)
    y = 2.0 * x + us[site] + uc[camp] + rng.normal(0, np.sqrt(0.8), n)

    df = pd.DataFrame({
        "site": [f"S{i+1}" for i in site],
        "campaign": [f"C{i+1}" for i in camp],
        "Polymer_pct": (28 + 3 * x).round(3),
        "Q12h": y.round(4),
    })
    return df, {"var_site": 1.5, "var_campaign": 0.5, "var_resid": 0.8,
                "expect": "both terms significant by LRT"}


def highly_variable_drug(seed: int = 31) -> tuple[pd.DataFrame, dict]:
    """
    A highly variable drug where unscaled ABE is close to unpassable but
    reference scaling is appropriate. Ground truth: products are
    equivalent (true GMR = 1.0) and CVwR = 45%.
    """
    return pd.DataFrame({
        "scenario": ["identical", "modest difference", "real difference"],
        "true_GMR": [1.00, 1.10, 1.30],
        "CVwR_pct": [45.0, 45.0, 45.0],
        "n_subjects": [36, 36, 36],
    }), {
        "cv_wr_pct": 45.0,
        "expect": "RSABE/ABEL pass the equivalent product; ABE struggles",
    }


def generate_all(outdir: str | None = None) -> dict:
    """Write every scenario to CSV for the app's demo loader."""
    import os

    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(outdir, exist_ok=True)
    written = {}

    doe, _ = matrix_tablet_doe()
    doe.to_csv(f"{outdir}/demo_matrix_doe.csv", index=False)
    written["matrix DoE (n=48)"] = f"{outdir}/demo_matrix_doe.csv"

    noise, _ = pure_noise()
    noise.to_csv(f"{outdir}/demo_pure_noise.csv", index=False)
    written["pure noise (n=30)"] = f"{outdir}/demo_pure_noise.csv"

    trap, _ = spurious_correlation_trap()
    trap.to_csv(f"{outdir}/demo_n3_eudragit.csv", index=False)
    written["real n=3 case"] = f"{outdir}/demo_n3_eudragit.csv"

    mix, _ = mixture_design()
    mix.to_csv(f"{outdir}/demo_mixture.csv", index=False)
    written["mixture design (n=30)"] = f"{outdir}/demo_mixture.csv"

    perc, _ = percolation_series()
    perc.to_csv(f"{outdir}/demo_percolation.csv", index=False)
    written["percolation series"] = f"{outdir}/demo_percolation.csv"

    mc, _ = multi_campaign_doe()
    mc.to_csv(f"{outdir}/demo_multi_campaign.csv", index=False)
    written["multi-campaign DoE (n=48)"] = f"{outdir}/demo_multi_campaign.csv"

    cb, _ = campaign_banded_doe()
    cb.to_csv(f"{outdir}/demo_campaign_banded.csv", index=False)
    written["campaign-banded DoE (CV leakage)"] = f"{outdir}/demo_campaign_banded.csv"

    cx, _ = crossed_site_campaign()
    cx.to_csv(f"{outdir}/demo_crossed_site_campaign.csv", index=False)
    written["crossed site x campaign"] = f"{outdir}/demo_crossed_site_campaign.csv"

    hvd, _ = highly_variable_drug()
    hvd.to_csv(f"{outdir}/demo_highly_variable.csv", index=False)
    written["highly variable drug"] = f"{outdir}/demo_highly_variable.csv"

    doe2, truth = matrix_tablet_doe()
    prof = pd.DataFrame(truth["profiles"].round(2),
                        columns=[f"t_{x:g}h" for x in truth["timepoints"]])
    prof.insert(0, "batch", doe2["batch"])
    prof.to_csv(f"{outdir}/demo_dissolution_profiles.csv", index=False)
    written["dissolution profiles"] = f"{outdir}/demo_dissolution_profiles.csv"

    return written


if __name__ == "__main__":
    for k, v in generate_all().items():
        print(f"{k:28s} -> {v}")
