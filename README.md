# Formulation Intelligence

Factor identification, dissolution modelling and virtual bioequivalence for
**small-n pharmaceutical formulation data**.

Built around one rule: below a minimum sample size, or whenever the design
matrix is rank deficient, **no statistical model is admissible** — and this
app refuses to fit one rather than return a number that looks authoritative
and is not.

That refusal is the feature. Small-n formulation data produces chance
correlations constantly. At n=3 with 4 factors there is a **73% probability**
that at least one *entirely random* factor reaches |r| ≥ 0.90 against a random
response, and a **32% probability** one reaches |r| ≥ 0.99. A correlation bar
chart at that sample size is not evidence, and a model fitted past the gate
will find the noise and present it as a driver.

---

## Quickstart

```bash
git clone <your-repo-url>
cd formulation-intelligence
pip install -r requirements.txt
python data/simulate.py          # generate demo datasets
streamlit run app.py
```

Verify the install:

```bash
python -m pytest . -q                  # 78 unit tests
python run_validation.py               # 75 ground-truth checks
```

---

## What it does

| Page | Purpose |
|---|---|
| **Main** | Load data, select response and factors, see the design gate verdict |
| **1 · Design Diagnostics** | Rank, mixture constraint, VIF, correlated clusters, chance-correlation risk |
| **2 · Model Arena** | Leaderboard: nested CV → Q² → 1-SE rule → y-randomisation |
| **3 · Factor Identification** | VIP, bootstrap coefficients, stability selection, permutation importance, SHAP |
| **4 · Dissolution Modelling** | Weibull/KP/Peppas-Sahlin fitting, fPCA, bootstrap f2, **discrimination power** |
| **5 · Percolation & Threshold** | Mechanistic mode for n<8: threshold location and prediction envelopes |
| **6 · IVIVC & Virtual BE** | Reference-drift check, deconvolution, Level A, Monte Carlo BE |
| **7 · Predict & Optimise** | Predictions with applicability-domain flags, response surfaces |
| **8 · Report** | Reproducible Markdown report with seeds and package versions |
| **9 · Mixed Effects** | Batch/site/campaign random intercepts, ICC, effective n |
| **10 · Reference-Scaled BE** | FDA RSABE and EMA ABEL for highly variable drugs |

---

## 1. Target type → model family

### Scalar response (hardness, Q30, %release at 12h, tensile strength)

| Data regime | Model | Why |
|---|---|---|
| Structured DoE, n<20 | OLS with 2FI/quadratic + Lenth's method | Effects estimable; ANOVA is regulator-familiar |
| n 20–60, collinear X | **PLS regression** | Pharma workhorse, handles p>n, VIP for factor ID, ICH Q8 accepted |
| n 20–80, nonlinear/smooth | **Gaussian Process (Matérn 5/2)** | Gives *prediction intervals* — essential for design space |
| n 60–300 | Random Forest / XGBoost | Captures thresholds (e.g. polymer % tipping point) |
| Mixture components | **Scheffé canonical, no intercept** | Ingredients sum to a constant; ordinary regression is invalid |

Mixture constraints are **auto-detected** (row-sum CV < 2%) and Scheffé models
are added automatically with the intercept suppressed.

### Dissolution profile

Never model each timepoint independently. Three routes, best first:

1. **Parameterise then predict** — fit Weibull (or Korsmeyer-Peppas /
   first-order) per formulation, model the 2–3 parameters against the factors,
   reconstruct the profile. Preserves monotonicity, needs far less data.
2. **Functional PCA** on the profile matrix, model the first 2–3 PC scores.
3. **Multi-output PLS (PLS2)** as a quick baseline.

The Peppas-Sahlin decomposition is included because the **erosion term** is
often the real factor-response variable for a matrix tablet, not Q30.

### Batch, site and campaign structure

Pooling across manufacturing campaigns and treating the result as one flat
dataset is the quiet error in formulation analytics. If batches differ
systematically, observations within a batch are correlated, pooled standard
errors are too small, and the **effective** sample size is well below the row
count.

The app fits a REML random-intercept model (self-contained, no extra
dependency) and reports the ICC, the design effect and the effective n. Select
a grouping column on the main page and **the design gate tests the effective n
instead of the row count**. On the bundled multi-campaign demo this is
decisive: 48 rows are admitted when the campaigns are ignored and **blocked at
an effective n of 8.9** once they are not.

Direction of the correction depends on where a factor varies:

| Factor type | Effect of the mixed model | Why |
|---|---|---|
| Varies **between** groups (site, RM lot, campaign) | SE **inflates** — 3.06× in the validation suite | Effective n is the number of groups, not rows |
| Varies **within** groups (polymer %, hardness) | SE **shrinks** | The group offset leaves the residual instead of sitting in it |

The between-group case is the dangerous one: pooled OLS reported an SE of
0.176 where the honest value was 0.537.

The random effect is tested by likelihood ratio against the **50:50 mixture of
a point mass at zero and χ²₁**, because the null sits on the boundary of the
parameter space. Using a plain χ²₁ roughly doubles the p-value and hides real
batch structure.

**Crossed factors.** Site and campaign are usually crossed, not nested — every
site runs every campaign — and nesting them charges site-to-site variation to
campaigns. Any number of independent random terms is supported, and
`lrt_terms` drops each in turn to show which one carries the structure. A
single pooled ICC cannot separate them.

**Random slopes.** When a factor's effect differs between sites (a polymer
level behaving differently on different compression lines), a random slope
stops that heterogeneity being counted as noise. Intercept and slope variances
are estimated independently; the correlation between them is **not** fitted —
this is lme4's `(1|g) + (0+x|g)`, not `(x|g)`. Estimating the correlation needs
more groups than formulation work usually provides.

**Satterthwaite degrees of freedom.** Computed from the observed REML
information, and the effect is large:

| Covariate | Satterthwaite df | Rows / groups |
|---|---|---|
| Varies **between** groups | **6.03** | n = 80, q = 8 |
| Varies **within** groups | **71.2** | n = 80, q = 8 |

Same model, same table, an order of magnitude apart. Using the row count for
both is how a between-group effect gets declared significant when it is not.

### BE outcome

The trap is treating it as pass/fail classification. With 5–20 BE studies a
classifier memorises. Instead:

1. **IVIVC (Level A)** — deconvolute in-vivo data (Wagner-Nelson for
   one-compartment, Loo-Riegelman for two, or model-independent numerical),
   regress fraction absorbed vs fraction dissolved.
2. Predict the dissolution profile from formulation factors → **convolve** to
   a PK profile.
3. **Monte Carlo** the BE study using the known intra-subject CV → output the
   *probability that the 90% CI of the GMR falls within 80–125%*.

That is a probability with a mechanistic chain behind it. A black-box
classifier on BE outcomes is not.

### Highly variable drugs (CVwR > 30%)

Unscaled ABE is close to unpassable once the within-subject CV of the
reference exceeds about 30% — the study needs enormous n even when the
products are genuinely identical. Both frameworks are implemented, and they
are **not** interchangeable:

| | FDA RSABE | EMA ABEL |
|---|---|---|
| Mechanism | Scales the *criterion* | Widens the *limits* |
| Constant | θ = (ln1.25/0.25)² = 0.7967 | k = 0.760 |
| Bound | Upper 95% by Howe's approximation | 90% CI vs exp(±k·swR) |
| Scaling from | CVwR ≥ 30% | CVwR > 30%, capped at 50% |
| Cap | none | 0.6984–1.4319 |
| Also requires | point estimate in 80–125% | point estimate in 80–125%, Cmax only |

Both need a replicate design, since CVwR must be estimated from repeated
administrations of the reference. The simulator makes the scaling decision
from the **observed** CVwR in each simulated study, exactly as a real study
would — which exposes a failure mode that is easy to miss: with a true CVwR of
32%, scaling was permitted in only **67%** of simulated studies, because the
rest observed a CVwR below the 30% threshold and lost the widened limits
entirely.

Howe's bound is validated by empirical type I error at the scaled boundary:
**4.88–4.89% against a nominal 5%** across CVwR of 35%, 45% and 60%. That check
is the only one that catches a mis-specified confidence bound.

---

## 2. How the app selects a model

```
detect task → design gate → candidate list → nested CV →
rank by Q² → 1-SE rule → y-randomisation → applicability domain
```

Hard-coded rules:

- **Validation** — LOOCV when n<30; repeated 5-fold ×10 otherwise. Never a
  single train/test split at these sizes. **When a grouping column is
  supplied, folds are drawn at the group level** (leave-one-group-out below
  9 groups, GroupKFold above), including for the permutation null.
- **Ranking** — Q² (predicted R²) and RMSECV. Training R² is displayed for
  diagnosis only and never used to rank.
- **1-SE rule** — among models within one standard error of the best Q², the
  *simplest* wins. This kills the "XGBoost wins by 0.01" trap.
- **Y-randomisation** — the response is shuffled and the model refitted. If
  the real Q² is not clearly above the null, the app reports **NO SIGNAL** and
  selects nothing. Non-negotiable.
- **Applicability domain** — leverage (Williams h\* = 3p/n) and Hotelling's T²
  on PCA scores. Every prediction outside the hull gets an extrapolation
  warning rather than a number.

### Cross-validation leakage

Random folds split a batch across train and test. If campaigns occupy distinct
regions of factor space — which is normal, since each campaign runs its own
setpoint band — the model locates the campaign from the factor values and
memorises its offset, and that offset is then credited as predictive skill.

On the bundled campaign-banded demo:

| Fold scheme | Q² |
|---|---|
| Repeated 5-fold (random) | **0.778** |
| Leave-one-group-out | **0.072** |

The model has essentially no ability to predict a campaign it has never seen,
yet random CV credits it with 78% of variance. Group folds cost nothing when
the grouping is meaningless (0.974 vs 0.975 on random labels), so there is no
reason not to use them.

**Cost control.** The permutation test is the expensive step, so it runs only
on contenders, on a cheaper single-pass CV, with a measured per-model budget.
A model too expensive to validate within budget is **excluded from selection**
— an unvalidated model is not a candidate.

---

## 3. Factor identification

Several methods, reported by agreement, because correlated factors mislead any
single one:

- **VIP scores** from PLS (>1 = influential)
- **Bootstrap coefficients** with percentile CIs and sign consistency
- **Permutation importance** with confidence bands
- **Stability selection** — bootstrap Elastic Net, selection frequency per
  factor. Far more trustworthy than one fit: a factor selected in 90% of
  subsamples is real, one at 40% is a coin flip however large its coefficient.
- **SHAP** for tree/GP models (optional dependency)
- **Correlated clusters** — factors in one cluster are reported as a *group*;
  splitting importance between them is meaningless

A `votes` column counts how many independent methods flag each factor. One
vote means design a better experiment, not a conclusion.

---

## The small-n mechanistic mode

When the gate blocks statistical modelling, the app routes to percolation
analysis. With 3–6 batches and a monotonic driver you cannot fit a response
surface, but you **can**:

- Compute the **piecewise log-slope** between levels. A large jump is a
  threshold — the single most informative statistic available at n=3.
- Fit `k ~ |x − xc|^μ` with **μ fixed** at the 3-D theoretical value (≈1.9), so
  that a residual degree of freedom remains. A free exponent would fit three
  points exactly and demonstrate nothing.
- Produce a **prediction envelope** at an untested level, bounded between a
  smooth log-linear interpolant and a sharp step swept across the bracketing
  interval.

The envelope width *is* the answer. An envelope spanning more than half the BE
window is flagged **UNINFORMATIVE** even when it sits technically inside it —
it says the outcome is undetermined, not that the batch will pass.

### Reference-arm drift

Comparing test products **across separate BE studies** is confounded with the
reference arm, subject panel and RLD lot. The app recomputes every GMR against
a common reference and flags studies whose reference deviates by more than
10%. In the bundled n=3 demo this moves one batch's GMR from 0.824 to 0.945 —
the difference between "slow formulation" and "measured against a high
reference".

---

## Validation

`python run_validation.py` → **75/75 checks passed** (~140 s).

Every check has a known ground truth. Highlights:

| Check | Result |
|---|---|
| n=3 rank-deficient design blocked | rank=2 < p=4 ✅ |
| Decoy tops the naive correlation ranking | r = +1.0000 ✅ (the trap the gate exists to stop) |
| Pure noise (n=30) reports NO SIGNAL | best Q² = −0.055 ✅ |
| Planted driver recovered from n=48 | Polymer_pct ranked 1st, decoy 0 votes ✅ |
| Weibull parameters recovered | α=30.00, β=0.850, R²=1.0000 ✅ |
| Identical profiles flagged non-discriminating | ✅ |
| Wagner-Nelson round trip | max abs error 0.024 ✅ |
| BE power monotone in n and in CV | ✅ |
| Percolation threshold located | found [28, 30], true εc=28.5 ✅ |
| Wide envelope flagged UNINFORMATIVE | 83% of the BE window ✅ |
| Extrapolation refused | leverage 107.5 > h*=0.25 ✅ |
| ICC estimator unbiased | true 0.60 → 0.597; true 0.00 → 0.015 ✅ |
| Gate blocks on effective n | 48 rows admitted pooled, blocked at effective n=8.9 ✅ |
| Between-group covariate false precision caught | SE inflates 3.06× ✅ |
| FDA RSABE type I error at the boundary | 0.0488 / 0.0489 / 0.0489 vs nominal 0.05 ✅ |
| EMA limits at the CVwR cap | 0.6984–1.4319 ✅ |
| Scaling withheld below CVwR 30% | applied in 0.0% of studies ✅ |
| Point-estimate constraint binds | p_pass 0.289 with, 0.533 without ✅ |
| Group CV exposes leakage | random Q²=0.778 vs group Q²=0.072 ✅ |
| Group CV costs nothing on meaningless labels | 0.974 vs 0.975 ✅ |
| Crossed variance components unbiased | site 1.559/1.50, campaign 0.400/0.50 ✅ |
| Random-slope variances unbiased | 0.962/1.00, 0.631/0.60, 0.254/0.25 ✅ |
| Satterthwaite df, between-group | 6.03 against q−2 = 6 ✅ |
| Satterthwaite df, within-group | 71.2 against n−q−1 = 71 ✅ |

Results are written to `validation_results.csv`.

---

## Repository layout

```
formulation-intelligence/
├── app.py                       # Streamlit entry point
├── requirements.txt
├── core/
│   ├── design_diagnostics.py    # the admissibility gate
│   ├── validation.py            # Q², CV schemes, 1-SE, y-randomisation
│   ├── models.py                # model zoo, Scheffé, Lenth
│   ├── model_arena.py           # leaderboard engine
│   ├── applicability.py         # leverage, Hotelling T², GP intervals
│   ├── factor_id.py             # VIP, stability selection, SHAP, clusters
│   ├── dissolution.py           # profile models, fPCA, f2 CI, discrimination
│   ├── ivivc.py                 # deconvolution, Level A, convolution
│   ├── be_simulation.py         # virtual BE, power, reference drift
│   └── percolation.py           # threshold detection, envelopes
├── pages/                       # eight Streamlit pages
├── data/simulate.py             # synthetic data with known ground truth
├── tests/                       # 31 unit tests
└── validation/run_validation.py # 38 ground-truth checks
```

---

## Interpretation notes

- **f2 is unstable.** Report the lower bound of the bootstrap 90% CI, not the
  point estimate.
- **Fix seeds and pin versions.** Every page reads the session seed; the Report
  page records it with package versions.
- **Scale inside the Pipeline**, never on full-data statistics before CV.
- **Batch/site as a random effect** — select a grouping column on the main
  page. The gate will then test the effective n. If the ICC exceeds 0.10, any
  conclusion from a pooled analysis of that dataset needs re-checking.
- **Select the grouping column** whenever rows span campaigns. It switches the
  gate to effective n *and* switches cross-validation to group folds.
- **If your dissolution method cannot rank-order batches you know differ in
  vivo**, change the method, not the model. The Dissolution page's
  discrimination test tells you which, and suggests the axes to vary (ionic
  strength and counterion, early sampling, mechanical stress, vessel-to-vessel
  %RSD rather than mean profiles).

---

## Limitations

- The BE simulator assumes log-normal PK. Reference-scaled ABE is implemented
  for full and partial replicate designs; other replicate structures are not.
- Mixed models fit independent random terms only. The correlation between a
  random intercept and a random slope on the same factor is not estimated.
- With crossed factors the reported design effect and effective n come from
  the dominant grouping factor against the total ICC. There is no single
  cluster size for crossed designs, so read the per-term variances and the
  LRT table instead; the fit warns about this.
- Satterthwaite degrees of freedom use numerical derivatives of the REML
  likelihood. They are accurate on the designs in the validation suite but
  will be slower and less stable with many variance components.
- Numerical deconvolution requires a unit impulse response (IV or oral
  solution); without one, use Wagner-Nelson or Loo-Riegelman.
- Level A is fitted as linear regression. Non-linear correlations and time
  scaling are reported but not automatically applied.
- No PBPK/ACAT compartment model — regional absorption is not simulated. For
  colonic-window questions, pair this with a dedicated PBPK tool.

## License

MIT
