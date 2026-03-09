# insurance-calibration

Model calibration testing for insurance pricing. Answers three questions a pricing actuary needs before signing off a model:

1. **Is the model globally unbiased?** — does sum(predicted claims) equal sum(actual claims)?
2. **Is each price cohort self-financing?** — are low-risk and high-risk segments cross-subsidising each other?
3. **Does miscalibration come from poor levelling or poor ranking?** — is it cheaper to recalibrate or refit?

No Python library existed that answered these questions for insurance data — with exposure weighting, Poisson/Gamma/Tweedie deviance, and the Murphy decomposition framework. `scikit-learn`'s calibration module handles binary classification only. The R `actuaRE` package works only with its own model objects. This library takes raw prediction arrays and works with anything.

Based on Lindholm & Wüthrich (SAJ 2025) and Brauer et al. (arXiv:2510.04556, 2025).

---

## Installation

```bash
pip install insurance-calibration
```

Requires Python 3.10+. Dependencies: numpy, scipy >= 1.12, polars, matplotlib.

---

## Quickstart

```python
import numpy as np
from insurance_calibration import check_balance, murphy_decomposition, CalibrationChecker

# Load your holdout predictions
y = df["claim_frequency"].to_numpy()        # observed rate
y_hat = df["model_prediction"].to_numpy()   # predicted rate
exposure = df["earned_exposure"].to_numpy()  # years

# Global balance check
balance = check_balance(y, y_hat, exposure, distribution='poisson', seed=0)
print(f"Balance ratio: {balance.balance_ratio:.3f}")
print(f"95% CI: [{balance.ci_lower:.3f}, {balance.ci_upper:.3f}]")
print(f"Balanced: {balance.is_balanced}")

# Murphy decomposition — what is driving the deviance?
murphy = murphy_decomposition(y, y_hat, exposure, distribution='poisson')
print(f"\nMurphy Decomposition:")
print(f"  Uncertainty:     {murphy.uncertainty:.4f}")
print(f"  Discrimination:  {murphy.discrimination:.4f} ({murphy.discrimination_pct:.1f}%)")
print(f"  Miscalibration:  {murphy.miscalibration:.4f} ({murphy.miscalibration_pct:.1f}%)")
print(f"    Global MCB:    {murphy.global_mcb:.4f}  <- fixable by recalibration")
print(f"    Local MCB:     {murphy.local_mcb:.4f}  <- needs model refit")
print(f"\nVerdict: {murphy.verdict}")

# Full diagnostic report
checker = CalibrationChecker(distribution='poisson', alpha=0.05)
report = checker.check(y, y_hat, exposure, seed=0)
print(report.summary())
```

---

## The three diagnostics

### 1. Balance property test (`check_balance`)

The weakest and easiest to fix. Tests whether:

```
sum(v_i * y_i) = sum(v_i * mu_hat_i)
```

The balance ratio `alpha = sum(v*y) / sum(v*y_hat)` should be 1.0. The test uses a Poisson z-test for the p-value and bootstrap resampling for the confidence interval.

A model that is globally imbalanced is charging the wrong average premium. This is the most commercially important failure mode — the model's portfolio-level loss ratio will be wrong.

### 2. Auto-calibration test (`check_auto_calibration`)

A stronger, local property. Tests whether:

```
E[Y | mu_hat(X)] = mu_hat(X)   for all prediction levels
```

Every price cohort should be on average self-financing. If high-risk predictions are systematically too low and low-risk predictions too high (or vice versa), the model has shape miscalibration even if the overall level is correct.

GLMs with canonical links satisfy this on the training set by construction. GBMs and neural networks generally do not — they require isotonic recalibration.

The test bins predictions into quantile groups and either applies a bootstrap MCB test (Algorithm 1, Brauer et al. 2025) or a Hosmer-Lemeshow chi-squared test.

```python
from insurance_calibration import check_auto_calibration

result = check_auto_calibration(y, y_hat, exposure, n_bins=10, method='bootstrap', seed=0)
print(result.p_value)          # < 0.05 means auto-calibration is violated
print(result.worst_bin_ratio)  # 0.78 = worst cohort is 22% under-predicted
print(result.per_bin)          # Polars DataFrame with per-bin obs/exp
```

### 3. Murphy decomposition (`murphy_decomposition`)

Decomposes the total deviance into:

```
D(y, y_hat) = UNC - DSC + MCB
```

- **UNC** (Uncertainty): baseline deviance from an intercept-only model. Determined by data difficulty, not the model.
- **DSC** (Discrimination): improvement from having a well-ranked model. High DSC = good Gini.
- **MCB** (Miscalibration): excess deviance from wrong price levels. Should be near zero.

MCB splits further into:
- **GMCB** (Global MCB): fixable by multiplying all predictions by the balance ratio. Cheap to fix.
- **LMCB** (Local MCB): residual after balance correction. Requires model refit.

The verdict logic:
- `MCB / UNC < 1%` and `DSC > 0` → **OK**
- `GMCB >= LMCB` → **RECALIBRATE** (scale the model, preserve structure)
- `LMCB > GMCB` → **REFIT** (model structure is wrong)

---

## Rectification

```python
from insurance_calibration import rectify_balance, isotonic_recalibrate

# Multiplicative: multiply all predictions by alpha = sum(v*y) / sum(v*y_hat)
y_hat_corrected = rectify_balance(y_hat, y, exposure, method='multiplicative')

# Affine: fit log(mu) = beta_0 + beta_1 * log(y_hat), minimising Poisson deviance
# Corrects both global level and slope errors
y_hat_affine = rectify_balance(y_hat, y, exposure, method='affine', distribution='poisson')

# Isotonic: empirical auto-calibration (use on HOLDOUT data only)
y_hat_recal = isotonic_recalibrate(y, y_hat, exposure)
```

---

## Distributions

All functions accept a `distribution` parameter:

| Distribution | Use case | Notes |
|---|---|---|
| `'poisson'` | Claim frequency | y = claims/year, handles y=0 |
| `'gamma'` | Claim severity | y must be > 0 |
| `'tweedie'` | Pure premium | Specify `tweedie_power` (1 < p < 2) |
| `'normal'` | Any regression | Mean squared error |

---

## Exposure weighting

`exposure` is the policy duration in years (earned exposure). All statistics weight by exposure. If `exposure=None`, uniform weights are used.

The convention throughout: `y` is a **rate** (claims per year), not a count. `exposure` converts rates to counts for deviance calculations.

---

## Monitoring workflow

```python
checker = CalibrationChecker(
    distribution='poisson',
    alpha=0.32,    # Brauer et al. (2025) recommend 0.32 for monitoring, not 0.05
    n_bins=10,
    bootstrap_n=999,
)

# Reference period (model launch)
checker.fit(y_holdout, y_hat_holdout, exposure_holdout, seed=0)

# Ongoing monitoring — run each quarter
report = checker.check(y_new_quarter, y_hat_new_quarter, exposure_new_quarter, seed=0)
print(report.verdict())   # 'OK' | 'MONITOR' | 'RECALIBRATE' | 'REFIT'
print(report.summary())   # human-readable paragraph
df = report.to_polars()   # one-row DataFrame for logging to a monitoring table
```

Note on `alpha=0.32`: at the standard 0.05 level, a one-standard-deviation deterioration in calibration has very low detection probability. The economically costly error is failing to detect real miscalibration, not triggering false alarms. 0.32 corresponds to ~50% power for detecting a one-SD shift (Brauer et al., 2025).

---

## Visualisation

```python
from insurance_calibration._plots import (
    plot_auto_calibration,
    plot_murphy,
    plot_balance_over_time,
    plot_calibration_report,
)

# Auto-calibration reliability diagram
fig = plot_auto_calibration(result)

# Murphy decomposition bar chart
fig = plot_murphy(murphy_result)

# Balance ratio over time with CI band
fig = plot_balance_over_time(periods, ratios, lowers, uppers)

# Combined 3-panel report figure
fig = plot_calibration_report(report)
fig.savefig("calibration_report.png", dpi=150, bbox_inches="tight")
```

---

## Integration with insurance-validation

`insurance-validation` tests model discrimination (ranking ability): Gini, double lift, Lorenz curve.

`insurance-calibration` tests calibration (price level correctness): balance, auto-calibration, Murphy.

They are complementary. The intended workflow:

```
1. insurance-validation  →  Is the Gini acceptable?
   If Gini is too low: REFIT (problem is ranking, not levelling)

2. insurance-calibration →  Is the model well-calibrated?
   balance_ratio in [0.97, 1.03]? → proceed
   balance_ratio outside range?   → check Murphy
   MCB mostly GMCB?  → RECALIBRATE (cheap)
   MCB mostly LMCB?  → REFIT (expensive)
```

---

## References

1. Lindholm, M. and Wüthrich, M.V. (2025). "The Balance Property in Insurance Pricing." *Scandinavian Actuarial Journal*. DOI: 10.1080/03461238.2025.2552909

2. Brauer, A., Denuit, M., Krvavych, Y. et al. (2025). "Model Monitoring: A General Framework with an Application to Non-life Insurance Pricing." arXiv:2510.04556

3. Denuit, M., Charpentier, A. and Trufin, J. (2021). "Autocalibration and Tweedie-dominance for Insurance Pricing with Machine Learning." *Insurance: Mathematics and Economics*, 101(B), 485–497.

4. Wüthrich, M.V. and Ziegel, J. (2024). "Isotonic Recalibration under a Low Signal-to-Noise Ratio." *Scandinavian Actuarial Journal*, 2024(3), 279–299.

---

## Licence

MIT. Built by [Burning Cost](https://burningcost.github.io).
