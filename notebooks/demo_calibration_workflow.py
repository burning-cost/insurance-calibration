# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-calibration: Full Workflow Demo
# MAGIC
# MAGIC This notebook demonstrates the full calibration testing workflow on synthetic
# MAGIC motor insurance frequency data. It shows:
# MAGIC
# MAGIC 1. A well-calibrated GBM-style model (baseline)
# MAGIC 2. A globally miscalibrated model (30% over-prediction)
# MAGIC 3. A shape-miscalibrated model (flat predictions, wrong structure)
# MAGIC 4. The Murphy decomposition for each, with verdict
# MAGIC 5. Rectification methods
# MAGIC
# MAGIC All data is synthetic. The schema reflects UK motor insurance conventions:
# MAGIC - `y`: claim frequency (claims per year)
# MAGIC - `y_hat`: model-predicted claim frequency
# MAGIC - `exposure`: earned exposure in years

# COMMAND ----------

# MAGIC %pip install insurance-calibration

# COMMAND ----------

# MAGIC %md ## 1. Generate synthetic motor insurance data

# COMMAND ----------

import numpy as np
import polars as pl
import warnings
warnings.filterwarnings("ignore")

from insurance_calibration import (
    check_balance,
    check_auto_calibration,
    murphy_decomposition,
    rectify_balance,
    isotonic_recalibrate,
    CalibrationChecker,
)

rng = np.random.default_rng(2025)

N = 10_000  # 10,000 policies on holdout

# Risk factors (synthetic)
vehicle_age = rng.integers(0, 15, N)       # vehicle age in years
driver_age = rng.integers(17, 80, N)        # driver age
vehicle_value = rng.gamma(3, 8000, N)       # vehicle value in GBP
miles_pa = rng.gamma(2, 5000, N)            # annual mileage

# True log-rate (true underlying model)
log_true_rate = (
    -3.5
    + 0.02 * np.maximum(25 - driver_age, 0)   # young driver surcharge
    - 0.008 * np.maximum(vehicle_age - 5, 0)  # older vehicles slightly lower
    + 0.00002 * miles_pa                       # mileage loading
    + rng.normal(0, 0.2, N)                    # heterogeneity
)
true_rate = np.exp(log_true_rate)

# Observed claims (Poisson process)
exposure = rng.uniform(0.3, 1.0, N)         # partial years on holdout
counts = rng.poisson(exposure * true_rate)
y_observed = counts / exposure               # claim frequency (rate)

# Well-calibrated model prediction (slight noise from finite training)
log_pred_good = log_true_rate + rng.normal(0, 0.1, N)
y_hat_good = np.exp(log_pred_good)

# Globally miscalibrated: systematic 30% over-prediction
y_hat_overpredict = y_hat_good * 1.30

# Shape-miscalibrated: predictions flattened toward grand mean
grand_mean = float(np.sum(exposure * y_observed) / np.sum(exposure))
y_hat_flat = 0.4 * y_hat_good + 0.6 * grand_mean
# Rescale to be globally balanced
alpha_flat = np.sum(exposure * y_observed) / np.sum(exposure * y_hat_flat)
y_hat_flat *= alpha_flat

print(f"Dataset: {N:,} policies, total exposure = {exposure.sum():.0f} years")
print(f"Mean observed frequency: {np.average(y_observed, weights=exposure):.4f}")
print(f"Mean good prediction:    {np.average(y_hat_good, weights=exposure):.4f}")
print(f"Mean over-prediction:    {np.average(y_hat_overpredict, weights=exposure):.4f}")
print(f"Mean flat prediction:    {np.average(y_hat_flat, weights=exposure):.4f}")

# COMMAND ----------

# MAGIC %md ## 2. Balance property test

# COMMAND ----------

print("=" * 60)
print("BALANCE PROPERTY TEST")
print("=" * 60)

for label, y_hat in [
    ("Good model", y_hat_good),
    ("Over-prediction (+30%)", y_hat_overpredict),
    ("Flat (shape error)", y_hat_flat),
]:
    result = check_balance(y_observed, y_hat, exposure, distribution="poisson", seed=42)
    status = "OK" if result.is_balanced else "IMBALANCED"
    print(f"\n{label}:")
    print(f"  Balance ratio: {result.balance_ratio:.4f}")
    print(f"  95% CI:        [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    print(f"  p-value:       {result.p_value:.4f}")
    print(f"  Verdict:       {status}")

# COMMAND ----------

# MAGIC %md ## 3. Auto-calibration test

# COMMAND ----------

print("=" * 60)
print("AUTO-CALIBRATION TEST")
print("=" * 60)

for label, y_hat in [
    ("Good model", y_hat_good),
    ("Over-prediction (+30%)", y_hat_overpredict),
    ("Flat (shape error)", y_hat_flat),
]:
    result = check_auto_calibration(
        y_observed, y_hat, exposure,
        distribution="poisson", n_bins=10,
        method="hosmer_lemeshow",  # faster than bootstrap for demo
        seed=42
    )
    status = "CALIBRATED" if result.is_calibrated else "NOT CALIBRATED"
    print(f"\n{label}:")
    print(f"  p-value:         {result.p_value:.4f}")
    print(f"  MCB score:       {result.mcb_score:.6f}")
    print(f"  Worst bin ratio: {result.worst_bin_ratio:.4f}")
    print(f"  Verdict:         {status}")

# COMMAND ----------

# MAGIC %md ## 4. Murphy decomposition

# COMMAND ----------

print("=" * 60)
print("MURPHY DECOMPOSITION")
print("=" * 60)

for label, y_hat in [
    ("Good model", y_hat_good),
    ("Over-prediction (+30%)", y_hat_overpredict),
    ("Flat (shape error)", y_hat_flat),
]:
    result = murphy_decomposition(y_observed, y_hat, exposure, distribution="poisson")
    print(f"\n{label}:")
    print(f"  Total deviance: {result.total_deviance:.6f}")
    print(f"  UNC:            {result.uncertainty:.6f}")
    print(f"  DSC:            {result.discrimination:.6f} ({result.discrimination_pct:.1f}%)")
    print(f"  MCB:            {result.miscalibration:.6f} ({result.miscalibration_pct:.1f}%)")
    print(f"    GMCB:         {result.global_mcb:.6f}  <- fixable by recalibration")
    print(f"    LMCB:         {result.local_mcb:.6f}  <- needs refit")
    print(f"  VERDICT:        {result.verdict}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Interpretation:**
# MAGIC - The good model has low MCB (miscalibration) relative to DSC (discrimination)
# MAGIC - The over-predicted model has high GMCB (global), fixed by multiplying by balance ratio
# MAGIC - The flat model has high LMCB (local), needs refit — balance correction doesn't help much

# COMMAND ----------

# MAGIC %md ## 5. Rectification

# COMMAND ----------

print("=" * 60)
print("RECTIFICATION")
print("=" * 60)

# Fix the over-predicted model
y_hat_fixed_mult = rectify_balance(y_hat_overpredict, y_observed, exposure, method="multiplicative")
y_hat_fixed_affine = rectify_balance(y_hat_overpredict, y_observed, exposure, method="affine")

from insurance_calibration import poisson_deviance

dev_original = poisson_deviance(y_observed, y_hat_overpredict, exposure)
dev_mult = poisson_deviance(y_observed, y_hat_fixed_mult, exposure)
dev_affine = poisson_deviance(y_observed, y_hat_fixed_affine, exposure)

print(f"\nOver-predicted model rectification:")
print(f"  Original deviance:      {dev_original:.6f}")
print(f"  After multiplicative:   {dev_mult:.6f}  (balance ratio: {np.sum(exposure*y_observed)/np.sum(exposure*y_hat_overpredict):.4f})")
print(f"  After affine:           {dev_affine:.6f}")

# Isotonic recalibration
y_hat_isotonic = isotonic_recalibrate(y_observed, y_hat_overpredict, exposure)
y_hat_isotonic_clipped = np.maximum(y_hat_isotonic, 1e-10)
dev_isotonic = poisson_deviance(y_observed, y_hat_isotonic_clipped, exposure)
print(f"  After isotonic:         {dev_isotonic:.6f}  (empirical auto-calibration)")

# COMMAND ----------

# MAGIC %md ## 6. Full CalibrationChecker pipeline

# COMMAND ----------

print("=" * 60)
print("CALIBRATION CHECKER PIPELINE")
print("=" * 60)

checker = CalibrationChecker(
    distribution="poisson",
    alpha=0.05,
    n_bins=10,
    bootstrap_n=499,
    autocal_method="hosmer_lemeshow",
)

# Run on over-predicted model
report = checker.check(y_observed, y_hat_overpredict, exposure, seed=42)

print("\nOver-predicted model report:")
print(report.summary())
print(f"\nFinal verdict: {report.verdict()}")

# Machine-readable output
df = report.to_polars()
print(f"\nReport as Polars DataFrame: {df.shape[1]} columns")
display(df.select([
    "balance_balance_ratio", "balance_is_balanced",
    "murphy_total_deviance", "murphy_discrimination_pct", "murphy_miscalibration_pct",
    "murphy_verdict", "verdict"
]))

# COMMAND ----------

# MAGIC %md ## 7. Per-bin reliability table

# COMMAND ----------

print("Per-bin reliability diagnostics (over-predicted model):")
display(report.auto_calibration.per_bin)

# COMMAND ----------

# MAGIC %md ## 8. Gamma model (severity example)

# COMMAND ----------

print("=" * 60)
print("GAMMA SEVERITY MODEL EXAMPLE")
print("=" * 60)

# Simulate claim severity data (only policies with at least one claim)
n_claims = 800
y_severity = rng.gamma(shape=2, scale=2500, size=n_claims)   # observed severity
y_hat_sev = rng.gamma(shape=2, scale=2500, size=n_claims) * 0.9  # biased down 10%
claim_counts = rng.integers(1, 5, n_claims).astype(float)    # weight = number of claims

result = check_balance(y_severity, y_hat_sev, claim_counts, distribution="gamma", seed=0)
print(f"\nSeverity model balance:")
print(f"  Balance ratio: {result.balance_ratio:.4f}")
print(f"  Balanced: {result.is_balanced}")
print(f"  (Model under-estimates severity by ~10%)")

murphy_sev = murphy_decomposition(y_severity, y_hat_sev, claim_counts, distribution="gamma")
print(f"\nSeverity Murphy decomposition:")
print(f"  Verdict: {murphy_sev.verdict}")
print(f"  MCB: {murphy_sev.miscalibration:.6f} ({murphy_sev.miscalibration_pct:.1f}%)")
print(f"  GMCB: {murphy_sev.global_mcb:.6f} (fixable by recalibration)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC The three diagnostic tests form a hierarchy:
# MAGIC
# MAGIC | Test | What it detects | Cheapest fix |
# MAGIC |------|----------------|--------------|
# MAGIC | Balance | Global level error | Multiply by alpha |
# MAGIC | Auto-calibration | Cohort-level errors | Isotonic recalibration |
# MAGIC | Murphy GMCB > LMCB | Global structure wrong | Multiplicative/affine |
# MAGIC | Murphy LMCB > GMCB | Local structure wrong | Refit the model |
# MAGIC
# MAGIC For UK motor frequency models, a balance ratio outside [0.97, 1.03] typically
# MAGIC triggers a calibration review before the rating table is filed.
