"""insurance-calibration: Model calibration testing for insurance pricing.

Implements the three-property framework from Lindholm & Wüthrich (SAJ 2025)
and the Murphy score decomposition from Brauer et al. (arXiv:2510.04556, 2025).

Three questions answered before model go-live:

1. **Is the model globally unbiased?** — :func:`check_balance`
2. **Is each price cohort self-financing?** — :func:`check_auto_calibration`
3. **Does miscalibration come from poor levelling or poor ranking?** — :func:`murphy_decomposition`

Rectification methods:
- :func:`rectify_balance` — multiplicative or affine correction
- :func:`isotonic_recalibrate` — empirical auto-calibration (holdout data only)

Pipeline class:
- :class:`CalibrationChecker` — fit/check workflow for monitoring pipelines

All functions accept ``(y_actual, y_predicted, exposure)`` arrays.
Exposure-weighted throughout. Model-agnostic — works with any prediction array.

Distributions supported: 'poisson', 'gamma', 'tweedie', 'normal'.
"""

from __future__ import annotations

from ._balance import check_balance
from ._autocal import check_auto_calibration
from ._murphy import murphy_decomposition
from ._rectify import rectify_balance, isotonic_recalibrate
from ._deviance import (
    deviance,
    poisson_deviance,
    gamma_deviance,
    tweedie_deviance,
    normal_deviance,
)
from ._types import (
    BalanceResult,
    AutoCalibResult,
    MurphyResult,
    CalibrationReport,
)
from .report import CalibrationChecker

__version__ = "0.1.0"
__all__ = [
    # Functional API
    "check_balance",
    "check_auto_calibration",
    "murphy_decomposition",
    "rectify_balance",
    "isotonic_recalibrate",
    # Deviance functions
    "deviance",
    "poisson_deviance",
    "gamma_deviance",
    "tweedie_deviance",
    "normal_deviance",
    # Result types
    "BalanceResult",
    "AutoCalibResult",
    "MurphyResult",
    "CalibrationReport",
    # Pipeline class
    "CalibrationChecker",
    "__version__",
]
