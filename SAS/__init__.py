"""SAS: Structured Adaptive Sampling for resource-efficient sleep staging.

Structured observation only (no N1-specific head). Default policy is strided
patches with MASS-aligned inter-epoch masking.

    python -m SAS --smoke
    python -m SAS --dataset dreams --pi strided --budget 0.4
    python -m SAS --dataset shhs --pi strided --budget 0.4
"""

from . import _env  # noqa: F401
from .paths import ROOT, PACKAGE_DIR
from .config import DATASETS, PI_CHOICES, BUDGET_GRID

__all__ = ["ROOT", "PACKAGE_DIR", "DATASETS", "PI_CHOICES", "BUDGET_GRID"]
