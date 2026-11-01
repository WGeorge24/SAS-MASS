"""SAS experiment configuration: structured sampling, MASS-aligned training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
NUM_CLASSES = 5
NUM_PATCHES = 30
NUM_EPOCHS_WINDOW = 32
FEAT_DIM = 128
STEP = 32

BAND_SLICES = {
    "delta": (0, 10),
    "theta": (10, 20),
    "alpha": (20, 31),
    "beta": (31, 77),
    "gamma": (77, 128),
}


@dataclass
class DatasetConfig:
    name: str
    rel_dir: str
    data_key: str
    split: str
    batch_size: int
    file_glob: str
    enabled: bool = True
    notes: str = ""

    @property
    def data_dir(self) -> Path:
        return ROOT / self.rel_dir


DATASETS: Dict[str, DatasetConfig] = {
    "dreams": DatasetConfig("dreams", "dataset/DREAMS", "PSD", "loso", 8, "subject*.mat"),
    "edf20": DatasetConfig("edf20", "dataset/EDF20", "PSD100", "subject20", 16, "SC4*.mat"),
    "edf78": DatasetConfig("edf78", "dataset/EDF78", "PSD100", "subject20", 64, "SC4*.mat"),
    "shhs": DatasetConfig(
        "shhs", "dataset/SHHS", "PSD", "subject20", 128, "shhs1-*.mat",
        enabled=True, notes="329 SHHS-1 recordings (AttnSleep subset), C4-A1, 20-fold.",
    ),
}

DEFAULT_INTER_VISIBLE = 0.8
PI_CHOICES = ("strided", "uniform", "duty_cycle", "band", "random")
BUDGET_GRID: List[float] = [0.10, 0.20, 0.40, 1.00]

# Ai et al., AAAI 2026, Table 1. Nominal signal integrity -> (ra, re).
# MASS MaskGenerator uses ceil, so the realized visible fraction is
# 72%: 24/30 patches x 28/32 epochs = 0.70
# 40%: 15/30 patches x 25/32 epochs = 0.390625
# 10%:  6/30 patches x 16/32 epochs = 0.10
# Keys are round(budget, 2). Other budgets keep the inter_visible=0.8 formula.
PAPER_TABLE1_MASK: Dict[float, Tuple[float, float]] = {
    0.72: (0.2, 0.1),
    0.40: (0.5, 0.2),
    0.10: (0.8, 0.5),
}


def budget_to_mass_mask_ratios(
    budget: float, inter_visible: float = DEFAULT_INTER_VISIBLE
) -> Tuple[float, float]:
    """(intra_mask_ratio, inter_mask_ratio).

    Table 1 budgets 0.72 / 0.40 / 0.10 use the paper's (ra, re).
    A non-default inter_visible, or any other budget, falls back to
    visible product ≈ budget with that inter-epoch visibility.
    """
    budget = float(budget)
    if budget >= 0.999:
        return 0.0, 0.0
    if abs(float(inter_visible) - DEFAULT_INTER_VISIBLE) < 1e-12:
        paper = PAPER_TABLE1_MASK.get(round(budget, 2))
        if paper is not None:
            return paper
    inter_visible = min(max(float(inter_visible), budget), 1.0)
    inter_mask = round(1.0 - inter_visible, 8)
    intra_visible = min(1.0, budget / max(inter_visible, 1e-6))
    intra_mask = max(0.0, round(1.0 - intra_visible, 8))
    return intra_mask, inter_mask


def visible_patch_count(intra_mask: float, num_patches: int = NUM_PATCHES) -> int:
    import math
    mask_len = int(math.ceil(num_patches * float(intra_mask) - 1e-9))
    return max(1, num_patches - mask_len)


@dataclass
class ModelHParams:
    embed_dim: int = 128
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    num_patches: int = NUM_PATCHES
    num_epoches: int = NUM_EPOCHS_WINDOW
    global_encoder_depth: int = 4
    intra_encoder_depth: int = 4
    inter_encoder_depth: int = 2
    inter_hidden_dim: int = 256
    feat_dim: int = FEAT_DIM


@dataclass
class TrainHParams:
    lr: float = 1e-4
    weight_decay: float = 0.01
    total_epochs: int = 100
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    seed: int = 2025
    grad_clip: float = 5.0
    cosine_weight: float = 2.0
    transition_weight: float = 0.5
    select_metric: str = "acc"


@dataclass
class RunConfig:
    dataset: str = "dreams"
    pi: str = "strided"
    budget: float = 0.40
    mix_random: float = 0.25
    align_inter: bool = True
    bands: Optional[List[str]] = None
    fold: Optional[int] = None
    max_folds: Optional[int] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    seed: int = 2025
    device: str = "cuda"
    fast: bool = False
    model: ModelHParams = field(default_factory=ModelHParams)
    train: TrainHParams = field(default_factory=TrainHParams)

    @property
    def run_tag(self) -> str:
        return f"{self.dataset}_{self.pi}_B{self.budget:.2f}"
