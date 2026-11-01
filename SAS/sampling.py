"""Structured observation policies π (no N1 head).

Default `strided` keeps the paper Table 1 (ra, re) and replaces random
intra-epoch patches with an evenly spaced 1s grid. Train-time phase jitter
+ a small random-intra mix recover MASS-style regularization without
collapsing to "first k seconds" (the previous E1 failure mode).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from .config import (
    BAND_SLICES,
    FEAT_DIM,
    NUM_EPOCHS_WINDOW,
    NUM_PATCHES,
    budget_to_mass_mask_ratios,
    visible_patch_count,
)


@dataclass
class PolicyOutput:
    epoch_visible: torch.Tensor  # [B, Ev]
    epoch_masked: torch.Tensor  # [B, Em]
    patch_visible: torch.Tensor  # [B, Ev, Pv]
    freq_mask: Optional[torch.Tensor]  # [F], 1=keep
    time_ratio: float
    freq_ratio: float

    @property
    def overall_ratio(self) -> float:
        return self.time_ratio * self.freq_ratio


def _mask_len(num_tokens: int, mask_ratio: float) -> int:
    return int(math.ceil(num_tokens * float(mask_ratio) - 1e-9))


def _randperm_batch(num_tokens: int, batch_size: int, generator=None, device=None) -> torch.Tensor:
    rows = []
    for _ in range(batch_size):
        if generator is not None:
            rows.append(torch.randperm(num_tokens, generator=generator))
        else:
            rows.append(torch.randperm(num_tokens))
    out = torch.stack(rows, dim=0)
    return out.to(device) if device is not None else out


def _split_visible_masked(perm: torch.Tensor, mask_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    masked = perm[:, :mask_len]
    visible = perm[:, mask_len:]
    return visible, masked


def frequency_keep_mask(budget: float, bands: Optional[List[str]] = None, dim: int = FEAT_DIM) -> torch.Tensor:
    mask = torch.zeros(dim)
    if bands:
        for name in bands:
            if name not in BAND_SLICES:
                raise ValueError(f"Unknown band {name}. Choose from {list(BAND_SLICES)}")
            lo, hi = BAND_SLICES[name]
            mask[lo:hi] = 1.0
        if mask.sum() == 0:
            mask[0] = 1.0
        return mask
    keep = max(1, min(dim, int(round(dim * float(budget)))))
    mask[:keep] = 1.0
    return mask


def _uniquify_row(idx: torch.Tensor, n: int) -> torch.Tensor:
    """Make a 1-D index tensor unique while preserving order, filling from 0..n-1."""
    k = int(idx.numel())
    seen = set()
    out: List[int] = []
    for v in idx.tolist():
        iv = int(v) % n
        if iv not in seen:
            seen.add(iv)
            out.append(iv)
    fill = 0
    while len(out) < k:
        if fill not in seen:
            seen.add(fill)
            out.append(fill)
        fill += 1
        if fill >= n:
            break
    return torch.tensor(out[:k], device=idx.device, dtype=torch.long)


def strided_patch_idx(k: int, n: int, offset: int, device) -> torch.Tensor:
    """Evenly cover [0, n) with k patches, starting at `offset` (phase)."""
    if k >= n:
        return torch.arange(n, device=device, dtype=torch.long)
    k = max(1, k)
    step = n / float(k)
    pos = (float(offset % n) + torch.arange(k, device=device, dtype=torch.float32) * step) % n
    idx = pos.floor().long() % n
    return _uniquify_row(idx, n)


def uniform_patch_idx(k: int, n: int, device) -> torch.Tensor:
    if k >= n:
        return torch.arange(n, device=device, dtype=torch.long)
    k = max(1, k)
    idx = torch.linspace(0, n - 1, k, device=device).round().long().clamp(0, n - 1)
    return _uniquify_row(idx, n)


class ObservationPolicy(nn.Module):
    name = "base"

    def __init__(
        self,
        budget: float,
        num_epochs: int = NUM_EPOCHS_WINDOW,
        num_patches: int = NUM_PATCHES,
        seed: Optional[int] = 2025,
        mix_random: float = 0.0,
        align_inter: bool = True,
    ):
        super().__init__()
        self.budget = float(budget)
        self.num_epochs = num_epochs
        self.num_patches = num_patches
        self.mix_random = float(mix_random)
        self.align_inter = bool(align_inter)
        self.intra_mask, self.inter_mask = budget_to_mass_mask_ratios(budget)
        if not self.align_inter:
            self.inter_mask = 0.0
            self.intra_mask = max(0.0, 1.0 - self.budget)
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(int(seed))
        self.last_output: Optional[PolicyOutput] = None

    def sample(self, batch_size: int, device, eeg: Optional[torch.Tensor] = None) -> PolicyOutput:
        raise NotImplementedError

    def apply_frequency(self, eeg: torch.Tensor, freq_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if freq_mask is None:
            return eeg
        mask = freq_mask.to(device=eeg.device, dtype=eeg.dtype).view(1, 1, 1, -1)
        return eeg * mask

    def _epoch_split(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        e_mask_len = _mask_len(self.num_epochs, self.inter_mask)
        if e_mask_len <= 0:
            vis = torch.arange(self.num_epochs, device=device).unsqueeze(0).expand(batch_size, -1)
            msk = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
            return vis, msk
        e_perm = _randperm_batch(self.num_epochs, batch_size, self.generator, device)
        return _split_visible_masked(e_perm, e_mask_len)

    def _random_patches(self, batch_size: int, n_vis_epochs: int, device) -> torch.Tensor:
        k = visible_patch_count(self.intra_mask, self.num_patches)
        p_perm = _randperm_batch(self.num_patches, batch_size * n_vis_epochs, self.generator, device)
        return p_perm[:, :k].view(batch_size, n_vis_epochs, k)

    def _maybe_mix_random(self, structured: torch.Tensor, batch_size: int, n_vis_epochs: int, device) -> torch.Tensor:
        if (not self.training) or self.mix_random <= 0:
            return structured
        if torch.rand((), device=device).item() < self.mix_random:
            return self._random_patches(batch_size, n_vis_epochs, device)
        return structured

    def _pack(self, epoch_vis, epoch_msk, patch_vis, freq_mask=None, freq_ratio=1.0) -> PolicyOutput:
        ev = epoch_vis.shape[1]
        time_ratio = (ev / self.num_epochs) * (patch_vis.shape[-1] / self.num_patches)
        out = PolicyOutput(epoch_vis, epoch_msk, patch_vis, freq_mask, time_ratio, freq_ratio)
        self.last_output = out
        return out


class RandomPolicy(ObservationPolicy):
    """MASS random multi-level mask (in-folder baseline)."""

    name = "random"

    def sample(self, batch_size: int, device, eeg=None) -> PolicyOutput:
        epoch_vis, epoch_msk = self._epoch_split(batch_size, device)
        patch_vis = self._random_patches(batch_size, epoch_vis.shape[1], device)
        return self._pack(epoch_vis, epoch_msk, patch_vis)


class StridedPolicy(ObservationPolicy):
    """Evenly spaced 1s patches + train-time phase offset. Default SAS policy."""

    name = "strided"

    def sample(self, batch_size: int, device, eeg=None) -> PolicyOutput:
        epoch_vis, epoch_msk = self._epoch_split(batch_size, device)
        ev = epoch_vis.shape[1]
        k = visible_patch_count(self.intra_mask, self.num_patches)
        n = self.num_patches
        stride = max(1, n // k)
        if self.training:
            offset = int(torch.randint(0, stride, (1,), device=device).item())
        else:
            offset = 0
        idx = strided_patch_idx(k, n, offset, device)
        patch_vis = idx.view(1, 1, -1).expand(batch_size, ev, -1).contiguous()
        patch_vis = self._maybe_mix_random(patch_vis, batch_size, ev, device)
        return self._pack(epoch_vis, epoch_msk, patch_vis)


class UniformPolicy(ObservationPolicy):
    """Fixed linspace grid (no phase jitter). Ablation of strided."""

    name = "uniform"

    def sample(self, batch_size: int, device, eeg=None) -> PolicyOutput:
        epoch_vis, epoch_msk = self._epoch_split(batch_size, device)
        ev = epoch_vis.shape[1]
        k = visible_patch_count(self.intra_mask, self.num_patches)
        idx = uniform_patch_idx(k, self.num_patches, device)
        patch_vis = idx.view(1, 1, -1).expand(batch_size, ev, -1).contiguous()
        patch_vis = self._maybe_mix_random(patch_vis, batch_size, ev, device)
        return self._pack(epoch_vis, epoch_msk, patch_vis)


class DutyCyclePolicy(ObservationPolicy):
    """First-k seconds (wearable duty-cycle). Expected weaker; kept as ablation."""

    name = "duty_cycle"

    def sample(self, batch_size: int, device, eeg=None) -> PolicyOutput:
        epoch_vis, epoch_msk = self._epoch_split(batch_size, device)
        ev = epoch_vis.shape[1]
        k = visible_patch_count(self.intra_mask, self.num_patches)
        idx = torch.arange(k, device=device)
        patch_vis = idx.view(1, 1, -1).expand(batch_size, ev, -1).contiguous()
        patch_vis = self._maybe_mix_random(patch_vis, batch_size, ev, device)
        return self._pack(epoch_vis, epoch_msk, patch_vis)


class BandPolicy(ObservationPolicy):
    """Keep low-frequency (or named EEG) bins; all time patches stay visible."""

    name = "band"

    def __init__(self, budget: float, bands: Optional[List[str]] = None, **kwargs):
        kwargs.pop("mix_random", None)
        super().__init__(budget, mix_random=0.0, **kwargs)
        self.bands = bands
        self.register_buffer("freq_keep", frequency_keep_mask(budget, bands), persistent=False)
        self.intra_mask = 0.0
        self.inter_mask = 0.0

    def sample(self, batch_size: int, device, eeg=None) -> PolicyOutput:
        epoch_vis = torch.arange(self.num_epochs, device=device).unsqueeze(0).expand(batch_size, -1)
        epoch_msk = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
        patch_vis = torch.arange(self.num_patches, device=device).view(1, 1, -1).expand(
            batch_size, self.num_epochs, -1
        ).contiguous()
        freq = self.freq_keep.to(device)
        freq_ratio = float(freq.mean().item())
        return self._pack(epoch_vis, epoch_msk, patch_vis, freq, freq_ratio)


def build_policy(
    name: str,
    budget: float,
    bands: Optional[List[str]] = None,
    mix_random: float = 0.25,
    align_inter: bool = True,
    **kwargs,
) -> ObservationPolicy:
    name = name.lower()
    common = dict(mix_random=mix_random, align_inter=align_inter, **kwargs)
    if name == "random":
        return RandomPolicy(budget, mix_random=0.0, align_inter=True, **kwargs)
    if name == "strided":
        return StridedPolicy(budget, **common)
    if name == "uniform":
        return UniformPolicy(budget, **common)
    if name == "duty_cycle":
        return DutyCyclePolicy(budget, **common)
    if name == "band":
        return BandPolicy(budget, bands=bands, align_inter=False, **kwargs)
    raise ValueError(f"Unknown policy {name}. Choose from strided, uniform, duty_cycle, band, random")
