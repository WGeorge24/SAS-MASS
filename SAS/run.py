"""CLI for SAS structured-sampling sleep staging.

From the repository root:

    python -m SAS --smoke
    python -m SAS --dataset dreams --pi strided --budget 0.4
    python -m SAS --dataset dreams --pi random --budget 0.4
    python -m SAS --dataset edf20 --pi strided --budget 0.4
    python -m SAS --dataset edf78 --pi strided --budget 0.4
    python -m SAS --dataset shhs --pi strided --budget 0.4
    python -m SAS --dataset dreams --pi strided --fast --fold 0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from . import _env  # noqa: F401
from .config import DATASETS, PI_CHOICES, RunConfig
from . import data as D
from . import sampling as S
from . import train as T
from .metrics import format_metrics, mean_std
from .model import SAS
from .paths import PACKAGE_DIR, ROOT

import torch


def build_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sas")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAS: structured sampling for resource-efficient sleep staging"
    )
    p.add_argument("--dataset", default="dreams", choices=list(DATASETS.keys()))
    p.add_argument("--pi", default="strided", choices=list(PI_CHOICES),
                   help="strided=default SAS; random=MASS replica; duty_cycle/uniform/band=ablations")
    p.add_argument("--budget", type=float, default=0.40, help="Observation budget B in (0, 1]")
    p.add_argument("--mix_random", type=float, default=0.25,
                   help="Train-only fraction of batches that use random intra patches (strided/uniform/duty_cycle)")
    p.add_argument("--no_align_inter", action="store_true",
                   help="Keep all 32 epochs visible (old E1 geometry). Default is MASS-aligned inter-mask.")
    p.add_argument("--bands", default=None, help="Comma-separated EEG bands for --pi band")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--max_folds", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fast", action="store_true", help="1 fold, 2 epochs — sanity check")
    p.add_argument("--smoke", action="store_true", help="Dummy-tensor forward/backward, no data")
    return p.parse_args()


def make_run_config(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig(
        dataset=args.dataset,
        pi=args.pi,
        budget=args.budget,
        mix_random=0.0 if args.pi in ("random", "band") else args.mix_random,
        align_inter=not args.no_align_inter,
        bands=[x.strip() for x in args.bands.split(",")] if args.bands else None,
        fold=args.fold,
        max_folds=args.max_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        fast=args.fast,
    )
    if args.epochs is not None:
        cfg.train.total_epochs = args.epochs
        cfg.train.warmup_epochs = max(1, args.epochs // 10)
    if args.fast:
        cfg.train.total_epochs = min(2, cfg.train.total_epochs)
        cfg.train.warmup_epochs = 0
        cfg.fold = 0 if cfg.fold is None else cfg.fold
        cfg.max_folds = 1
    cfg.train.seed = args.seed
    return cfg


def build_model(cfg: RunConfig, mean, std, device: torch.device) -> SAS:
    policy = S.build_policy(
        cfg.pi,
        cfg.budget,
        bands=getattr(cfg, "bands", None),
        num_epochs=cfg.model.num_epoches,
        num_patches=cfg.model.num_patches,
        seed=cfg.seed,
        mix_random=cfg.mix_random,
        align_inter=getattr(cfg, "align_inter", True),
    )
    model = SAS(
        hparams=cfg.model,
        policy=policy,
        mean=mean,
        std=std,
        seed=cfg.seed,
    )
    return model.to(device)


def smoke_test(cfg: RunConfig, logger: logging.Logger) -> None:
    device = T.get_device(cfg.device)
    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)
    model = build_model(cfg, mean, std, device)
    b, t, p, f = 2, cfg.model.num_epoches, cfg.model.num_patches, cfg.model.feat_dim
    eeg = torch.randn(b, t, p, f, device=device)
    label = torch.randint(0, 5, (b, t), device=device)
    trans = (torch.rand(b, t, device=device) > 0.8).long()
    model.train()
    out = model(eeg, transition=trans, return_aux=True)
    assert out["stage"].shape == (b * t, 5), out["stage"].shape
    loss = T.mass_loss(
        out["stage"], label.reshape(-1), out["transition"], out["transition_label"],
        torch.tensor([0.8, 0.2], device=device), cfg.train,
    )
    loss.backward()
    model.eval()
    with torch.no_grad():
        out_e = model(eeg, transition=trans, return_aux=True)
        preds = out_e["stage"].argmax(dim=-1)
    pol = out["policy"]
    logger.info(
        "Smoke OK  pi=%s B=%.2f mix=%.2f align_inter=%s loss=%.4f pred=%s vis_e=%s vis_p=%s obs=%.3f",
        cfg.pi, cfg.budget, cfg.mix_random, getattr(cfg, "align_inter", True),
        float(loss.detach()), tuple(preds.shape),
        tuple(pol.epoch_visible.shape), tuple(pol.patch_visible.shape), pol.overall_ratio,
    )


def run_cv(cfg: RunConfig, logger: logging.Logger, out_dir: Path) -> dict:
    ds = D.assert_dataset_ready(cfg.dataset)
    device = T.get_device(cfg.device)
    batch_size = cfg.batch_size or ds.batch_size
    fold_metrics: List[dict] = []
    n_mat = len(D.list_mat_files(ds))
    logger.info("dataset=%s files=%s batch_size=%s data_key=%s split=%s",
                ds.name, n_mat, batch_size, ds.data_key, ds.split)
    if cfg.dataset == "shhs" and n_mat < 329:
        logger.warning(
            "SHHS expected 329 shhs1-*.mat (AttnSleep subset), found %s. "
            "Finish: python preprocess/preprocess.py --dataset shhs",
            n_mat,
        )

    for fold_idx, train_files, test_files, test_subjects in D.iter_splits(
        ds, seed=cfg.seed, fold=cfg.fold, max_folds=cfg.max_folds
    ):
        logger.info("==== %s fold %s n_train=%s n_test=%s test_subjects=%s ====",
                    cfg.dataset, fold_idx, len(train_files), len(test_files), test_subjects)
        train_iter, test_iter, stats = D.load_fold_tensors(
            train_files, test_files, ds.data_key, batch_size, cfg.seed, str(device)
        )
        logger.info(
            "train %s test %s mean=%.4f std=%.4f",
            stats["train_shape"], stats["test_shape"], stats["mean"], stats["std"],
        )
        model = build_model(cfg, torch.tensor(stats["mean"]), torch.tensor(stats["std"]), device)
        metrics = T.train_one_fold(
            model, train_iter, test_iter, cfg, fold_idx,
            stats["trans_weights"], out_dir, logger,
        )
        metrics["test_subjects"] = list(map(str, test_subjects))
        fold_metrics.append(metrics)
        del model, train_iter, test_iter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = _summarize(fold_metrics)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"config": _cfg_dict(cfg), "folds": fold_metrics, "summary": summary}, f, indent=2)
    logger.info("==== %s summary ====\n%s", cfg.run_tag, json.dumps(summary, indent=2))
    return summary


def _summarize(fold_metrics: List[dict]) -> dict:
    keys = ["acc", "mf1", "kappa", "mgm", "obs_ratio"]
    out = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if k in m]
        mu, sd = mean_std(vals)
        out[k] = {"mean": mu, "std": sd}
    f1_stack = [m["f1_per_class"] for m in fold_metrics if "f1_per_class" in m]
    if f1_stack:
        import numpy as np
        arr = np.asarray(f1_stack, dtype=float)
        out["f1_per_class_mean"] = arr.mean(axis=0).tolist()
        out["stage_names"] = ["W", "N1", "N2", "N3", "REM"]
    return out


def _cfg_dict(cfg: RunConfig) -> dict:
    return {
        "dataset": cfg.dataset,
        "pi": cfg.pi,
        "budget": cfg.budget,
        "mix_random": cfg.mix_random,
        "align_inter": getattr(cfg, "align_inter", True),
        "epochs": cfg.train.total_epochs,
        "seed": cfg.seed,
        "run_tag": cfg.run_tag,
    }


def main():
    args = parse_args()
    T.set_seed(args.seed)
    cfg = make_run_config(args)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.smoke:
        logger = build_logger(PACKAGE_DIR / "results" / "_smoke.log")
        logger.info("Repo root: %s", ROOT)
        logger.info("Smoke config: %s", json.dumps(_cfg_dict(cfg), indent=2))
        for pi in PI_CHOICES:
            cfg.pi = pi
            cfg.budget = 0.40
            cfg.mix_random = 0.0 if pi in ("random", "band") else 0.25
            cfg.align_inter = pi != "band"
            smoke_test(cfg, logger)
        logger.info("All smoke tests passed.")
        return

    out_dir = PACKAGE_DIR / "results" / f"{cfg.run_tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(out_dir / "train.log")
    logger.info("Repo root: %s", ROOT)
    logger.info("Run config: %s", json.dumps(_cfg_dict(cfg), indent=2))
    logger.info("Output: %s", out_dir)
    run_cv(cfg, logger, out_dir)


if __name__ == "__main__":
    if __package__ is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from SAS.run import main as _main
        _main()
    else:
        main()
