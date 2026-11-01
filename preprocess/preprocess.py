#!/usr/bin/env python3
"""Preprocess raw EDF sleep recordings into MASS-compatible .mat files.

Run from the repository root:

    python preprocess/preprocess.py --dataset dreams --input_dir <edf_dir>
    python preprocess/preprocess.py --dataset edf20 --input_dir <sleep-cassette>
    python preprocess/preprocess.py --dataset edf78 --input_dir <sleep-cassette>
    python preprocess/preprocess.py --dataset shhs --input_dir <shhs1_edf> --ann_dir <shhs1_xml>
    python preprocess/preprocess.py --dataset shhs --input_dir <shhs1_edf> --ann_dir <shhs1_xml> --limit 1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Sequence

import numpy as np
from scipy.io import savemat

_MYTEST_DIR = os.path.dirname(os.path.abspath(__file__))
if _MYTEST_DIR not in sys.path:
    sys.path.insert(0, _MYTEST_DIR)

from preprocess_common import (
    TARGET_SFREQ,
    align_dreams_recording,
    align_physionet_recording,
    align_shhs_recording,
    epochs_to_psd100,
    find_channel,
    is_edf20_recording,
    is_edf78_recording,
    pair_physionet_files,
    read_edf_channel,
    resample_signal,
    signals_to_psd100,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHHS_CHANNEL_CANDIDATES = ("EEG C4-A1", "C4-A1", "EEG")


def preprocess_dreams(
    input_dir: str,
    output_dir: str,
    overwrite: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    channel_candidates = ("EEG Cz-A1", "Cz-A1", "EEG C3-A1", "C3-A1")

    for subject_id in range(1, 21):
        edf_path = os.path.join(input_dir, f"subject{subject_id}.edf")
        hyp_path = os.path.join(input_dir, f"HypnogramAASM_subject{subject_id}.txt")
        output_path = os.path.join(output_dir, f"subject{subject_id}.mat")

        if not os.path.exists(edf_path):
            raise FileNotFoundError(f"Missing DREAMS EDF file: {edf_path}")
        if not os.path.exists(hyp_path):
            raise FileNotFoundError(f"Missing DREAMS hypnogram file: {hyp_path}")
        if os.path.exists(output_path) and not overwrite:
            print(f"[DREAMS] Skip existing file: {output_path}")
            continue

        print(f"[DREAMS] Processing subject {subject_id} ...")
        epochs, labels = align_dreams_recording(edf_path, hyp_path, channel_candidates)
        psd = signals_to_psd100(epochs)
        savemat(
            output_path,
            {
                "PSD": psd,
                "label": labels.reshape(-1, 1),
            },
            do_compression=True,
        )
        print(
            f"[DREAMS] Saved {output_path} | epochs={len(labels)} | PSD shape={psd.shape}"
        )


def preprocess_edf20(
    input_dir: str,
    output_dir: str,
    overwrite: bool = False,
    trim_wake: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    channel_candidates = ("EEG Fpz-Cz", "Fpz-Cz")

    processed = 0
    for psg_path, hyp_path in pair_physionet_files(input_dir):
        recording_name = os.path.basename(psg_path).replace("-PSG.edf", "")
        if not is_edf20_recording(recording_name):
            continue

        output_path = os.path.join(output_dir, f"{recording_name}.mat")
        if os.path.exists(output_path) and not overwrite:
            print(f"[EDF20] Skip existing file: {output_path}")
            continue

        print(f"[EDF20] Processing {recording_name} ...", flush=True)
        from mne.io import read_raw_edf

        raw = read_raw_edf(psg_path, preload=True, verbose=False)
        channel = find_channel(raw, channel_candidates)
        signal_values = raw.get_data(picks=[channel])[0].astype(np.float32)
        src_sfreq = float(raw.info["sfreq"])
        signal_values = resample_signal(signal_values, src_sfreq, TARGET_SFREQ)

        epochs, labels = align_physionet_recording(
            signal_values,
            TARGET_SFREQ,
            hyp_path,
            trim_wake=trim_wake,
        )
        psd = epochs_to_psd100(epochs)
        savemat(
            output_path,
            {
                "PSD100": psd,
                "label": labels.reshape(-1, 1),
            },
            do_compression=True,
        )
        processed += 1
        print(
            f"[EDF20] Saved {output_path} | epochs={len(labels)} | PSD100 shape={psd.shape}"
        )

    if processed == 0:
        raise RuntimeError(
            f"No Sleep-EDF-20 recordings found in {input_dir}. "
            "Expected files like SC4001E0-PSG.edf with subject ids 00-19."
        )


def preprocess_edf78(
    input_dir: str,
    output_dir: str,
    overwrite: bool = False,
    trim_wake: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    channel_candidates = ("EEG Fpz-Cz", "Fpz-Cz")

    processed = 0
    for psg_path, hyp_path in pair_physionet_files(input_dir):
        recording_name = os.path.basename(psg_path).replace("-PSG.edf", "")
        if not is_edf78_recording(recording_name):
            continue

        output_path = os.path.join(output_dir, f"{recording_name}.mat")
        if os.path.exists(output_path) and not overwrite:
            print(f"[EDF78] Skip existing file: {output_path}")
            continue

        print(f"[EDF78] Processing {recording_name} ...", flush=True)
        from mne.io import read_raw_edf

        raw = read_raw_edf(psg_path, preload=True, verbose=False)
        channel = find_channel(raw, channel_candidates)
        signal_values = raw.get_data(picks=[channel])[0].astype(np.float32)
        src_sfreq = float(raw.info["sfreq"])
        signal_values = resample_signal(signal_values, src_sfreq, TARGET_SFREQ)

        epochs, labels = align_physionet_recording(
            signal_values,
            TARGET_SFREQ,
            hyp_path,
            trim_wake=trim_wake,
        )
        psd = epochs_to_psd100(epochs)
        savemat(
            output_path,
            {
                "PSD100": psd,
                "label": labels.reshape(-1, 1),
            },
            do_compression=True,
        )
        processed += 1
        print(
            f"[EDF78] Saved {output_path} | epochs={len(labels)} | PSD100 shape={psd.shape}"
        )

    if processed == 0:
        raise RuntimeError(
            f"No Sleep-EDF-78 recordings found in {input_dir}. "
            "Expected PhysioNet sleep-cassette files like SC4001E0-PSG.edf."
        )


def _first_existing_dir(candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def load_shhs_ids(id_list: str) -> list[str]:
    ids: list[str] = []
    with open(id_list, "r", encoding="utf-8") as handle:
        for line in handle:
            rec_id = line.strip()
            if rec_id and not rec_id.startswith("#"):
                ids.append(rec_id)
    if not ids:
        raise ValueError(f"No recording IDs in {id_list}")
    return ids


def preprocess_shhs(
    input_dir: str,
    output_dir: str,
    ann_dir: str,
    id_list: str,
    overwrite: bool = False,
    trim_wake: bool = True,
    limit: int | None = None,
) -> None:
    """Convert the 329-subject SHHS-1 subset to MASS ``PSD`` .mat files.

    Protocol matches MASS / AttnSleep: C4-A1, 125 Hz -> 100 Hz, N3/N4 merged,
    unknown stages dropped, 30 minutes of wake kept at sleep onset/offset.
    """
    os.makedirs(output_dir, exist_ok=True)
    rec_ids = load_shhs_ids(id_list)
    if limit is not None:
        rec_ids = rec_ids[: max(0, limit)]

    processed = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for rec_id in rec_ids:
        edf_path = os.path.join(input_dir, f"{rec_id}.edf")
        xml_path = os.path.join(ann_dir, f"{rec_id}-profusion.xml")
        output_path = os.path.join(output_dir, f"{rec_id}.mat")

        if not os.path.exists(edf_path):
            failed.append((rec_id, f"missing EDF: {edf_path}"))
            print(f"[SHHS] Missing EDF {edf_path}", flush=True)
            continue
        if not os.path.exists(xml_path):
            failed.append((rec_id, f"missing XML: {xml_path}"))
            print(f"[SHHS] Missing XML {xml_path}", flush=True)
            continue
        if os.path.exists(output_path) and not overwrite:
            skipped += 1
            print(f"[SHHS] Skip existing file: {output_path}", flush=True)
            continue

        print(f"[SHHS] Processing {rec_id} ...", flush=True)
        try:
            signal_values, src_sfreq, channel = read_edf_channel(
                edf_path, SHHS_CHANNEL_CANDIDATES
            )
            signal_values = resample_signal(signal_values, src_sfreq, TARGET_SFREQ)
            epochs, labels = align_shhs_recording(
                signal_values,
                TARGET_SFREQ,
                xml_path,
                trim_wake=trim_wake,
            )
            psd = epochs_to_psd100(epochs)
            savemat(
                output_path,
                {
                    "PSD": psd,
                    "label": labels.reshape(-1, 1),
                },
                do_compression=True,
            )
        except Exception as exc:
            failed.append((rec_id, str(exc)))
            print(f"[SHHS] Failed {rec_id}: {exc}", flush=True)
            continue

        processed += 1
        print(
            f"[SHHS] Saved {output_path} | ch={channel} | "
            f"sfreq {src_sfreq:.1f}->{TARGET_SFREQ:.0f} Hz | "
            f"epochs={len(labels)} | PSD shape={psd.shape}",
            flush=True,
        )

    print(
        f"[SHHS] Done. processed={processed} skipped={skipped} failed={len(failed)} "
        f"listed={len(rec_ids)}"
    )
    if failed:
        print("[SHHS] Failures:")
        for rec_id, reason in failed:
            print(f"  {rec_id}: {reason}")
    if processed == 0 and skipped == 0:
        raise RuntimeError(
            "No SHHS recordings were processed. "
            f"EDF dir={input_dir} XML dir={ann_dir} id_list={id_list}"
        )


def resolve_input_dir(dataset: str, input_dir: str | None) -> str:
    if input_dir:
        return input_dir
    if dataset == "dreams":
        candidates = [
            "dataset/DREAMS/DatabaseSubjects",
            "dataset/DREAMS",
            os.path.join(REPO_ROOT, "dataset/DREAMS/DatabaseSubjects"),
            os.path.join(REPO_ROOT, "dataset/DREAMS"),
        ]
    elif dataset == "edf78":
        candidates = [
            "dataset/EDF78",
            "dataset/EDF78/sleep-cassette",
            os.path.join(REPO_ROOT, "dataset/EDF78"),
            os.path.join(REPO_ROOT, "dataset/EDF78/sleep-cassette"),
        ]
    elif dataset == "shhs":
        candidates = [
            "dataset/SHHS/raw/edfs/shhs1",
            os.path.join(REPO_ROOT, "dataset/SHHS/raw/edfs/shhs1"),
        ]
    else:
        candidates = [
            "dataset/EDF20",
            "dataset/EDF20/sleep-cassette",
            os.path.join(REPO_ROOT, "dataset/EDF20"),
            os.path.join(REPO_ROOT, "dataset/EDF20/sleep-cassette"),
        ]

    found = _first_existing_dir(candidates)
    if found:
        return found
    raise FileNotFoundError(f"Could not infer input directory for {dataset}. Use --input_dir.")


def resolve_ann_dir(ann_dir: str | None) -> str:
    if ann_dir:
        if not os.path.isdir(ann_dir):
            raise FileNotFoundError(f"SHHS annotation directory not found: {ann_dir}")
        return ann_dir
    candidates = [
        "dataset/SHHS/raw/annotations-events-profusion/shhs1",
        os.path.join(REPO_ROOT, "dataset/SHHS/raw/annotations-events-profusion/shhs1"),
    ]
    found = _first_existing_dir(candidates)
    if found:
        return found
    raise FileNotFoundError(
        "Could not infer SHHS Profusion XML directory. Use --ann_dir."
    )


def resolve_id_list(id_list: str | None) -> str:
    if id_list:
        if not os.path.isfile(id_list):
            raise FileNotFoundError(f"SHHS ID list not found: {id_list}")
        return id_list
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "selected_shhs1_files.txt"),
        "dataset/SHHS/selected_shhs1_files.txt",
        os.path.join(REPO_ROOT, "dataset/SHHS/selected_shhs1_files.txt"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find selected_shhs1_files.txt. Use --id_list."
    )


def resolve_output_dir(dataset: str, output_dir: str | None) -> str:
    if output_dir:
        return output_dir
    if dataset == "dreams":
        return os.path.join(REPO_ROOT, "dataset/DREAMS")
    if dataset == "edf78":
        return os.path.join(REPO_ROOT, "dataset/EDF78")
    if dataset == "shhs":
        return os.path.join(REPO_ROOT, "dataset/SHHS")
    return os.path.join(REPO_ROOT, "dataset/EDF20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess DREAMS / Sleep-EDF / SHHS EDF files to .mat"
    )
    parser.add_argument(
        "--dataset",
        choices=["dreams", "edf20", "edf78", "shhs", "all"],
        default="all",
        help="Which dataset to preprocess. `all` is DREAMS + EDF20 + EDF78 (not SHHS).",
    )
    parser.add_argument("--input_dir", type=str, default=None, help="Directory containing raw EDF files")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save .mat files")
    parser.add_argument(
        "--ann_dir",
        type=str,
        default=None,
        help="SHHS Profusion XML directory (default: dataset/SHHS/raw/annotations-events-profusion/shhs1)",
    )
    parser.add_argument(
        "--id_list",
        type=str,
        default=None,
        help="Text file of SHHS-1 IDs (default: preprocess/selected_shhs1_files.txt)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N recordings (useful for a SHHS smoke test)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .mat files")
    parser.add_argument(
        "--no_trim_wake",
        action="store_true",
        help="Disable DeepSleepNet-style wake trimming for Sleep-EDF-20/78 and SHHS",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = (
        ["dreams", "edf20", "edf78"]
        if args.dataset == "all"
        else [args.dataset]
    )

    for dataset in datasets:
        input_dir = resolve_input_dir(dataset, args.input_dir if args.dataset != "all" else None)
        output_dir = resolve_output_dir(dataset, args.output_dir if args.dataset != "all" else None)
        print(f"\n=== Preprocessing {dataset.upper()} ===")
        print(f"Input : {input_dir}")
        print(f"Output: {output_dir}")

        if dataset == "dreams":
            preprocess_dreams(input_dir, output_dir, overwrite=args.overwrite)
        elif dataset == "edf78":
            preprocess_edf78(
                input_dir,
                output_dir,
                overwrite=args.overwrite,
                trim_wake=not args.no_trim_wake,
            )
        elif dataset == "shhs":
            preprocess_shhs(
                input_dir,
                output_dir,
                ann_dir=resolve_ann_dir(args.ann_dir),
                id_list=resolve_id_list(args.id_list),
                overwrite=args.overwrite,
                trim_wake=not args.no_trim_wake,
                limit=args.limit,
            )
        else:
            preprocess_edf20(
                input_dir,
                output_dir,
                overwrite=args.overwrite,
                trim_wake=not args.no_trim_wake,
            )

    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Preprocessing failed: {exc}", file=sys.stderr)
        raise
