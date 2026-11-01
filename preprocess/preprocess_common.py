"""Shared preprocessing utilities for MASS sleep staging datasets."""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy import signal

TARGET_SFREQ = 100.0
EPOCH_SEC_SIZE = 30
SAMPLES_PER_EPOCH = int(TARGET_SFREQ * EPOCH_SEC_SIZE)  # 3000
N_FFT = 256
WIN_LENGTH = 100
HOP_LENGTH = 100
PSD_DIM = 128

# Standard 5-class labels used by MASS training code.
W, N1, N2, N3, REM, UNKNOWN = 0, 1, 2, 3, 4, 5

PHYSIONET_ANN2LABEL = {
    "Sleep stage W": W,
    "Sleep stage 1": N1,
    "Sleep stage 2": N2,
    "Sleep stage 3": N3,
    "Sleep stage 4": N3,
    "Sleep stage R": REM,
    "Sleep stage ?": UNKNOWN,
    "Movement time": UNKNOWN,
}

DREAMS_AASM2LABEL = {
    5: W,
    4: REM,
    3: N1,
    2: N2,
    1: N3,
}

# Compumedics Profusion (NSRR SHHS): 0=W, 1=N1, 2=N2, 3=N3, 4=N4, 5=REM.
SHHS_PROFUSION2LABEL = {
    0: W,
    1: N1,
    2: N2,
    3: N3,
    4: N3,
    5: REM,
}

HAMMING = np.hamming(WIN_LENGTH).astype(np.float32)


def find_channel(raw, candidates: Sequence[str]) -> str:
    """Return the first channel name that matches one of the candidates."""
    ch_names = raw.ch_names
    lowered = {name.lower(): name for name in ch_names}
    for candidate in candidates:
        key = candidate.lower()
        if key in lowered:
            return lowered[key]
    for candidate in candidates:
        key = candidate.lower().replace(" ", "")
        for name in ch_names:
            if key in name.lower().replace(" ", ""):
                return name
    raise ValueError(
        f"Cannot find channel among {list(candidates)}. Available channels: {ch_names}"
    )


def resample_signal(values: np.ndarray, src_sfreq: float, dst_sfreq: float = TARGET_SFREQ) -> np.ndarray:
    if math.isclose(src_sfreq, dst_sfreq):
        return values.astype(np.float32)
    gcd = math.gcd(int(round(src_sfreq)), int(round(dst_sfreq)))
    up = int(round(dst_sfreq / gcd))
    down = int(round(src_sfreq / gcd))
    return signal.resample_poly(values, up, down).astype(np.float32)


def epoch_to_psd100(epoch_signal: np.ndarray) -> np.ndarray:
    """Convert one 30-second epoch (3000 samples @ 100 Hz) to [30, 128] PSD features."""
    return epochs_to_psd100(epoch_signal.reshape(1, -1))[0]


def epochs_to_psd100(epochs: np.ndarray) -> np.ndarray:
    """Convert [N, 3000] epoch signals to [N, 30, 128] PSD features."""
    if epochs.ndim != 2:
        raise ValueError(f"Expected 2D epoch array, got shape {epochs.shape}")
    if epochs.shape[1] != SAMPLES_PER_EPOCH:
        raise ValueError(f"Expected {SAMPLES_PER_EPOCH} samples per epoch, got {epochs.shape[1]}")

    segments = epochs.reshape(len(epochs), 30, WIN_LENGTH) * HAMMING[None, None, :]
    spectrum = np.fft.rfft(segments, n=N_FFT, axis=-1)
    magnitude = np.abs(spectrum)
    psd_db = 20.0 * np.log10(magnitude + 1e-8)
    return psd_db[:, :, 1: 1 + PSD_DIM].astype(np.float32)


def signals_to_psd100(epochs: np.ndarray) -> np.ndarray:
    """Backward-compatible alias."""
    return epochs_to_psd100(epochs)


def trim_wake_edges(labels: np.ndarray, epochs: np.ndarray, edge_minutes: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """Keep 30 minutes of wake before first sleep epoch and after last sleep epoch."""
    non_wake = np.where(labels != W)[0]
    if len(non_wake) == 0:
        return epochs, labels

    edge_epochs = edge_minutes * 2  # 30 minutes = 60 thirty-second epochs
    start_idx = max(0, int(non_wake[0]) - edge_epochs)
    end_idx = min(len(labels) - 1, int(non_wake[-1]) + edge_epochs)
    select_idx = np.arange(start_idx, end_idx + 1)
    return epochs[select_idx], labels[select_idx]


def parse_physionet_hypnogram(ann_path: str, sampling_rate: float) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Parse PhysioNet hypnogram EDF and return labels plus valid / invalid sample indices."""
    import dhedfreader

    remove_idx: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    label_idx: List[np.ndarray] = []

    with open(ann_path, 'r', errors='ignore') as handle:
        reader_ann = dhedfreader.BaseEDFReader(handle)
        reader_ann.read_header()
        _, _, ann = zip(*reader_ann.records())

    for a in ann[0]:
        onset_sec, duration_sec, ann_char = a
        ann_str = "".join(ann_char).strip()
        label_key = ann_str[2:-1] if ann_str.startswith("[") else ann_str
        if label_key not in PHYSIONET_ANN2LABEL:
            raise KeyError(f"Unknown annotation label: {label_key!r} in {ann_path}")
        label = PHYSIONET_ANN2LABEL[label_key]
        if duration_sec % EPOCH_SEC_SIZE != 0:
            raise ValueError(
                f"Annotation duration is not a multiple of 30 s in {ann_path}: {duration_sec}"
            )

        duration_epochs = int(duration_sec / EPOCH_SEC_SIZE)
        idx = int(onset_sec * sampling_rate) + np.arange(int(duration_sec * sampling_rate), dtype=np.int64)
        if label != UNKNOWN:
            labels.append(np.full(duration_epochs, label, dtype=np.int32))
            label_idx.append(idx)
        else:
            remove_idx.append(idx)

    labels_arr = np.hstack(labels).astype(np.int32) if labels else np.array([], dtype=np.int32)
    return labels_arr, remove_idx, label_idx


def align_physionet_recording(
    signal_values: np.ndarray,
    sampling_rate: float,
    ann_path: str,
    trim_wake: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract labeled 30-second epochs from one PhysioNet PSG recording."""
    labels, remove_idx, label_idx = parse_physionet_hypnogram(ann_path, sampling_rate)

    select_idx = np.arange(len(signal_values))
    if remove_idx:
        remove_idx_arr = np.hstack(remove_idx)
        select_idx = np.setdiff1d(select_idx, remove_idx_arr)

    if label_idx:
        label_idx_arr = np.hstack(label_idx)
        select_idx = np.intersect1d(select_idx, label_idx_arr)
        if len(label_idx_arr) > len(select_idx):
            extra_idx = np.setdiff1d(label_idx_arr, select_idx)
            if np.all(extra_idx > select_idx[-1]):
                n_label_trims = int(math.ceil(len(extra_idx) / (EPOCH_SEC_SIZE * sampling_rate)))
                if n_label_trims:
                    labels = labels[:-n_label_trims]

    selected_signal = signal_values[select_idx]
    epoch_size = int(EPOCH_SEC_SIZE * sampling_rate)
    if len(selected_signal) % epoch_size != 0:
        raise ValueError(
            f"Selected signal length {len(selected_signal)} is not divisible by epoch size {epoch_size}"
        )

    n_epochs = len(selected_signal) // epoch_size
    epochs = np.asarray(np.split(selected_signal, n_epochs), dtype=np.float32)
    if len(epochs) != len(labels):
        min_len = min(len(epochs), len(labels))
        epochs = epochs[:min_len]
        labels = labels[:min_len]

    if trim_wake:
        epochs, labels = trim_wake_edges(labels, epochs)

    return epochs, labels.astype(np.int32)


def read_dreams_hypnogram(hyp_path: str) -> np.ndarray:
    """Read DREAMS AASM hypnogram (one label every 5 seconds)."""
    with open(hyp_path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        raise ValueError(f"Empty hypnogram file: {hyp_path}")

    if lines[0].startswith("[Hypnogram"):
        lines = lines[1:]

    stage_values = []
    for line in lines:
        try:
            stage_values.append(int(line))
        except ValueError as exc:
            raise ValueError(f"Invalid hypnogram value {line!r} in {hyp_path}") from exc
    return np.asarray(stage_values, dtype=np.int32)


def dreams_hypnogram_to_epoch_labels(stage_values: np.ndarray) -> np.ndarray:
    """Convert 5-second DREAMS labels to 30-second epoch labels via majority vote."""
    if len(stage_values) % 6 != 0:
        stage_values = stage_values[: len(stage_values) // 6 * 6]
    grouped = stage_values.reshape(-1, 6)
    epoch_labels = []
    for group in grouped:
        mapped = [DREAMS_AASM2LABEL[value] for value in group if value in DREAMS_AASM2LABEL]
        mapped = [label for label in mapped if label != UNKNOWN]
        if not mapped:
            epoch_labels.append(UNKNOWN)
            continue
        counts = np.bincount(mapped, minlength=UNKNOWN + 1)
        epoch_labels.append(int(np.argmax(counts[:UNKNOWN])))
    return np.asarray(epoch_labels, dtype=np.int32)


def align_dreams_recording(edf_path: str, hyp_path: str, channel_candidates: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load one DREAMS subject and return aligned epoch signals and labels."""
    from mne.io import read_raw_edf

    raw = read_raw_edf(edf_path, preload=True, verbose=False)
    channel = find_channel(raw, channel_candidates)
    signal_values = raw.get_data(picks=[channel])[0].astype(np.float32)
    signal_values = resample_signal(signal_values, raw.info["sfreq"], TARGET_SFREQ)

    stage_values = read_dreams_hypnogram(hyp_path)
    labels = dreams_hypnogram_to_epoch_labels(stage_values)

    n_epochs = min(len(labels), len(signal_values) // SAMPLES_PER_EPOCH)
    if n_epochs == 0:
        raise ValueError(f"No valid epochs for {edf_path}")

    usable_samples = n_epochs * SAMPLES_PER_EPOCH
    signal_values = signal_values[:usable_samples]
    labels = labels[:n_epochs]
    epochs = signal_values.reshape(n_epochs, SAMPLES_PER_EPOCH)

    valid_mask = labels != UNKNOWN
    epochs = epochs[valid_mask]
    labels = labels[valid_mask]
    if len(labels) == 0:
        raise ValueError(f"No labeled epochs after removing UNKNOWN stages in {edf_path}")

    return epochs.astype(np.float32), labels.astype(np.int32)


def is_edf20_recording(filename: str) -> bool:
    """Return True if filename belongs to Sleep-EDF-20 (subjects 00-19)."""
    match = re.match(r"SC4(\d{2})", filename)
    return bool(match and 0 <= int(match.group(1)) <= 19)


def is_edf78_recording(filename: str) -> bool:
    """Return True if filename belongs to Sleep-EDF-78 (all SC4 cassette recordings)."""
    return bool(re.match(r"SC4\d", filename))


def parse_shhs_profusion_hypnogram(xml_path: str) -> np.ndarray:
    """Read 30-second stage labels from a Compumedics Profusion XML file."""
    root = ET.parse(xml_path).getroot()
    raw_stages: List[int] = []
    for node in root.iter("SleepStage"):
        if node.text is None or not node.text.strip():
            continue
        raw_stages.append(int(node.text.strip()))
    if not raw_stages:
        raise ValueError(f"No <SleepStage> entries in {xml_path}")

    labels = np.empty(len(raw_stages), dtype=np.int32)
    for i, value in enumerate(raw_stages):
        labels[i] = SHHS_PROFUSION2LABEL.get(value, UNKNOWN)
    return labels


def match_channel_name(ch_names: Sequence[str], candidates: Sequence[str]) -> str:
    """Pick a channel by exact name first, then by substring (spaces ignored)."""
    lowered = {name.lower(): name for name in ch_names}
    for candidate in candidates:
        key = candidate.lower()
        if key in lowered:
            return lowered[key]
    for candidate in candidates:
        key = candidate.lower().replace(" ", "")
        for name in ch_names:
            if key == name.lower().replace(" ", ""):
                return name
    for candidate in candidates:
        key = candidate.lower().replace(" ", "")
        for name in ch_names:
            if key in name.lower().replace(" ", ""):
                return name
    raise ValueError(
        f"Cannot find channel among {list(candidates)}. Available channels: {list(ch_names)}"
    )


def read_edf_channel(
    edf_path: str, candidates: Sequence[str]
) -> Tuple[np.ndarray, float, str]:
    """Read one EEG channel from an EDF/EDF+ file without loading the whole PSG.

    SHHS recordings are ~40 MB with many non-EEG traces. Parsing the EDF header
    and seeking per-record keeps memory and runtime reasonable. Falls back to
    MNE if the header cannot be parsed.
    """
    try:
        return _read_edf_channel_raw(edf_path, candidates)
    except Exception as exc:
        try:
            from mne.io import read_raw_edf
        except Exception as mne_exc:
            raise RuntimeError(
                f"EDF parse failed for {edf_path}: {exc}; MNE fallback unavailable: {mne_exc}"
            ) from exc
        raw = read_raw_edf(edf_path, preload=False, verbose=False)
        channel = find_channel(raw, candidates)
        values = raw.get_data(picks=[channel])[0].astype(np.float32)
        return values, float(raw.info["sfreq"]), channel


def _read_edf_channel_raw(
    edf_path: str, candidates: Sequence[str]
) -> Tuple[np.ndarray, float, str]:
    """Read one channel using the official EDF field-major header layout."""

    def _ns_fields(handle, nchan: int, width: int) -> List[str]:
        block = handle.read(nchan * width)
        if len(block) != nchan * width:
            raise ValueError(f"Truncated EDF channel header: {edf_path}")
        return [block[i * width:(i + 1) * width].decode("ascii", errors="ignore").strip() for i in range(nchan)]

    with open(edf_path, "rb") as handle:
        header = handle.read(256)
        if len(header) != 256:
            raise ValueError(f"Truncated EDF header: {edf_path}")
        n_records = int(header[236:244].decode("ascii").strip())
        record_sec = float(header[244:252].decode("ascii").strip())
        nchan = int(header[252:256].decode("ascii").strip())
        if n_records <= 0 or record_sec <= 0 or nchan <= 0:
            raise ValueError(f"Invalid EDF header in {edf_path}")

        labels = _ns_fields(handle, nchan, 16)
        _ns_fields(handle, nchan, 80)  # transducer
        _ns_fields(handle, nchan, 8)  # physical dimension
        phys_min = [float(v) for v in _ns_fields(handle, nchan, 8)]
        phys_max = [float(v) for v in _ns_fields(handle, nchan, 8)]
        dig_min = [int(float(v)) for v in _ns_fields(handle, nchan, 8)]
        dig_max = [int(float(v)) for v in _ns_fields(handle, nchan, 8)]
        _ns_fields(handle, nchan, 80)  # prefiltering
        n_samples = [int(v) for v in _ns_fields(handle, nchan, 8)]
        reserved = handle.read(nchan * 32)
        if len(reserved) != nchan * 32:
            raise ValueError(f"Truncated EDF reserved header: {edf_path}")

        channel = match_channel_name(labels, candidates)
        idx = labels.index(channel)
        if n_samples[idx] <= 0:
            raise ValueError(f"Channel {channel} has 0 samples/record in {edf_path}")

        bytes_per_record = sum(count * 2 for count in n_samples)
        ch_offset = sum(n_samples[i] * 2 for i in range(idx))
        ch_nbytes = n_samples[idx] * 2
        rec_start = 256 * (1 + nchan)
        digital = np.empty(n_records * n_samples[idx], dtype=np.int16)
        for rec in range(n_records):
            handle.seek(rec_start + rec * bytes_per_record + ch_offset)
            chunk = handle.read(ch_nbytes)
            if len(chunk) != ch_nbytes:
                raise ValueError(f"Truncated EDF data record {rec} in {edf_path}")
            digital[rec * n_samples[idx] : (rec + 1) * n_samples[idx]] = np.frombuffer(
                chunk, dtype="<i2"
            )

    scale_den = float(dig_max[idx] - dig_min[idx])
    if scale_den == 0:
        raise ValueError(f"Digital min==max for channel {channel} in {edf_path}")
    scale = (phys_max[idx] - phys_min[idx]) / scale_den
    physical = (digital.astype(np.float32) - float(dig_min[idx])) * np.float32(scale) + np.float32(
        phys_min[idx]
    )
    sfreq = float(n_samples[idx]) / record_sec
    return physical, sfreq, channel


def align_shhs_recording(
    signal_values: np.ndarray,
    sampling_rate: float,
    xml_path: str,
    trim_wake: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pair C4-A1 samples with Profusion 30-second stages; drop unknown labels."""
    labels = parse_shhs_profusion_hypnogram(xml_path)
    epoch_size = int(round(EPOCH_SEC_SIZE * sampling_rate))
    n_sig = len(signal_values) // epoch_size
    n_epochs = min(n_sig, len(labels))
    if n_epochs == 0:
        raise ValueError(f"No overlapping epochs for {xml_path}")

    epochs = signal_values[: n_epochs * epoch_size].reshape(n_epochs, epoch_size).astype(np.float32)
    labels = labels[:n_epochs]
    valid = labels != UNKNOWN
    epochs = epochs[valid]
    labels = labels[valid]
    if len(labels) == 0:
        raise ValueError(f"No valid sleep-stage epochs after dropping unknowns in {xml_path}")

    if trim_wake:
        epochs, labels = trim_wake_edges(labels, epochs)
        if len(labels) == 0:
            raise ValueError(f"No epochs left after wake trimming in {xml_path}")
    return epochs, labels.astype(np.int32)


def pair_physionet_files(data_dir: str) -> Iterable[Tuple[str, str]]:
    """Yield (psg_path, hyp_path) pairs from a PhysioNet sleep-cassette directory."""
    import os

    psg_files = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith("-PSG.edf")
    )
    hyp_files = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.endswith("-Hypnogram.edf")
    )
    if len(psg_files) != len(hyp_files):
        raise ValueError(
            f"PSG/Hypnogram count mismatch in {data_dir}: "
            f"{len(psg_files)} PSG vs {len(hyp_files)} hypnogram files"
        )

    for psg_path, hyp_path in zip(psg_files, hyp_files):
        yield psg_path, hyp_path
