from __future__ import annotations

import csv
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

from ThaiSpoof.project.dataset import AudioItem


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureParams:
    feature: str = "lfcc"
    target_sr: int = 16000
    num_ceps: int = 20
    nfilts: int = 70
    nfft: int = 1024
    win_len: float = 0.030
    win_hop: float = 0.015
    pre_emph: float = 0.97
    use_energy: bool = True
    add_deltas: bool = True
    add_delta_deltas: bool = True
    lifter: int = 0


def params_for_feature(feature: str, target_sr: int) -> FeatureParams:
    feature = feature.lower()
    if feature == "lfcc":
        return FeatureParams(feature="lfcc", target_sr=target_sr)
    if feature == "mfcc":
        return FeatureParams(
            feature="mfcc",
            target_sr=target_sr,
            nfilts=40,
            win_len=0.025,
            win_hop=0.010,
            use_energy=False,
            lifter=22,
        )
    raise ValueError("feature must be 'lfcc' or 'mfcc'")


def read_audio_mono(path: Path, target_sr: int):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    sig, sr = sf.read(str(path), always_2d=False)
    sig = np.asarray(sig, dtype=np.float32)
    if sig.ndim == 2:
        sig = sig.mean(axis=1)
    if sr != target_sr:
        divisor = math.gcd(sr, target_sr)
        sig = resample_poly(sig, target_sr // divisor, sr // divisor).astype(np.float32)
        sr = target_sr
    return sig, sr


def _pre_emphasis(sig, coeff: float):
    import numpy as np

    if sig.size == 0:
        return sig
    return np.append(sig[0], sig[1:] - coeff * sig[:-1]).astype(np.float32)


def _frame_signal(sig, sr: int, win_len: float, win_hop: float):
    import numpy as np

    frame_len = max(1, int(round(win_len * sr)))
    hop = max(1, int(round(win_hop * sr)))
    if sig.size < frame_len:
        sig = np.pad(sig, (0, frame_len - sig.size))
    frame_count = 1 + int(math.ceil((sig.size - frame_len) / hop))
    padded_len = frame_len + (frame_count - 1) * hop
    if sig.size < padded_len:
        sig = np.pad(sig, (0, padded_len - sig.size))
    offsets = np.arange(frame_count)[:, None] * hop + np.arange(frame_len)[None, :]
    return sig[offsets]


def _hz_to_mel(hz):
    import numpy as np

    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    import numpy as np

    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _filterbank(kind: str, sr: int, nfft: int, nfilts: int):
    import numpy as np

    low_hz = 0.0
    high_hz = sr / 2.0
    if kind == "mfcc":
        points = _mel_to_hz(np.linspace(_hz_to_mel(low_hz), _hz_to_mel(high_hz), nfilts + 2))
    else:
        points = np.linspace(low_hz, high_hz, nfilts + 2)

    bins = np.floor((nfft + 1) * points / sr).astype(int)
    bins = np.clip(bins, 0, nfft // 2)

    fbanks = np.zeros((nfilts, nfft // 2 + 1), dtype=np.float32)
    for i in range(1, nfilts + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        right = min(right, nfft // 2)
        for j in range(left, center):
            fbanks[i - 1, j] = (j - left) / max(center - left, 1)
        for j in range(center, right):
            fbanks[i - 1, j] = (right - j) / max(right - center, 1)
    return fbanks


def _delta(feat, width: int = 2):
    import numpy as np

    denom = 2 * sum(i * i for i in range(1, width + 1))
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    out = np.zeros_like(feat)
    for t in range(feat.shape[0]):
        for i in range(1, width + 1):
            out[t] += i * (padded[t + width + i] - padded[t + width - i])
    return out / denom


def _lifter(ceps, lifter: int):
    import numpy as np

    if lifter <= 0:
        return ceps
    n = np.arange(ceps.shape[1])
    lift = 1 + (lifter / 2.0) * np.sin(np.pi * n / lifter)
    return ceps * lift


def compute_feature(sig, sr: int, params: FeatureParams):
    import numpy as np
    from scipy.fft import dct

    if params.pre_emph:
        sig = _pre_emphasis(sig, params.pre_emph)

    frames = _frame_signal(sig, sr, params.win_len, params.win_hop)
    windows = frames * np.hamming(frames.shape[1])[None, :]
    spectrum = np.fft.rfft(windows, params.nfft)
    power = np.abs(spectrum) ** 2
    log_energy = np.log(power.sum(axis=1) + 1e-12)

    fbanks = _filterbank(params.feature, sr, params.nfft, params.nfilts)
    filtered = np.dot(power, fbanks.T)
    log_filtered = np.log(filtered + 1e-12)
    base = dct(log_filtered, type=2, norm="ortho", axis=1)[:, : params.num_ceps]

    if params.use_energy:
        base[:, 0] = log_energy
    base = _lifter(base, params.lifter)

    features = base
    if params.add_deltas:
        first = _delta(base)
        features = np.concatenate([features, first], axis=1)
        if params.add_delta_deltas:
            second = _delta(first)
            features = np.concatenate([features, second], axis=1)
    return features.astype(np.float32)


def pad_or_repeat(mat, dim_x: int, dim_y: int):
    import numpy as np

    x = np.asarray(mat, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected a 2D feature matrix, got shape {x.shape}")
    if x.shape[0] == 0:
        x = np.zeros((1, x.shape[1]), dtype=np.float32)
    if x.shape[0] < dim_x:
        repeat_count = int(math.ceil(dim_x / x.shape[0]))
        x = np.concatenate([x] * repeat_count, axis=0)
    x = x[:dim_x, :]
    if x.shape[1] < dim_y:
        x = np.pad(x, ((0, 0), (0, dim_y - x.shape[1])))
    return x[:, :dim_y].astype(np.float32)


def extract_feature_file(item: AudioItem, params: FeatureParams):
    sig, sr = read_audio_mono(item.path, params.target_sr)
    return compute_feature(sig, sr, params)


def _display_split(split_name: str) -> str:
    first, second = split_name.split("_", 1)
    return f"{first.capitalize()}_{second}"


def save_feature_groups(splits: dict[str, list[AudioItem]], out_dir: Path, feature: str, target_sr: int) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = params_for_feature(feature, target_sr)
    summary_rows = []

    for split_name, items in splits.items():
        display = _display_split(split_name)
        features = []
        index_rows = []
        LOGGER.info("Extracting %s %s files for %s", len(items), feature.upper(), display)
        for idx, item in enumerate(items, start=1):
            arr = extract_feature_file(item, params)
            features.append(arr)
            index_rows.append([str(item.path), item.label, item.attack_type, arr.shape[0], arr.shape[1]])
            if idx % 50 == 0:
                LOGGER.info("  %s: %s/%s files", display, idx, len(items))

        pkl_path = out_dir / f"{feature.upper()}_{display}_{len(features)}.pkl"
        with pkl_path.open("wb") as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

        index_path = out_dir / f"INDEX_{display}_{len(features)}.csv"
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label", "attack_type", "frames", "dims"])
            writer.writerows(index_rows)

        summary_rows.append([split_name, len(features), pkl_path.name])

    with (out_dir / "SUMMARY_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "count", "feature_file"])
        writer.writerows(summary_rows)
