#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LFCC extractor (FIXED VERSION)
- Single ROOT, balanced sampling
- Outputs 4 groups: Train/Test x genuine/spoof
- NO CMVN / NO global normalization (safe for CNN/LCNN)
- LFCC + delta + delta-delta + log-energy
- Saves PKL + INDEX CSV + SUMMARY + ZIP

This version fixes the 0.5-accuracy collapse issue.
"""

# ===================== IMPORTS =====================
import os
import pickle
import logging
import random
from pathlib import Path
from typing import List, Dict, Optional
import csv
import zipfile

import h5py
import numpy as np
import soundfile as sf
from scipy.fft import dct

from spafe.utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
from spafe.utils.exceptions import ParameterError, ErrorMsgs
from spafe.fbanks.linear_fbanks import linear_filter_banks

# ================= USER CONFIG =====================
ROOT = r"D:\NECTEC\Thai Spoof\Thaispoof_eq_split_final"
OUT_BALANCED = r"D:\NECTEC\Thai Spoof\LFCC_Features_Balanced_FIXED"

N_GENUINE_TRAIN = 3000
N_GENUINE_TEST  = 1500
N_SPOOF_TRAIN   = 3000
N_SPOOF_TEST    = 1500

RANDOM_SEED = 1337
USE_CACHE = True

CLASS_ALIASES = {
    "genuine": ["genuine", "real"],
    "spoof": [
        "spoof", "mms_spoof", "mms_spoof_root",
        "tts_vaja_spoof", "tts", "vaja",
        "f0", "pitchshift", "speedchange",
    ],
}

# ===== LFCC PARAMS (SAFE VERSION) =====
LFCC_PARAMS = {
    "num_ceps": 20,
    "pre_emph": 1,
    "pre_emph_coeff": 0.97,
    "win_len": 0.030,
    "win_hop": 0.015,
    "win_type": "hamming",
    "nfilts": 70,
    "nfft": 1024,
    "low_freq": 0,
    "high_freq": None,
    "scale": "constant",
    "dct_type": 2,
    "use_energy": 1,
    "lifter": 0,
    "normalize": 0,          # <<< IMPORTANT: NO CMVN
    "add_deltas": True,
    "add_delta_deltas": True,
}

# ================= LOGGING =========================
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)

rng = random.Random(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ================= HELPERS =========================

def _delta(feat: np.ndarray, N: int = 2) -> np.ndarray:
    denom = 2 * sum(i * i for i in range(1, N + 1))
    padded = np.pad(feat, ((N, N), (0, 0)), mode="edge")
    out = np.zeros_like(feat)
    for t in range(feat.shape[0]):
        for n in range(1, N + 1):
            out[t] += n * (padded[t + N + n] - padded[t + N - n])
    return out / denom


def lfcc(sig: np.ndarray, fs: int = 16000, **P) -> np.ndarray:
    high_freq = P["high_freq"] or fs / 2
    low_freq = P["low_freq"] or 0

    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > fs / 2:
        raise ParameterError(ErrorMsgs["high_freq"])

    if P["pre_emph"]:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=P["pre_emph_coeff"])

    frames, frame_len = framing(sig, fs, P["win_len"], P["win_hop"])
    windows = windowing(frames, frame_len, P["win_type"])

    spectrum = np.fft.rfft(windows, P["nfft"])
    power_spec = np.abs(spectrum) ** 2

    eps = 1e-12
    log_energy = np.log(power_spec.sum(axis=1) + eps)

    fbanks, _ = linear_filter_banks(
        nfilts=P["nfilts"], nfft=P["nfft"], fs=fs,
        low_freq=low_freq, high_freq=high_freq, scale=P["scale"]
    )
    if isinstance(fbanks, tuple):
        fbanks = fbanks[0]

    feats = np.dot(power_spec, fbanks.T)
    feats = zero_handling(feats)
    log_feats = np.log10(feats + eps)

    base = dct(log_feats, type=P["dct_type"], norm="ortho", axis=1)[:, :P["num_ceps"]]

    if P["use_energy"]:
        base[:, 0] = log_energy

    out = base
    if P["add_deltas"]:
        d1 = _delta(base)
        out = np.concatenate([out, d1], axis=1)
        if P["add_delta_deltas"]:
            d2 = _delta(d1)
            out = np.concatenate([out, d2], axis=1)

    return out.astype(np.float32)


def extract_lfcc_single(fp: str) -> np.ndarray:
    sig, fs = sf.read(fp)
    if sig.ndim == 2:
        sig = sig.mean(axis=1)
    return lfcc(sig, fs, **LFCC_PARAMS)


class H5Cache:
    def __init__(self, path: Path):
        self.path = str(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, key: str):
        if not USE_CACHE or not os.path.exists(self.path):
            return None
        with h5py.File(self.path, "r") as h5:
            return h5[key][()] if key in h5 else None

    def put(self, key: str, arr: np.ndarray):
        if not USE_CACHE:
            return
        with h5py.File(self.path, "a") as h5:
            if key in h5:
                del h5[key]
            h5.create_dataset(key, data=arr, compression="gzip")


# ================= DATA UTILS ======================

def _norm_token(s: str) -> str:
    return s.replace("-", "_").replace(" ", "_").lower()


def _which_class(name: str) -> Optional[str]:
    t = _norm_token(name)
    for canon, aliases in CLASS_ALIASES.items():
        if t == _norm_token(canon) or any(t == _norm_token(a) for a in aliases):
            return canon
    return None


def collect_all_files(root: Path) -> Dict[str, List[str]]:
    pools = {"genuine": [], "spoof": []}
    for fp in root.rglob("*.wav"):
        for seg in fp.parts:
            cls = _which_class(seg)
            if cls:
                pools[cls].append(str(fp))
                break
    return pools


def pick_files(files: List[str], k: int) -> List[str]:
    files = list(files)
    if len(files) <= k:
        rng.shuffle(files)
        return files
    return rng.sample(files, k)


def extract_bucket(files: List[str], cache: H5Cache) -> List[np.ndarray]:
    out = []
    for i, fp in enumerate(files):
        feat = cache.get(fp)
        if feat is None:
            feat = extract_lfcc_single(fp)
            cache.put(fp, feat)
        out.append(feat)
        if (i + 1) % 50 == 0:
            logging.info("  processed %d/%d", i + 1, len(files))
    return out


# ================= SAVE UTILS ======================

def save_pkl(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info("saved %s (n=%d)", path.name, len(obj))


def save_index(files: List[str], feats: List[np.ndarray], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "T", "F"])
        for fp, arr in zip(files, feats):
            w.writerow([fp, arr.shape[0], arr.shape[1]])


def sanity_check(name: str, feats: List[np.ndarray]):
    allf = np.vstack(feats)
    logging.info(
        "[%s] mean=%.5f std=%.5f min=%.5f max=%.5f",
        name, allf.mean(), allf.std(), allf.min(), allf.max()
    )


def zip_outputs(out_root: Path):
    zip_path = out_root / "lfcc_outputs_all.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in out_root.rglob("*.pkl"):
            z.write(fp, fp.relative_to(out_root))
        for fp in out_root.rglob("INDEX_*.csv"):
            z.write(fp, fp.relative_to(out_root))
        summary = out_root / "SUMMARY_counts.csv"
        if summary.exists():
            z.write(summary, summary.name)
    logging.info("ZIP created: %s", zip_path)


# ================= MAIN ============================

def main():
    logging.info("ROOT = %s", ROOT)
    logging.info("OUT  = %s", OUT_BALANCED)

    out_root = Path(OUT_BALANCED)
    out_root.mkdir(parents=True, exist_ok=True)

    cache = H5Cache(out_root / "cache" / "lfcc_cache.h5")

    pools = collect_all_files(Path(ROOT))
    logging.info("Found genuine=%d spoof=%d", len(pools["genuine"]), len(pools["spoof"]))

    train_g = pick_files(pools["genuine"], N_GENUINE_TRAIN)
    test_g  = pick_files([p for p in pools["genuine"] if p not in train_g], N_GENUINE_TEST)

    train_s = pick_files(pools["spoof"], N_SPOOF_TRAIN)
    test_s  = pick_files([p for p in pools["spoof"] if p not in train_s], N_SPOOF_TEST)

    groups = {
        "Train_genuine": train_g,
        "Test_genuine":  test_g,
        "Train_spoof":   train_s,
        "Test_spoof":    test_s,
    }

    counts = {}

    for name, files in groups.items():
        counts[name] = len(files)
        logging.info("Extracting %s (%d files)", name, len(files))
        feats = extract_bucket(files, cache)
        sanity_check(name, feats)

        save_pkl(feats, out_root / f"LFCC_{name}_{len(files)}.pkl")
        save_index(files, feats, out_root / f"INDEX_{name}_{len(files)}.csv")

    # summary
    with open(out_root / "SUMMARY_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group", "Count"])
        for k, v in counts.items():
            w.writerow([k, v])
        w.writerow(["ALL", sum(counts.values())])

    zip_outputs(out_root)
    logging.info("DONE")


if __name__ == "__main__":
    main()
