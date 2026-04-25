#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import logging
import random
from pathlib import Path
import csv
import zipfile

import h5py
import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly

from spafe.utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
from spafe.fbanks.mel_fbanks import mel_filter_banks


# ================= USER CONFIG =================
ROOT = r"D:\NECTEC\Thai Spoof\Thaispoof_eq_split_final"
OUT_BALANCED = r"D:\NECTEC\Thai Spoof\MFCC_Features_Balanced_SINGLEROOT"

N_GENUINE_TRAIN = 3000
N_GENUINE_TEST  = 1500
N_SPOOF_TRAIN   = 3000
N_SPOOF_TEST    = 1500

TARGET_FS = 16000
USE_CACHE = True
RANDOM_SEED = 1337

CLASS_ALIASES = {
    "genuine": ["genuine", "real"],
    "spoof": [
        "spoof", "mms_spoof", "mms_spoof_root",
        "tts_vaja_spoof", "tts", "vaja",
        "f0", "pitchshift", "speedchange",
    ],
}

MFCC_PARAMS = {
    "num_ceps": 20,
    "pre_emph": 1,
    "pre_emph_coeff": 0.97,
    "win_len": 0.025,
    "win_hop": 0.010,
    "win_type": "hamming",
    "nfilts": 40,
    "nfft": 1024,
    "low_freq": 0,
    "high_freq": None,
    "scale": "constant",
    "dct_type": 2,
    "use_energy": 0,
    "lifter": 22,
    "add_deltas": True,
    "add_delta_deltas": True,
}


# ================= Logging / RNG =================
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
)
rng = random.Random(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ================= MFCC helpers =================
def _lifter(x, L=22):
    if L <= 0:
        return x
    n = np.arange(x.shape[1])
    lift = 1 + (L / 2.0) * np.sin(np.pi * n / L)
    return x * lift


def _delta(feat, N=2):
    denom = 2 * sum(i * i for i in range(1, N + 1))
    padded = np.pad(feat, ((N, N), (0, 0)), mode="edge")
    out = np.zeros_like(feat)
    for t in range(feat.shape[0]):
        for n in range(1, N + 1):
            out[t] += n * (padded[t + N + n] - padded[t + N - n])
    return out / denom


def _cmvn_utterance(x, eps=1e-8):
    """Cepstral Mean Variance Normalization per utterance."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def mfcc(sig, fs, **p):
    # resample to 16 kHz
    if fs != TARGET_FS:
        sig = resample_poly(sig, TARGET_FS, fs)
        fs = TARGET_FS

    # pre-emphasis
    if p["pre_emph"]:
        sig = pre_emphasis(sig, p["pre_emph_coeff"])

    # framing
    frames, flen = framing(sig, fs, p["win_len"], p["win_hop"])
    if frames.shape[0] < 3:
        frames = np.pad(frames, ((0, 3 - frames.shape[0]), (0, 0)))

    # windowing
    win = windowing(frames, flen, p["win_type"])

    # FFT -> power spectrum
    spec = np.abs(np.fft.rfft(win, p["nfft"])) ** 2

    # Mel filterbanks
    fb, _ = mel_filter_banks(
        nfilts=p["nfilts"],
        nfft=p["nfft"],
        fs=fs,
        low_freq=p["low_freq"],
        high_freq=p["high_freq"] or fs / 2,
        scale=p["scale"],
    )
    fbanks = zero_handling(np.dot(spec, fb.T))

    # log-mel
    log_fb = np.log(fbanks + 1e-12)

    # DCT -> MFCC
    base = dct(log_fb, type=p["dct_type"], axis=1, norm="ortho")[:, :p["num_ceps"]]

    # liftering
    if p["lifter"]:
        base = _lifter(base, p["lifter"])

    # append deltas
    feats = base
    if p["add_deltas"]:
        d1 = _delta(base)
        feats = np.concatenate([feats, d1], axis=1)
        if p["add_delta_deltas"]:
            d2 = _delta(d1)
            feats = np.concatenate([feats, d2], axis=1)

    # CMVN per utterance
    feats = _cmvn_utterance(feats)

    return feats.astype(np.float32)


# ================= Cache / IO =================
class H5Cache:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def get(self, key):
        if not USE_CACHE or not self.path.exists():
            return None
        with h5py.File(self.path, "r") as h5:
            return h5[key][()] if key in h5 else None

    def put(self, key, arr):
        if not USE_CACHE:
            return
        with h5py.File(self.path, "a") as h5:
            if key in h5:
                del h5[key]
            h5.create_dataset(key, data=arr, compression="gzip")


def extract_single(fp, cache):
    cached = cache.get(fp)
    if cached is not None:
        return cached
    sig, fs = sf.read(fp)
    if sig.ndim == 2:
        sig = sig.mean(axis=1)
    feat = mfcc(sig, fs, **MFCC_PARAMS)
    cache.put(fp, feat)
    return feat


# ================= Dataset helpers =================
def _norm(s):
    return s.replace("-", "_").lower()


def _class_from_path(fp):
    for p in Path(fp).parts:
        for k, v in CLASS_ALIASES.items():
            if _norm(p) == _norm(k) or any(_norm(p) == _norm(a) for a in v):
                return k
    return None


def collect(root):
    pools = {"genuine": [], "spoof": []}
    for fp in root.rglob("*.wav"):
        c = _class_from_path(fp)
        if c:
            pools[c].append(str(fp))
    return pools


def pick(lst, k):
    return rng.sample(lst, min(k, len(lst)))


# ================= Main =================
def main():
    out = Path(OUT_BALANCED)
    out.mkdir(parents=True, exist_ok=True)
    cache = H5Cache(out / "cache" / "mfcc_cache.h5")

    pools = collect(Path(ROOT))
    logging.info("Found genuine=%d spoof=%d", len(pools["genuine"]), len(pools["spoof"]))

    splits = {
        "Train_genuine": pick(pools["genuine"], N_GENUINE_TRAIN),
        "Test_genuine":  pick(pools["genuine"], N_GENUINE_TEST),
        "Train_spoof":   pick(pools["spoof"],   N_SPOOF_TRAIN),
        "Test_spoof":    pick(pools["spoof"],   N_SPOOF_TEST),
    }

    counts = {}
    for name, files in splits.items():
        feats = [extract_single(fp, cache) for fp in files]
        counts[name] = len(feats)

        with open(out / f"MFCC_{name}_{len(feats)}.pkl", "wb") as f:
            pickle.dump(feats, f)

        with open(out / f"INDEX_{name}_{len(feats)}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "T", "F"])
            for fp, a in zip(files, feats):
                w.writerow([fp, a.shape[0], a.shape[1]])
        for i, file in enumerate(files):
            if i % 100 == 0:
                print(f"Processing {i}/{len(files)}")


    with open(out / "SUMMARY_counts.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Group", "Count"])
        for k, v in counts.items():
            w.writerow([k, v])

    with zipfile.ZipFile(out / "mfcc_outputs_all.zip", "w") as z:
        for fp in out.rglob("*"):
            if fp.suffix in {".pkl", ".csv"}:
                z.write(fp, fp.relative_to(out))

    logging.info("DONE")


if __name__ == "__main__":
    main()
