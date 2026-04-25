# ===== MFCC pipeline (spafe-style) =====
from spafe.utils.preprocessing import pre_emphasis, framing, windowing, zero_handling
from spafe.utils.exceptions import ParameterError, ErrorMsgs
from spafe.fbanks.mel_fbanks import mel_filter_banks
from librosa import util
import numpy as np
from scipy.fft import dct

def _lifter(ceps, L=22):
    """Sinusoidal liftering (L>0 -> emphasize middle coeffs)."""
    if L is None or L <= 0:
        return ceps
    n_frames, n_ceps = ceps.shape
    n = np.arange(n_ceps)
    lift = 1 + (L/2.0)*np.sin(np.pi*n/L)
    return ceps * lift

def _cmvn_utterance(x, eps=1e-8):
    """Cepstral Mean Variance Normalization per utterance."""
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)

def _delta(feat, N=2):
    """
    Compute frame-wise deltas using symmetric window of size 2N+1.
    Matches common implementations (e.g., HTK-style).
    """
    if N < 1:
        return np.zeros_like(feat)
    denom = 2 * sum([i*i for i in range(1, N+1)])
    # pad at first/last frame by edge-value replication
    padded = np.pad(feat, ((N, N), (0, 0)), mode="edge")
    out = np.zeros_like(feat)
    for t in range(feat.shape[0]):
        num = np.zeros((feat.shape[1],), dtype=feat.dtype)
        for n in range(1, N+1):
            num += n * (padded[t+N+n] - padded[t+N-n])
        out[t] = num / denom
    return out

def mfcc(sig,
         fs=16000,
         num_ceps=13,
         pre_emph=1,
         pre_emph_coeff=0.97,
         win_len=0.025,
         win_hop=0.010,
         win_type="hamming",
         nfilts=40,
         nfft=1024,
         low_freq=None,
         high_freq=None,
         scale="constant",     # kept for API symmetry; used by mel_filter_banks
         dct_type=2,
         use_energy=0,         # if 1: replace C0 with log-energy
         lifter=22,
         normalize=0,          # if 1: CMVN per utterance
         add_deltas=True,      # append Δ
         add_delta_deltas=True # append ΔΔ
         ):
    """
    Compute MFCCs (optionally with CMVN, lifter, Δ/ΔΔ) in a spafe-like API.

    Returns:
        mfccs               : (T, D) base cepstra or stacked with deltas
        mfccs_base          : (T, num_ceps) base MFCC (before deltas)
        log_energy_per_frame: (T,) log energy (natural log), useful if use_energy=1
    """
    # init freqs
    high_freq = high_freq or fs / 2
    low_freq = low_freq or 0

    # run checks
    if low_freq < 0:
        raise ParameterError(ErrorMsgs["low_freq"])
    if high_freq > (fs / 2):
        raise ParameterError(ErrorMsgs["high_freq"])
    if nfilts < num_ceps:
        raise ParameterError(ErrorMsgs["nfilts"])

    # 0) pre-emphasis
    if pre_emph:
        sig = pre_emphasis(sig=sig, pre_emph_coeff=pre_emph_coeff)

    # 1) framing
    frames, frame_length = framing(sig=sig,
                                   fs=fs,
                                   win_len=win_len,
                                   win_hop=win_hop)

    # 2) windowing
    windows = windowing(frames=frames,
                        frame_len=frame_length,
                        win_type=win_type)

    # 3) STFT -> |.|^2 (power spectrum)
    # rfft size = nfft; returns (T, nfft//2 + 1)
    spectrum = np.fft.rfft(windows, nfft)
    power_spec = (np.abs(spectrum) ** 2)

    # per-frame log energy (natural log) for use_energy option
    # (Energy is sum of power across FFT bins; eps avoids log(0))
    eps = 1e-12
    log_energy_per_frame = np.log(power_spec.sum(axis=1) + eps)

    # 4) Mel filterbanks
    mel_fbanks_mat, _ = mel_filter_banks(
        nfilts=nfilts,
        nfft=nfft,
        fs=fs,
        low_freq=low_freq,
        high_freq=high_freq,
        scale=scale
    )
    if isinstance(mel_fbanks_mat, tuple):
        mel_fbanks_mat = mel_fbanks_mat[0]

    fbanks_out = np.dot(power_spec, mel_fbanks_mat.T)
    fbanks_out = zero_handling(fbanks_out)   # set exact zeros to eps as needed

    # 5) log-compression (use log10 or natural log; keep consistent across pipelines)
    log_fbanks = np.log(fbanks_out + eps)

    # 6) DCT -> MFCCs
    mfccs_base = dct(log_fbanks, type=dct_type, norm='ortho', axis=1)[:, :num_ceps]

    # 7) overwrite C0 with true log energy if requested
    if use_energy:
        mfccs_base[:, 0] = log_energy_per_frame

    # 8) (optional) liftering
    if lifter and lifter > 0:
        mfccs_base = _lifter(mfccs_base, L=lifter)

    # 9) (optional) CMVN per utterance
    if normalize:
        mfccs_base = _cmvn_utterance(mfccs_base)

    # 10) (optional) append deltas
    feats = mfccs_base
    if add_deltas:
        d1 = _delta(mfccs_base, N=2)
        feats = np.concatenate([feats, d1], axis=1)
        if add_delta_deltas:
            d2 = _delta(d1, N=2)
            feats = np.concatenate([feats, d2], axis=1)

    return feats.astype(np.float32), mfccs_base.astype(np.float32), log_energy_per_frame.astype(np.float32)
