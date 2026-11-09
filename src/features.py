"""Per-window features for the gradient-boosting model + compact spectra for the CNN (one subject at a time)."""
import numpy as np
from scipy.stats import kurtosis, skew
from . import spectral as sp

CNN_BAND = (sp.FREQS >= 0.5) & (sp.FREQS <= 4.0)      # 113 bins
BIN_EDGES = np.arange(0.7, 3.51, 0.1)                 # 0.1 Hz (6 bpm) bins over 42-210 bpm


def load_subject(path):
    d = np.load(path)
    acc = d["acc"]
    if np.linalg.norm(acc[:50], axis=2).mean() < 0.1:   # files written before the unit fix (extra /64)
        acc = acc * 64.0
    return d["ppg"], acc.astype(np.float32), d["hr"], d["activity"]


def _top_peaks(S, k=3):
    f = sp.FREQS[sp.BAND] * 60
    s = S[:, sp.BAND]
    idx = np.argsort(-s, axis=1)[:, :k]
    return np.take_along_axis(np.broadcast_to(f, s.shape), idx, 1), np.take_along_axis(s, idx, 1)


def _binned(S):
    cols = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        m = (sp.FREQS >= lo) & (sp.FREQS < hi)
        cols.append(S[:, m].max(axis=1))
    return np.stack(cols, 1)


def subject_features(ppg, acc):
    r = sp.subject_estimates(ppg, acc)
    F = {"hr_plain_fft": r["plain"], "hr_motion_removed": r["removed"], "hr_tracked": r["tracked"],
         "motion_weight": r["motion_w"], "acc_bp_std_max": r["acc_std"]}
    for name, S in (("raw", r["Sp"]), ("clean", r["Sc"]), ("acc", r["Sm"])):
        pf, pa = _top_peaks(S)
        for j in range(3):
            F[f"{name}_peak{j+1}_bpm"] = pf[:, j]; F[f"{name}_peak{j+1}_amp"] = pa[:, j]
        p = S[:, sp.BAND] / S[:, sp.BAND].sum(1, keepdims=True)
        F[f"{name}_spec_entropy"] = -(p * np.log(p + 1e-12)).sum(1)
    for name, S in (("raw", r["Sp"]), ("clean", r["Sc"])):
        B = _binned(S)
        for j in range(B.shape[1]):
            F[f"{name}_bin_{int(round(BIN_EDGES[j]*60))}bpm"] = B[:, j]
    bp = sp.bandpass(ppg.astype(np.float64))
    F["ppg_std"] = bp.std(1); F["ppg_skew"] = skew(bp, 1); F["ppg_kurt"] = kurtosis(bp, 1)
    F["ppg_zero_cross_rate"] = (np.diff(np.sign(bp), axis=1) != 0).mean(1)
    mag = np.linalg.norm(acc, axis=2)
    F["acc_mag_mean"] = mag.mean(1); F["acc_mag_std"] = mag.std(1)
    for a, ax in enumerate("xyz"):
        F[f"acc_{ax}_std"] = acc[:, :, a].std(1)
    Sa = sp.spectrum(acc, axis=1)[:, CNN_BAND, :]                         # (n, 113, 3)
    cnn = np.concatenate([r["Sp"][:, CNN_BAND, None], r["Sc"][:, CNN_BAND, None], Sa], axis=2)
    return F, cnn.transpose(0, 2, 1).astype(np.float32)                 # (n, 5, 113)
