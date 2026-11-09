"""FFT spectral heart-rate estimation from wrist PPG with accelerometer motion removal (NumPy/SciPy).

Pipeline per 8 s window (fs = 32 Hz):
  1. band-pass 0.5-4 Hz (Butterworth, zero phase) PPG and each accelerometer axis
  2. Hann-windowed, zero-padded FFT magnitude spectra, each normalised to max 1
  3. motion removal: subtract the strongest accelerometer axis spectrum (scaled by how much the
     wrist actually moves) from the PPG spectrum, so peaks shared with motion are suppressed
  4. tracking: pick the HR peak inside +/- SEARCH_BPM of the previous estimate (causal; no labels)
Baseline = plain FFT peak of the band-passed PPG, no motion removal, no tracking.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 32
NFFT = 1024                     # 32/1024 Hz = 1.875 bpm per bin before peak interpolation
HR_LO, HR_HI = 0.67, 3.5        # 40-210 bpm search band
SOS = butter(4, [0.5, 4.0], btype="band", fs=FS, output="sos")
FREQS = np.fft.rfftfreq(NFFT, 1 / FS)
BAND = (FREQS >= HR_LO) & (FREQS <= HR_HI)
MOTION_G = 0.05                 # band-passed acc std (g) at which motion removal reaches full weight (picked on S1,S2,S6; see BUILD-LOG)
SEARCH_BPM = 15.0


def bandpass(x, axis=-1):
    return sosfiltfilt(SOS, x, axis=axis)


def spectrum(x, axis=-1):
    """Normalised |FFT| of band-passed, Hann-windowed signal(s)."""
    x = bandpass(np.asarray(x, dtype=np.float64), axis=axis)
    x = x - x.mean(axis=axis, keepdims=True)
    w = np.hanning(x.shape[axis])
    shape = [1] * x.ndim; shape[axis] = -1
    S = np.abs(np.fft.rfft(x * w.reshape(shape), n=NFFT, axis=axis))
    m = S.max(axis=axis, keepdims=True)
    return S / np.where(m > 0, m, 1)


def motion_spectrum(acc):
    """acc: (L, 3) window. Returns (max-over-axes normalised acc spectrum, motion weight in [0,1])."""
    Sa = spectrum(acc, axis=0)                      # (F, 3)
    std = bandpass(np.asarray(acc, np.float64), axis=0).std(axis=0)
    weight = float(np.clip(std.max() / MOTION_G, 0.0, 1.0))
    return Sa.max(axis=1), weight


def clean_spectrum(ppg, acc, lam=1.0):
    Sp = spectrum(ppg)
    Sm, w = motion_spectrum(acc)
    Sc = np.clip(Sp - lam * w * Sm, 0.0, None)
    if Sc[BAND].max() <= 0:                          # everything removed -> fall back to raw spectrum
        Sc = Sp
    return Sp, Sc


def peak_bpm(S, lo_bpm=HR_LO * 60, hi_bpm=HR_HI * 60):
    """Peak frequency in bpm inside [lo, hi] with parabolic interpolation."""
    f = FREQS * 60
    idx = np.where((f >= lo_bpm) & (f <= hi_bpm))[0]
    k = idx[np.argmax(S[idx])]
    if 0 < k < len(S) - 1:
        a, b, c = S[k - 1], S[k], S[k + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        return float((k + np.clip(off, -0.5, 0.5)) * FS / NFFT * 60)
    return float(f[k])


def plain_fft_hr(ppg):
    return peak_bpm(spectrum(ppg))


def track(spectra, init_bpm=None, search=SEARCH_BPM):
    """Causal HR tracking over consecutive windows of one subject. spectra: (n, F)."""
    out = np.empty(len(spectra))
    prev = init_bpm if init_bpm is not None else peak_bpm(spectra[0])
    for i, S in enumerate(spectra):
        lo, hi = max(prev - search, HR_LO * 60), min(prev + search, HR_HI * 60)
        est = peak_bpm(S, lo, hi)
        g = peak_bpm(S)
        f = FREQS * 60
        loc = S[(f >= lo) & (f <= hi)].max()
        if loc < 0.3 * S[BAND].max():               # nothing in range: trust the global peak, small step
            est = prev + np.clip(g - prev, -search / 3, search / 3)
        out[i] = est
        prev = est
    return out


def subject_estimates(ppg, acc, lam=1.0, search=SEARCH_BPM):
    """ppg (n, L), acc (n, L, 3) consecutive windows of one subject (vectorised over windows).
    Returns dict of per-window estimates + spectra (raw PPG, cleaned PPG, motion)."""
    Sp = spectrum(ppg)                                            # (n, F)
    Sa = spectrum(acc, axis=1)                                    # (n, F, 3)
    Sm = Sa.max(axis=2)
    std = bandpass(np.asarray(acc, np.float64), axis=1).std(axis=1).max(axis=1)
    w = np.clip(std / MOTION_G, 0.0, 1.0)
    Sc = np.clip(Sp - lam * w[:, None] * Sm, 0, None)
    empty = Sc[:, BAND].max(axis=1) <= 0
    Sc[empty] = Sp[empty]
    plain = np.array([peak_bpm(s) for s in Sp])
    removed = np.array([peak_bpm(s) for s in Sc])
    tracked = track(Sc, init_bpm=plain[0], search=search)
    return dict(plain=plain, removed=removed, tracked=tracked, Sp=Sp, Sc=Sc, Sm=Sm, motion_w=w, acc_std=std)
