"""Export 20 band-passed windows (S1, evenly spaced) for the MATLAB/Octave cross-check, plus Python reference
outputs of the same FFT motion-removal core (spectra, motion weight, HR peaks)."""
import pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import spectral as sp
from src.features import load_subject

ROOT = pathlib.Path(__file__).resolve().parents[1]
ppg, acc, hr, act = load_subject(ROOT / "data" / "windows" / "S1.npz")
idx = np.linspace(0, len(ppg) - 1, 20).astype(int)
p = sp.bandpass(ppg[idx].astype(np.float64))                    # (20, 256)
a = sp.bandpass(acc[idx].astype(np.float64), axis=1)            # (20, 256, 3)
np.savetxt(ROOT / "results" / "matlab_windows.csv", np.hstack([p, a[:, :, 0], a[:, :, 1], a[:, :, 2]]), delimiter=",", fmt="%.10e")
# Python reference of the core on the same band-passed inputs (band-pass already applied -> core only)
def spec(x, axis=-1):
    x = x - x.mean(axis=axis, keepdims=True)
    w = np.hanning(x.shape[axis]); sh = [1] * x.ndim; sh[axis] = -1
    S = np.abs(np.fft.rfft(x * w.reshape(sh), n=sp.NFFT, axis=axis))
    return S / S.max(axis=axis, keepdims=True)
Sp, Sm = spec(p), spec(a, axis=1).max(axis=2)
wgt = np.clip(a.std(axis=1).max(axis=1) / sp.MOTION_G, 0, 1)
Sc = np.clip(Sp - wgt[:, None] * Sm, 0, None)
ref = np.column_stack([idx, [sp.peak_bpm(s) for s in Sp], [sp.peak_bpm(s) for s in Sc], wgt])
np.savetxt(ROOT / "results" / "python_ref_peaks.csv", ref, delimiter=",", header="window,plain_bpm,removed_bpm,motion_weight", comments="", fmt="%.10f")
np.savetxt(ROOT / "data" / "python_ref_clean_spectra.csv", Sc, delimiter=",", fmt="%.12e")
print("exported", len(idx), "windows")
