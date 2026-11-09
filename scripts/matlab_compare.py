"""Compare Octave/MATLAB outputs with the Python reference -> results/matlab_check.json"""
import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
py = np.loadtxt(ROOT / "results" / "python_ref_peaks.csv", delimiter=",", skiprows=1)
ml = np.loadtxt(ROOT / "results" / "matlab_peaks.csv", delimiter=",", skiprows=1)
Sp = np.loadtxt(ROOT / "data" / "python_ref_clean_spectra.csv", delimiter=",")
Sm = np.loadtxt(ROOT / "data" / "matlab_clean_spectra.csv", delimiter=",")
r = {"windows": int(len(py)), "subject": "S1",
     "max_abs_diff_plain_bpm": float(np.abs(py[:, 1] - ml[:, 1]).max()),
     "max_abs_diff_removed_bpm": float(np.abs(py[:, 2] - ml[:, 2]).max()),
     "max_abs_diff_motion_weight": float(np.abs(py[:, 3] - ml[:, 3]).max()),
     "max_abs_diff_clean_spectrum": float(np.abs(Sp - Sm).max()),
     "scope": "FFT + Hann + normalisation + accelerometer spectrum subtraction + interpolated peak; band-pass applied in Python before export"}
(ROOT / "results" / "matlab_check.json").write_text(json.dumps(r, indent=1))
print(r)
