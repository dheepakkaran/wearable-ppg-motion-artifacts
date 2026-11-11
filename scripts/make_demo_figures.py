"""Demo figures for README and docs/index.html.

Figure 1 needs the local windows from `make windows` (data/windows/S1.npz, gitignored);
figure 2 uses only the committed results/predictions.csv.gz.
Run: uv run --with matplotlib --with numpy --with scipy --with pandas python scripts/make_demo_figures.py
"""
import sys, pathlib
import numpy as np, pandas as pd
R = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from src import spectral as sp
from src.features import load_subject   # handles the accelerometer unit fix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = R / "docs" / "assets"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#737373", "axes.labelcolor": "#3d3d3d", "xtick.color": "#737373", "ytick.color": "#737373"})
ACT = {0: "transient", 1: "sitting", 2: "stairs", 3: "table soccer", 4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working"}
SUBJ, WIN = "S1", 3543          # a walking window where the arm swing beats the pulse in the plain spectrum

# Figure 1: one 8 s walking window, time signals and spectra
P, A, H, ACTS = load_subject(R / "data" / "windows" / f"{SUBJ}.npz")
ppg, acc, hr = P[WIN], A[WIN].astype(np.float64), float(H[WIN])
assert int(ACTS[WIN]) == 7
Sp, Sc = sp.clean_spectrum(ppg, acc)
Sm, w = sp.motion_spectrum(acc)
plain, removed = sp.peak_bpm(Sp), sp.peak_bpm(Sc)
t = np.arange(len(ppg)) / sp.FS; f = sp.FREQS * 60; band = (f >= 30) & (f <= 210)
fig = plt.figure(figsize=(9, 4.4), dpi=150); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])
a1 = fig.add_subplot(gs[0, 0]); a1.plot(t, sp.bandpass(ppg), color="#0a0a0a", lw=1); a1.set_ylabel("wrist PPG\n(band-passed)")
a1.set_title(f"{SUBJ}, walking, one 8 s window", loc="left", fontsize=10)
a2 = fig.add_subplot(gs[1, 0], sharex=a1)
for k, c in zip(range(3), ["#0a0a0a", "#737373", "#bdbdbd"]):
    a2.plot(t, sp.bandpass(acc[:, k]), color=c, lw=1, label="xyz"[k])
a2.set_ylabel("accelerometer (g)"); a2.set_xlabel("seconds"); a2.legend(frameon=False, ncol=3, fontsize=8, loc="lower right", bbox_to_anchor=(1, 0.98))
a3 = fig.add_subplot(gs[:, 1])
a3.plot(f[band], Sp[band], color="#a3a3a3", lw=1.6, label=f"PPG spectrum: peak {plain:.1f} bpm")
a3.plot(f[band], Sm[band], color="#bdbdbd", lw=1, ls="--", label="accelerometer spectrum (motion)")
a3.plot(f[band], Sc[band], color="#0a0a0a", lw=1.6, label=f"after motion removal: peak {removed:.1f} bpm")
a3.axvline(hr, color="#0a0a0a", lw=.8, ls=":"); a3.text(hr + 2, 1.02, f"ECG reference {hr:.1f} bpm", fontsize=8.5, color="#3d3d3d")
a3.set_xlabel("frequency (bpm)"); a3.set_ylabel("normalised magnitude"); a3.set_ylim(0, 1.12)
a3.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
fig.tight_layout(); fig.savefig(OUT / "demo-motion-removal-window.png", facecolor="white"); plt.close(fig)
print(f"window {WIN}: ref {hr:.2f}, plain {plain:.2f}, removed {removed:.2f}, motion weight {w:.2f}")

# Figure 2: whole S1 recording, reference vs estimates
p = pd.read_csv(R / "results" / "predictions.csv.gz"); s = p[p.subject == SUBJ].sort_values("window")
m = s.window * 2 / 60
fig, ax = plt.subplots(figsize=(9, 3.6), dpi=150)
ax.plot(m, s.plain_fft, ".", ms=1.5, color="#c8c8c8", label="plain FFT peak", zorder=1)
ax.plot(m, s.fft_motion_tracked, lw=.8, color="#8a8a8a", label="FFT + motion removal + tracking", zorder=2)
ax.plot(m, s.gbm, lw=.8, color="#3d3d3d", label="gradient boosting (subject left out)", zorder=3)
ax.plot(m, s.hr_ref, lw=1.4, color="#0a0a0a", label="ECG reference", zorder=4)
seg = (s.activity != s.activity.shift()).cumsum(); n_lab = 0
for _, g in s.groupby(seg):
    a = int(g.activity.iloc[0])
    if a == 0 or len(g) < 100: continue
    x0, x1 = g.window.min() * 2 / 60, g.window.max() * 2 / 60
    ax.axvspan(x0, x1, color="#f0f0f0", zorder=0); ax.text((x0 + x1) / 2, 196 - 9 * (n_lab % 2), ACT[a], ha="center", fontsize=7.5, color="#737373"); n_lab += 1
ax.set_ylim(35, 205); ax.set_xlabel("minutes"); ax.set_ylabel("heart rate (bpm)")
mae = {k: (s[k] - s.hr_ref).abs().mean() for k in ["plain_fft", "fft_motion_tracked", "gbm"]}
ax.set_title(f"{SUBJ}, whole recording: MAE plain {mae['plain_fft']:.2f}, tracked {mae['fft_motion_tracked']:.2f}, "
             f"boosting {mae['gbm']:.2f} bpm", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.tight_layout(); fig.savefig(OUT / "demo-s1-heart-rate-trace.png", facecolor="white"); plt.close(fig)
print("S1 MAE", mae)
