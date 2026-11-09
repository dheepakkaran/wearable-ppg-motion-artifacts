# Results: wrist PPG heart rate under motion (PPG-DaLiA, leave-one-subject-out)

Every number in this file is printed by `scripts/results_tables.py` from `results/metrics.json` (made by `src/evaluate.py`) and `references/published_reiss2019.json` (published values copied from the paper). Per-window predictions: `results/predictions.csv.gz`. Not a medical device; no clinical claims.

**Setup:** all 15 subjects, 64,697 windows (8 s, 2 s shift), wrist PPG + wrist 3-axis accelerometer, reference HR from the dataset's ECG labels. Every method is scored leave-one-subject-out (15 folds; `tests/test_loso.py` checks that the test subject is never in training). The FFT methods have no trained parameters, so for them LOSO just means "scored per subject"; their four settings were hand-picked on S1, S2 and S6 (see BUILD-LOG), so those three subjects are slightly optimistic for the FFT rows.

## Overall
| Method (this repo, LOSO, 15 subjects) | MAE mean of subjects ± std (bpm) | MAE over all windows (bpm) | Worst activity |
|---|---|---|---|
| Plain FFT peak (baseline) | 17.24 ± 7.84 | 17.11 | stairs (36.16) |
| FFT + motion removal | 16.37 ± 5.79 | 16.32 | stairs (28.27) |
| FFT + motion removal + tracking | 12.15 ± 5.82 | 12.09 | cycling (24.8) |
| Gradient boosting (SelectKBest k=20) | 8.29 ± 3.88 | 8.21 | stairs (15.04) |
| 1D CNN on spectra | 12.43 ± 5.94 | 12.11 | cycling (25.72) |

What this shows:
- **Motion removal alone helps a little** (17.24 → 16.37 bpm mean of subjects), mostly on stairs (36.16 → 28.27), cycling and walking. It hurts on quiet activities (sitting 2.73 → 4.06, working 7.99 → 9.13), where subtracting a low-motion accelerometer spectrum removes some of the real pulse peak.
- **Motion removal + peak tracking** is the useful spectral method: 12.15 bpm, 29.5% lower MAE than the plain FFT baseline (17.24).
- **Gradient boosting** on 20 selected spectral/motion features is the best model here: 8.29 ± 3.88 bpm.
- **The small CNN** (22,785 parameters, 3 epochs, one seed, no ensemble) reaches 12.43 bpm: about the same as the spectral tracker, worse than gradient boosting. It is weakest on cycling (25.72) and sitting (11.06), probably undertrained (not tested further). It was cut to 3 epochs and every second training window to fit the time box.
- **Hard subject:** S5 is the worst subject for every method (plain FFT 40.16, GBM 19.94). Reiss et al. also report S5 as their worst session (CNN ensemble 18.51) and note it is an outlier in mean heart rate.

## Per activity (MAE in bpm, mean over windows of that activity)
| Activity | windows | Plain FFT peak (baseline) | FFT + motion removal | FFT + motion removal + tracking | Gradient boosting (SelectKBest k=20) | 1D CNN on spectra | Reiss 2019 CNN ensemble | Reiss 2019 SpaMaPlus |
|---|---|---|---|---|---|---|---|---|
| transient | 17515 | 21.13 | 19.68 | 14.27 | 9.86 | 12.3 | 8.77 | 14.34 |
| sitting | 4569 | 2.73 | 4.06 | 2.85 | 4.23 | 11.06 | 4.93 | 4.27 |
| stairs | 3239 | 36.16 | 28.27 | 24.28 | 15.04 | 23.72 | 16.98 | 25.42 |
| table_soccer | 2310 | 27.93 | 26.49 | 23.65 | 13.22 | 15.44 | 12.16 | 21.48 |
| cycling | 3473 | 27.9 | 23.57 | 24.8 | 12.57 | 25.72 | 12.48 | 11.97 |
| driving | 6843 | 11.76 | 13.56 | 8.71 | 6.66 | 9.66 | 4.96 | 6.24 |
| lunch | 13554 | 12.57 | 13.21 | 8.28 | 5.94 | 8.97 | 5.22 | 7.33 |
| walking | 4697 | 27.13 | 23.14 | 16.34 | 10.41 | 12.12 | 9.21 | 18.16 |
| working | 8497 | 7.99 | 9.13 | 6.01 | 4.84 | 8.42 | 3.84 | 4.91 |


## Per subject (MAE in bpm)
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Plain FFT peak (baseline) | 11.58 | 13.78 | 16.63 | 19.32 | 40.16 | 23.68 | 8.49 | 15.42 | 23.84 | 13.58 | 24.24 | 12.55 | 9.25 | 13.17 | 12.91 |
| FFT + motion removal | 12.97 | 16.67 | 13.2 | 19.62 | 32.85 | 18.1 | 8.29 | 18.13 | 23.03 | 13.1 | 19.5 | 12.92 | 11.22 | 13.79 | 12.1 |
| FFT + motion removal + tracking | 8.3 | 9.82 | 8.36 | 15.26 | 29.22 | 15.02 | 4.59 | 10.39 | 16.63 | 8.77 | 18.25 | 9.98 | 8.44 | 8.24 | 10.94 |
| Gradient boosting (SelectKBest k=20) | 7.91 | 7.26 | 4.83 | 7.49 | 19.94 | 9.48 | 3.95 | 11.41 | 9.47 | 5.82 | 8.97 | 11.79 | 5.19 | 5.12 | 5.66 |
| 1D CNN on spectra | 11.2 | 8.37 | 7.5 | 8.97 | 27.48 | 21.84 | 6.13 | 18.0 | 14.73 | 8.02 | 15.64 | 13.06 | 8.47 | 6.48 | 10.63 |

Feature selection: SelectKBest(f_regression, k=20) of 91 candidate features, refit inside each of the 15 training folds. Times selected (out of 15 folds):

## Feature selection (gradient boosting)

The 18 features picked in every fold are the three FFT HR estimates, the top spectral peaks before and after motion removal, the motion strength, a few low-HR spectral bins and accelerometer variance: the model mostly learns *when to trust which spectral estimate* given how much the wrist moves.

## Comparison with published PPG-DaLiA results
Source: Reiss A, Indlekofer I, Schmidt P, Van Laerhoven K. *Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks.* Sensors 2019;19(14):3079, doi:10.3390/s19143079 (values copied from the PMC full text, Tables 4, 10, 11, 12; see `references/published_reiss2019.json`). Their "All" value is the mean of the 15 per-session MAEs ± std across sessions, the same as our "mean of subjects" column.

| Method | MAE (bpm), PPG-DaLiA, LOSO | Source |
|---|---|---|
| Schaeck2017 (classical) | 20.45 ± 7.1 | Reiss 2019 |
| **This repo: plain FFT peak** | **17.24 ± 7.84** | `results/metrics.json` |
| **This repo: FFT + motion removal** | **16.37 ± 5.79** | `results/metrics.json` |
| SpaMa (classical, spectral) | 15.56 ± 7.5 | Reiss 2019 |
| **This repo: 1D CNN on spectra** | **12.43 ± 5.94** | `results/metrics.json` |
| **This repo: FFT + motion removal + tracking** | **12.15 ± 5.82** | `results/metrics.json` |
| SpaMaPlus (classical, spectral + tracking) | 11.06 ± 4.8 | Reiss 2019 |
| CNN constrained, 26K params | 9.99 ± 5.9 | Reiss 2019 |
| CNN average of 7 runs | 8.82 ± 3.8 | Reiss 2019 |
| **This repo: gradient boosting, 20 selected features** | **8.29 ± 3.88** | `results/metrics.json` |
| CNN ensemble of 7 (Deep PPG) | 7.65 ± 4.2 | Reiss 2019 |

Setup differences (why this is a rough comparison, not a ranking):
- Same data, same 8 s / 2 s windows, same 64,697 windows, same leave-one-subject-out idea. Their per-activity counts differ from ours by ≤ 20 windows (activity per window assigned differently).
- Their CNN takes time-frequency spectra of PPG + accelerometer and is trained 15,000 iterations with a validation set; ours is a much smaller run (3 epochs, half the training windows, one seed).
- Our spectral parameters were hand-picked on 3 of the 15 subjects (not nested); their classical methods' parameters come from the IEEE datasets.
- Our gradient boosting is not a method from the paper; it stacks our own FFT estimates and motion features.
- No significance test was run; differences of ~1 bpm between rows should not be read as real.

## Where methods fail
- **Stairs, cycling, table soccer** stay the hardest (worst activity for every method). On cycling, motion removal + tracking only moves 27.90 → 24.80 even though the wrist moves a lot (median band-passed acc std 0.173 g, `results/data_card.json`); gradient boosting gets it to 12.57. Why the spectral tracker fails there was not investigated in this session.
- **Sitting** is easy for the plain FFT (2.73) and the CNN is bad there (11.06), likely because the short CNN run did not learn the low-HR range well (not checked further).

## MATLAB / Octave cross-check
`matlab/spectral_check.m` (plain MATLAB syntax, no toolboxes) re-implements the FFT motion-removal core and ran under GNU Octave 10.3.0 on 20 band-passed S1 windows (`results/matlab_windows.csv`). Against Python (`results/matlab_check.json`): max difference 1.0e-10 bpm for the plain and motion-removed HR peaks, 1.2e-11 on the cleaned spectra, 0 on the motion weight. The band-pass filter itself is applied in Python before export (MATLAB's `filtfilt` needs the Signal Processing Toolbox). A MATLAB Online run (`results/matlab_online_run.txt`) is still to do by Dheepak.

## MATLAB Online check (2026-11-09)
`matlab/spectral_check.m` run in **MATLAB R2026b Update 1** (MATLAB Online) on the same 20 windows: max |MATLAB − Python| = 1.0e-10 bpm (plain and motion-removed HR), 0 on motion weight, 1.2e-11 on cleaned spectra. Log: `results/matlab_online_run.txt`; outputs: `results/matlab_online/`; comparison: `results/matlab_online_check.json` (spectra CSV stays in gitignored `data/`).
