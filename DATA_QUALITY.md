# Data quality: PPG-DaLiA

Every number here comes from `results/data_card.json` and `results/windows_by_subject_activity.csv` (made by `scripts/data_card.py`).

## Source and license
- **Dataset:** PPG-DaLiA, UCI Machine Learning Repository #495, https://archive.ics.uci.edu/dataset/495/ppg+dalia, DOI 10.24432/C53890.
- **License (as stated on the UCI page, read 2026-11-09):** "This dataset is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license."
- **Citation:** Reiss, A., Indlekofer, I., & Schmidt, P. (2019). PPG-DaLiA [Dataset]. UCI Machine Learning Repository. Paper: Reiss et al., *Sensors* 2019, 19(14), 3079.
- Download: one 2.7 GB `data.zip` (inside the UCI zip) in gitignored `data/raw/`. Nothing from it is committed.

## What was used
| Item | Value |
|---|---|
| Subjects | all 15 (S1-S15) |
| Windows | 64,697 (8 s window, 2 s shift; same total as Table 11 of Reiss et al. 2019) |
| PPG | wrist Empatica E4 BVP, 64 Hz, decimated to 32 Hz (`scipy.signal.decimate`, zero phase) |
| Accelerometer | wrist E4 3-axis, 32 Hz. In `SX.pkl` it is already in g (mean magnitude about 1); the readme's "1/64 g" applies to the raw `ACC.csv` |
| Reference HR | `data['label']` in each `SX.pkl`: mean instantaneous HR from manually corrected chest-ECG R-peaks per 8 s window (readme) |
| Activity per window | majority vote of the 4 Hz `data['activity']` samples inside the window |
| Reference HR range | 41.7 to 187.0 bpm, median 85.3; 0 missing |
| Flat PPG windows | 0 (band-passed PPG std < 1e-3) |

## Windows per activity
| Activity | Windows | Mean reference HR (bpm) | Median wrist motion (band-passed acc std, g) |
|---|---|---|---|
| transient (ID 0) | 17,515 | 94.7 | 0.191 |
| sitting (1) | 4,569 | 61.0 | 0.004 |
| stairs (2) | 3,239 | 118.8 | 0.263 |
| table soccer (3) | 2,310 | 90.4 | 0.165 |
| cycling (4) | 3,473 | 123.2 | 0.173 |
| driving (5) | 6,843 | 84.7 | 0.084 |
| lunch (6) | 13,554 | 83.6 | 0.064 |
| walking (7) | 4,697 | 99.3 | 0.235 |
| working (8) | 8,497 | 76.3 | 0.021 |

Per-activity counts differ from the paper's Table 11 by at most 20 windows per activity (e.g. transient 17,515 here vs 17,495 there), because the paper does not say how it assigns one activity to a window; the majority vote here is our own choice.

## Known issues
- **S6** has only 2,622 windows: no lunch, walking or working (the readme: a hardware issue means S6.pkl holds only the valid first ~1.5 h). It is still one LOSO fold.
- **S10** has the most windows (5,321), mostly longer transient periods.
- Windows overlap by 75%, so neighbouring windows are not independent. That is why every split is by subject (LOSO), never by window.
- 69.3% of windows have full motion-removal weight (band-passed acc std >= 0.05 g), so most of the data has visible wrist motion.
- Memory: each `SX.pkl` is a 0.8-1.6 GB Python-2 text pickle, mostly 700 Hz chest signals. `src/extract.py` streams it with a pure-Python unpickler that skips the chest arrays, so peak RAM per subject was about 240 MB.
