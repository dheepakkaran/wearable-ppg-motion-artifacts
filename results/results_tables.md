| Method (this repo, LOSO, 15 subjects) | MAE mean of subjects ± std (bpm) | MAE over all windows (bpm) | Worst activity |
|---|---|---|---|
| Plain FFT peak (baseline) | 17.24 ± 7.84 | 17.11 | stairs (36.16) |
| FFT + motion removal | 16.37 ± 5.79 | 16.32 | stairs (28.27) |
| FFT + motion removal + tracking | 12.15 ± 5.82 | 12.09 | cycling (24.8) |
| Gradient boosting (SelectKBest k=20) | 8.29 ± 3.88 | 8.21 | stairs (15.04) |
| 1D CNN on spectra | 12.43 ± 5.94 | 12.11 | cycling (25.72) |

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

| Subject | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | S15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Plain FFT peak (baseline) | 11.58 | 13.78 | 16.63 | 19.32 | 40.16 | 23.68 | 8.49 | 15.42 | 23.84 | 13.58 | 24.24 | 12.55 | 9.25 | 13.17 | 12.91 |
| FFT + motion removal | 12.97 | 16.67 | 13.2 | 19.62 | 32.85 | 18.1 | 8.29 | 18.13 | 23.03 | 13.1 | 19.5 | 12.92 | 11.22 | 13.79 | 12.1 |
| FFT + motion removal + tracking | 8.3 | 9.82 | 8.36 | 15.26 | 29.22 | 15.02 | 4.59 | 10.39 | 16.63 | 8.77 | 18.25 | 9.98 | 8.44 | 8.24 | 10.94 |
| Gradient boosting (SelectKBest k=20) | 7.91 | 7.26 | 4.83 | 7.49 | 19.94 | 9.48 | 3.95 | 11.41 | 9.47 | 5.82 | 8.97 | 11.79 | 5.19 | 5.12 | 5.66 |
| 1D CNN on spectra | 11.2 | 8.37 | 7.5 | 8.97 | 27.48 | 21.84 | 6.13 | 18.0 | 14.73 | 8.02 | 15.64 | 13.06 | 8.47 | 6.48 | 10.63 |

Feature selection: SelectKBest(f_regression, k=20) of 91 candidate features, refit inside each of the 15 training folds. Times selected (out of 15 folds):
`hr_plain_fft` 15, `hr_motion_removed` 15, `hr_tracked` 15, `motion_weight` 15, `acc_bp_std_max` 15, `raw_peak1_bpm` 15, `raw_peak2_bpm` 15, `raw_peak3_bpm` 15, `clean_peak1_bpm` 15, `clean_peak2_bpm` 15, `clean_peak3_bpm` 15, `clean_bin_54bpm` 15, `clean_bin_60bpm` 15, `ppg_zero_cross_rate` 15, `acc_mag_mean` 15, `acc_mag_std` 15, `acc_x_std` 15, `acc_y_std` 15, `raw_bin_114bpm` 14, `clean_bin_66bpm` 13, `acc_z_std` 1, `raw_bin_108bpm` 1, `raw_bin_120bpm` 1
