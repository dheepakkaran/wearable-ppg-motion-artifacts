"""Print the markdown tables used in RESULTS.md straight from results/metrics.json + references/ (no hand-typed numbers)."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
M = json.loads((ROOT / "results" / "metrics.json").read_text())
P = json.loads((ROOT / "references" / "published_reiss2019.json").read_text())
meth = list(M["methods"])
names = {"plain_fft": "Plain FFT peak (baseline)", "fft_motion_removed": "FFT + motion removal",
         "fft_motion_tracked": "FFT + motion removal + tracking", "gbm": "Gradient boosting (SelectKBest k=20)", "cnn": "1D CNN on spectra"}
print("| Method (this repo, LOSO, 15 subjects) | MAE mean of subjects ± std (bpm) | MAE over all windows (bpm) | Worst activity |")
print("|---|---|---|---|")
for m in meth:
    r = M["methods"][m]; pa = r["per_activity"]; w = max(pa, key=pa.get)
    print(f"| {names[m]} | {r['mae_mean_of_subjects']} ± {r['mae_std_of_subjects']} | {r['mae_all_windows']} | {w} ({pa[w]}) |")
print()
acts = list(M["methods"][meth[0]]["per_activity"])
pub = P["per_activity_mae_bpm_table11"]
print("| Activity | windows | " + " | ".join(names[m] for m in meth) + " | Reiss 2019 CNN ensemble | Reiss 2019 SpaMaPlus |")
print("|---|---|" + "---|" * (len(meth) + 2))
for a in acts:
    print(f"| {a} | {M['windows_per_activity'][a]} | " + " | ".join(str(M["methods"][m]["per_activity"][a]) for m in meth) + f" | {pub[a][2]} | {pub[a][3]} |")
print()
subs = M["subjects"]
print("| Subject | " + " | ".join(subs) + " |"); print("|---|" + "---|" * len(subs))
for m in meth:
    print(f"| {names[m]} | " + " | ".join(str(M["methods"][m]["per_subject"][s]) for s in subs) + " |")
print()
g = M["model_info"]["gbm"]
print(f"Feature selection: SelectKBest(f_regression, k={g['k']}) of {g['n_candidate_features']} candidate features, refit inside each of the 15 training folds. Times selected (out of 15 folds):")
print(", ".join(f"`{k}` {v}" for k, v in g["times_selected_out_of_folds"].items()))
