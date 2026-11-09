"""Data card numbers for DATA_QUALITY.md -> results/data_card.json + results/windows_by_subject_activity.csv"""
import json, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
ACT = {0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer", 4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working"}
df = pd.read_csv(ROOT / "data" / "features.csv.gz", usecols=lambda c: c in {"subject", "hr_ref", "activity", "acc_bp_std_max", "ppg_std", "motion_weight"})
order = sorted(df.subject.unique(), key=lambda s: int(s[1:]))
tab = pd.crosstab(df.subject, df.activity.map(ACT)).reindex(order)
tab["total"] = tab.sum(axis=1); tab.to_csv(ROOT / "results" / "windows_by_subject_activity.csv")
card = {
 "subjects": order, "n_subjects": len(order), "n_windows": int(len(df)),
 "window": "8 s, 2 s shift; PPG = wrist E4 BVP 64 Hz decimated to 32 Hz; ACC = wrist E4 3-axis 32 Hz (g)",
 "hr_ref_bpm": {"min": round(float(df.hr_ref.min()), 1), "median": round(float(df.hr_ref.median()), 1), "max": round(float(df.hr_ref.max()), 1)},
 "nan_hr_ref": int(df.hr_ref.isna().sum()),
 "windows_per_activity": {ACT[a]: int(n) for a, n in df.activity.value_counts().sort_index().items()},
 "mean_hr_per_activity": {ACT[a]: round(float(v), 1) for a, v in df.groupby("activity").hr_ref.mean().items()},
 "windows_per_subject": {s: int(n) for s, n in df.subject.value_counts().reindex(order).items()},
 "flat_ppg_windows (bp std < 1e-3)": int((df.ppg_std < 1e-3).sum()),
 "share_windows_full_motion_weight": round(float((df.motion_weight >= 1).mean()), 3),
 "median_acc_bp_std_g_per_activity": {ACT[a]: round(float(v), 3) for a, v in df.groupby("activity").acc_bp_std_max.median().items()},
}
(ROOT / "results" / "data_card.json").write_text(json.dumps(card, indent=1)); print(json.dumps(card, indent=1)); print(tab.to_string())
