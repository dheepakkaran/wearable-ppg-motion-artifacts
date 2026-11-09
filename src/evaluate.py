"""LOSO evaluation of every method -> results/metrics.json, results/predictions.csv.gz, results/*.csv.

Methods
  plain_fft        : band-pass + FFT peak of the PPG, no motion removal (baseline, no training)
  fft_motion_removed: accelerometer spectrum subtracted from the PPG spectrum, per-window peak
  fft_motion_tracked: motion-removed spectrum + causal HR peak tracking (the spectral method)
  gbm              : SelectKBest(f_regression, k) -> HistGradientBoostingRegressor, fit per fold
  cnn              : small 1D CNN on PPG / cleaned PPG / 3-axis accelerometer spectra, fit per fold
"""
import json, pathlib, sys, time
from collections import Counter
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from .loso import loso_splits

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ACT = {0: "transient", 1: "sitting", 2: "stairs", 3: "table_soccer", 4: "cycling", 5: "driving",
       6: "lunch", 7: "walking", 8: "working"}
K_BEST = 20
META = ["subject", "hr_ref", "activity", "window"]


def run_gbm(df):
    feats = [c for c in df.columns if c not in META]
    X, y = df[feats].values, df.hr_ref.values
    pred = np.empty(len(df)); chosen = Counter()
    for s, tr, te in loso_splits(df.subject.values):
        assert not set(df.subject.values[tr]) & set(df.subject.values[te])
        m = Pipeline([("select", SelectKBest(f_regression, k=K_BEST)),
                      ("gbm", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                            max_leaf_nodes=31, random_state=0))])
        m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
        chosen.update(np.array(feats)[m.named_steps["select"].get_support()])
    return pred, {"k": K_BEST, "n_candidate_features": len(feats),
                  "times_selected_out_of_folds": dict(chosen.most_common())}


def run_cnn(epochs=3, seed=0, train_stride=2, batch=512):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(4)
    d = np.load(ROOT / "data" / "cnn_inputs.npz", allow_pickle=True)
    X, y, subj = d["X"], d["y"], d["subject"]
    pred = np.empty(len(y)); t0 = time.time()

    def net():
        return nn.Sequential(nn.Conv1d(5, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                             nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                             nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(8),
                             nn.Flatten(), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
    n_params = sum(p.numel() for p in net().parameters())
    for s, tr, te in loso_splits(subj):
        assert not set(subj[tr]) & set(subj[te])
        tr = tr[::train_stride]   # 8 s windows with 2 s shift overlap 75%; every 2nd window keeps CPU time <= ~5 min
        m = net(); opt = torch.optim.Adam(m.parameters(), 2e-3)
        Xt, yt = torch.from_numpy(X[tr]), torch.from_numpy(y[tr] / 100.0)
        for ep in range(epochs):
            m.train(); perm = torch.randperm(len(tr))
            for i in range(0, len(tr), batch):
                b = perm[i:i + batch]
                loss = (m(Xt[b]).squeeze(1) - yt[b]).abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pred[te] = m(torch.from_numpy(X[te])).squeeze(1).numpy() * 100.0
        print(f"cnn fold {s} mae {np.abs(pred[te]-y[te]).mean():.2f} t={time.time()-t0:.0f}s", flush=True)
    return pred, {"epochs": epochs, "train_stride": train_stride, "batch": batch, "optimizer": "Adam lr 2e-3, L1 loss", "params": n_params, "input": "5 x 113 spectra (0.5-4 Hz): PPG, motion-removed PPG, acc x/y/z",
                  "train_seconds_total": round(time.time() - t0, 1)}


def summarize(df, methods):
    out = {}
    for m in methods:
        e = (df[m] - df.hr_ref).abs()
        per_subj = e.groupby(df.subject).mean()
        out[m] = {"mae_all_windows": round(float(e.mean()), 2),
                  "mae_mean_of_subjects": round(float(per_subj.mean()), 2),
                  "mae_std_of_subjects": round(float(per_subj.std(ddof=0)), 2),
                  "per_activity": {ACT[int(a)]: round(float(v), 2) for a, v in e.groupby(df.activity).mean().items()},
                  "per_subject": {k: round(float(v), 2) for k, v in per_subj.items()}}
    return out


def main(skip_cnn=False):
    RES.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "features.csv.gz")
    out = df[META].copy()
    out["plain_fft"] = df.hr_plain_fft; out["fft_motion_removed"] = df.hr_motion_removed
    out["fft_motion_tracked"] = df.hr_tracked
    t = time.time(); out["gbm"], gbm_info = run_gbm(df); gbm_info["seconds"] = round(time.time() - t, 1)
    methods = ["plain_fft", "fft_motion_removed", "fft_motion_tracked", "gbm"]
    info = {"gbm": gbm_info}
    if not skip_cnn:
        out["cnn"], info["cnn"] = run_cnn(); methods.append("cnn")
    metrics = {"n_windows": int(len(df)), "subjects": sorted(df.subject.unique().tolist(), key=lambda s: int(s[1:])),
               "validation": "leave-one-subject-out", "windows_per_activity": {ACT[int(a)]: int(n) for a, n in df.activity.value_counts().sort_index().items()},
               "methods": summarize(out, methods), "model_info": info}
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=1))
    out.round(2).to_csv(RES / "predictions.csv.gz", index=False)
    tab = pd.DataFrame({m: metrics["methods"][m]["per_activity"] for m in methods})
    tab.loc["ALL (window mean)"] = [metrics["methods"][m]["mae_all_windows"] for m in methods]
    tab.loc["ALL (mean of subjects)"] = [metrics["methods"][m]["mae_mean_of_subjects"] for m in methods]
    tab.to_csv(RES / "mae_by_activity.csv")
    pd.DataFrame({m: metrics["methods"][m]["per_subject"] for m in methods}).to_csv(RES / "mae_by_subject.csv")
    print(tab.to_string())


if __name__ == "__main__":
    main(skip_cnn="--no-cnn" in sys.argv)
