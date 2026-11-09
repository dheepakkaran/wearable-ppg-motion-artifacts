"""Features + CNN spectra for every extracted subject -> data/features.csv.gz, data/cnn_inputs.npz."""
import gc, pathlib
import numpy as np, pandas as pd
from .features import load_subject, subject_features

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    files = sorted((ROOT / "data" / "windows").glob("S*.npz"), key=lambda p: int(p.stem[1:]))
    frames, X = [], []
    for f in files:
        ppg, acc, hr, act = load_subject(f)
        F, cnn = subject_features(ppg, acc)
        df = pd.DataFrame(F); df.insert(0, "window", np.arange(len(hr)))
        df.insert(0, "activity", act); df.insert(0, "hr_ref", hr); df.insert(0, "subject", f.stem)
        frames.append(df); X.append(cnn)
        print(f.stem, len(df), flush=True)
        del ppg, acc; gc.collect()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(ROOT / "data" / "features.csv.gz", index=False)
    np.savez(ROOT / "data" / "cnn_inputs.npz", X=np.concatenate(X), y=df.hr_ref.values.astype(np.float32),
             subject=df.subject.values)
    print("features", df.shape)


if __name__ == "__main__":
    main()
