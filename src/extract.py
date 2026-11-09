"""Stream each subject pickle out of data/raw/data.zip (one at a time), cut 8 s / 2 s windows of
wrist BVP + wrist 3-axis accelerometer aligned with the dataset's reference HR labels, save a compact
per-subject file data/windows/S<k>.npz, then free everything before the next subject (RAM is tight).

Wrist BVP 64 Hz is decimated to 32 Hz (HR content < 4 Hz) so PPG and accelerometer share fs = 32 Hz.
Window i starts at 2*i seconds (the label convention documented in the PPG-DaLiA readme).
"""
import codecs, gc, json, pathlib, pickle, sys, zipfile
import numpy as np
from scipy.signal import decimate

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "data.zip"
OUT = ROOT / "data" / "windows"
FS = 32
WIN_S, SHIFT_S = 8, 2
ACT_FS = 4


SKIP_BYTES = 60 * 2**20   # protocol-0 STRING lines longer than this (700 Hz chest arrays) are never held in RAM


class _Skipped:
    """Placeholder for a chest array whose data line was streamed past without loading."""


class LeanUnpickler(pickle._Unpickler):
    """The PPG-DaLiA pickles are Python-2 protocol-0 (text) pickles of ~1.2-1.6 GB, almost all of it
    700 Hz chest signals. A normal pickle.load needs that much RAM; this pure-Python unpickler reads each
    STRING line in 1 MB chunks and discards lines above SKIP_BYTES, so peak RAM stays ~100 MB."""
    dispatch = dict(pickle._Unpickler.dispatch)

    def load_string(self):
        chunks, total = [], 0
        while True:
            c = self._file_readline(2**20)
            total += len(c)
            if total <= SKIP_BYTES:
                chunks.append(c)
            elif chunks:
                chunks = []
            if not c or c.endswith(b"\n"):
                break
        if total > SKIP_BYTES:
            self.append(_Skipped()); return
        data = b"".join(chunks)[:-1]
        if len(data) >= 2 and data[0] == data[-1] and data[0] in b"\"'":
            data = data[1:-1]
        else:
            raise pickle.UnpicklingError("the STRING opcode argument must be quoted")
        self.append(self._decode_string(codecs.escape_decode(data)[0]))
    dispatch[pickle.STRING[0]] = load_string

    def load_build(self):
        state = self.stack[-1]
        if isinstance(state, tuple) and any(isinstance(x, _Skipped) for x in state):
            self.stack.pop(); self.stack[-1] = _Skipped(); return
        pickle._Unpickler.load_build(self)
    dispatch[pickle.BUILD[0]] = load_build


def window_subject(d):
    bvp = np.asarray(d["signal"]["wrist"]["BVP"], dtype=np.float64).ravel()
    acc = np.asarray(d["signal"]["wrist"]["ACC"], dtype=np.float32)  # pkl wrist ACC is already in g (mean |a| ~ 1)
    hr = np.asarray(d["label"], dtype=np.float32).ravel()
    act = np.asarray(d["activity"]).ravel().astype(np.int16)
    bvp32 = decimate(bvp, 2, zero_phase=True).astype(np.float32)
    L = WIN_S * FS
    n = min(len(hr), (len(bvp32) - L) // (SHIFT_S * FS) + 1, (len(acc) - L) // (SHIFT_S * FS) + 1)
    ppg = np.empty((n, L), np.float32)
    ax = np.empty((n, L, 3), np.float32)
    wa = np.empty(n, np.int16)
    for i in range(n):
        s = i * SHIFT_S * FS
        ppg[i] = bvp32[s:s + L]
        ax[i] = acc[s:s + L]
        a = act[i * SHIFT_S * ACT_FS:(i * SHIFT_S + WIN_S) * ACT_FS]
        wa[i] = np.bincount(a).argmax() if len(a) else -1
    info = dict(n_label=int(len(hr)), n_windows=int(n), bvp_len=int(len(bvp)), acc_len=int(len(acc)),
                act_len=int(len(act)), duration_min=round(len(bvp) / 64 / 60, 1))
    return ppg, ax, hr[:n], wa, info


def main(subjects=None):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    with zipfile.ZipFile(RAW) as z:
        names = sorted((n for n in z.namelist() if n.endswith(".pkl")),
                       key=lambda s: int(pathlib.Path(s).stem.lstrip("S")))
        for name in names:
            sid = pathlib.Path(name).stem
            if subjects and sid not in subjects:
                continue
            dst = OUT / f"{sid}.npz"
            if dst.exists():
                print(sid, "exists, skip"); continue
            with z.open(name) as f:
                d = LeanUnpickler(f, encoding="latin1").load()
            d["signal"].pop("chest", None)  # 700 Hz chest channels (already skipped while streaming)
            gc.collect()
            ppg, ax, hr, wa, info = window_subject(d)
            del d; gc.collect()
            np.savez(dst, ppg=ppg, acc=ax, hr=hr, activity=wa)
            meta[sid] = info
            print(sid, info, flush=True)
            del ppg, ax, hr, wa; gc.collect()
    mp = OUT / "extract_meta.json"
    old = json.loads(mp.read_text()) if mp.exists() else {}
    old.update(meta)
    mp.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
