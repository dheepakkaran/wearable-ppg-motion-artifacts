"""Download PPG-DaLiA (UCI ML Repository #495, CC BY 4.0) into data/raw/ and unpack the inner data.zip.

Subject pickles are NOT extracted here; src/extract.py reads them one at a time from data.zip
so only one subject is ever on disk/in RAM at once.
"""
import pathlib, subprocess, sys, zipfile

URL = "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"
RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    outer = RAW / "ppg_dalia.zip"
    inner = RAW / "data.zip"
    if not inner.exists():
        if not outer.exists() or not zipfile.is_zipfile(outer):
            subprocess.run(["curl", "-L", "-o", str(outer), URL], check=True)
        with zipfile.ZipFile(outer) as z:
            print("outer zip:", z.namelist())
            z.extractall(RAW)
    with zipfile.ZipFile(inner) as z:
        pk = sorted(n for n in z.namelist() if n.endswith(".pkl"))
    print(f"{len(pk)} subject pickles in data.zip:", pk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
