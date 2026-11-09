import numpy as np
from src.loso import loso_splits


def test_no_subject_in_train_and_test():
    subj = np.array(["S1"] * 5 + ["S2"] * 3 + ["S10"] * 4)
    seen = []
    for s, tr, te in loso_splits(subj):
        assert set(subj[te]) == {s}
        assert s not in set(subj[tr])
        assert len(tr) + len(te) == len(subj)
        seen.append(s)
    assert seen == ["S1", "S2", "S10"]
