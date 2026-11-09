"""Leave-one-subject-out splits (one test subject per fold; that subject never appears in training)."""
import numpy as np


def loso_splits(subjects):
    subjects = np.asarray(subjects)
    for s in sorted(np.unique(subjects), key=lambda x: int(str(x)[1:])):
        test = subjects == s
        yield s, np.where(~test)[0], np.where(test)[0]
