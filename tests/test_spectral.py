import numpy as np
from src import spectral as sp

FS, L = 32, 256
t = np.arange(L) / FS


def test_plain_fft_finds_clean_pulse():
    ppg = np.sin(2 * np.pi * 1.25 * t)          # 75 bpm
    assert abs(sp.plain_fft_hr(ppg) - 75) < 1.5


def test_motion_removal_beats_plain_when_motion_dominates():
    rng = np.random.default_rng(0)
    hr_f, mo_f = 1.5, 2.2                         # 90 bpm pulse, 132 bpm arm swing
    ppg = 0.4 * np.sin(2 * np.pi * hr_f * t) + 1.0 * np.sin(2 * np.pi * mo_f * t) + 0.05 * rng.standard_normal(L)
    acc = np.stack([0.3 * np.sin(2 * np.pi * mo_f * t + p) for p in (0, 1, 2)], axis=1)
    plain = sp.plain_fft_hr(ppg)
    _, Sc = sp.clean_spectrum(ppg, acc)
    removed = sp.peak_bpm(Sc)
    assert abs(plain - mo_f * 60) < 2             # baseline locks onto motion
    assert abs(removed - hr_f * 60) < 2           # motion removal recovers HR


def test_no_motion_keeps_spectrum():
    ppg = np.sin(2 * np.pi * 1.0 * t)
    acc = np.zeros((L, 3)) + 1e-6 * np.random.default_rng(1).standard_normal((L, 3))
    Sp, Sc = sp.clean_spectrum(ppg, acc)
    assert np.allclose(Sp, Sc, atol=1e-3)


def test_tracking_stays_near_previous():
    spectra = []
    for k in range(5):
        x = np.sin(2 * np.pi * 1.2 * t) + (1.5 if k == 3 else 0) * np.sin(2 * np.pi * 2.8 * t)
        spectra.append(sp.spectrum(x))
    est = sp.track(np.array(spectra), init_bpm=72)
    assert np.all(np.abs(est - 72) < 3)          # spurious 168 bpm peak in window 3 ignored
