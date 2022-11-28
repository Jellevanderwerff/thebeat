from thebeat.stats import *
from thebeat.core import Sequence
import numpy as np


def test_ugof():
    seq = Sequence.generate_isochronous(n=10, ioi=500)
    assert get_ugof(seq, 500, 'median') == 0.0


def test_ks():
    rng = np.random.default_rng(seed=123)
    seq = Sequence.generate_random_uniform(n=10, a=400, b=600, rng=rng)
    assert ks_test(seq.iois)[0] == 0.2724021511351798


def test_npvi():
    seq = Sequence.generate_isochronous(n=10, ioi=500)
    assert get_npvi(seq) == 0.0


def test_acf():
    # ms
    rng = np.random.default_rng(seed=123)
    seq = Sequence.generate_random_uniform(n=10, a=450, b=550, rng=rng)
    fig, ax = acf_plot(seq, resolution=1, smoothing_window=50, smoothing_sd=10, suppress_display=True)
    assert fig, ax

    # s
    seq = Sequence.generate_random_uniform(n=10, a=0.45, b=0.55, rng=rng)
    fig, ax = acf_plot(seq, resolution=0.001, smoothing_window=50, smoothing_sd=10, suppress_display=True)
    assert fig, ax
