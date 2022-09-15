from combio.stats import *
from combio.core import Sequence
import numpy as np


def test_ugof():
    seq = Sequence.generate_isochronous(n=10, ioi=500)
    assert get_ugof(seq, 500, 'median') == 0.0


def test_ks():
    rng = np.random.default_rng(seed=123)
    seq = Sequence.generate_random_uniform(n=10, a=400, b=600, rng=rng)
    assert ks_test(seq.iois)[0] == 0.27372630223062433


def test_npvi():
    seq = Sequence.generate_isochronous(n=10, ioi=500)
    assert get_npvi(seq) == 0.0


def test_acf():
    rng = np.random.default_rng(seed=123)
    seq = Sequence.generate_random_uniform(n=10, a=450, b=550, rng=rng)
    fig, ax = acf_plot(seq, 1, 250, 20, suppress_display=True)
    assert fig
    assert ax


