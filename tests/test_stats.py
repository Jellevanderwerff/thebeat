from combio.stats import *
from combio.core import Sequence
import numpy as np


def test_ugof():
    seq = Sequence.generate_isochronous(n=10, ioi=500)
    assert ugof(seq, 500, 'median') == 0.0


def test_ks():
    rng = np.random.default_rng(seed=123)
    seq = Sequence.generate_random_uniform(n=10, a=400, b=600, rng=rng)
    assert ks_test(seq.iois)[0] == 0.27372630223062433