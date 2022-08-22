import numpy as np
import pytest

import stimulus as stim


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_iois(rng):
    seq = stim.Sequence.generate_random_normal(10, 500, 25, rng=rng)

    assert isinstance(seq.iois, np.ndarray)
    # assert seq.iois.dtype == np.float32
    assert len(seq.iois) == 9
    assert np.all(seq.iois == [508., 474., 519., 524., 451., 467., 503., 492., 500.])
    assert len(seq.onsets) == 10


def test_integer_ratios():
    seq = stim.Sequence.from_integer_ratios([4, 2, 2, 3, 1, 1], value_of_one_in_ms=500)
    assert seq.integer_ratios_from_total_duration == [4, 2, 2, 3, 1, 1]
