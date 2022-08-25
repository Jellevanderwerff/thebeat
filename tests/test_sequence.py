import numpy as np
import pytest
from combio.core import *


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_iois(rng):
    seq = Sequence.generate_random_normal(10, 500, 25, rng=rng)

    assert isinstance(seq.iois, np.ndarray)
    assert len(seq.iois) == 9
    assert np.all(seq.iois == [508., 474., 519., 524., 451., 467., 503., 492., 500.])
    assert len(seq.onsets) == 10
    assert seq.metrical is False

    integer_ratios = [1, 5, 8, 2, 5, 4, 4, 2, 1]
    seq = Sequence.from_integer_ratios(numerators=integer_ratios, value_of_one_in_ms=500)
    assert seq.integer_ratios == integer_ratios
