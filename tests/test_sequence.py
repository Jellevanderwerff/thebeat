import numpy as np
import pytest
import combio.core


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_iois(rng):
    seq = combio.core.Sequence.generate_random_normal(10, 500, 25, rng=rng)

    assert isinstance(seq.iois, np.ndarray)
    assert len(seq.iois) == 9
    assert np.all(np.round(seq.iois) == [508., 474., 519., 524., 451., 467., 503., 492., 500.])
    assert len(seq.onsets) == 10
    assert seq.metrical is False

    # from and to integer ratios
    integer_ratios = [1, 5, 8, 2, 5, 4, 4, 2, 1]
    seq = combio.core.Sequence.from_integer_ratios(numerators=integer_ratios, value_of_one=500)
    assert np.all(seq.integer_ratios == integer_ratios)

    # test whether copy of IOIs is returned instead of object itself
    s = combio.core.Sequence([1, 2, 3, 4])
    iois = s.iois
    iois[0] = -42
    assert s.iois[0] != -42

    seq = combio.core.Sequence.generate_isochronous(4, 500, metrical=True)
    with pytest.raises(ValueError):
        seq.onsets = [0, 50, 100]


def test_exception():
    seq = combio.core.Sequence.generate_isochronous(n=10, ioi=500)
    seq.change_tempo(0.5)
    with pytest.raises(ValueError):
        seq.change_tempo(-1)

    with pytest.raises(ValueError):
        combio.core.Sequence.from_onsets([20, 20, 20])


def test_onset_not_zero():
    seq = combio.core.Sequence.from_onsets([20, 50, 100])
    assert np.all(seq.onsets == [20, 50, 100])
    assert np.all(seq.iois == [30, 50])

    with pytest.raises(ValueError):
        combio.core.Sequence([50, 50, 50], metrical=True, first_onset=50)
