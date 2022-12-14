import numpy as np
import pytest
import thebeat.core
import matplotlib.pyplot as plt


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_iois(rng):
    seq = thebeat.core.Sequence.generate_random_normal(10, 500, 25, rng=rng)

    assert isinstance(seq.iois, np.ndarray)
    assert len(seq.iois) == 9
    assert np.all(np.round(seq.iois) == [508., 474., 519., 524., 451., 467., 503., 492., 500.])
    assert len(seq.onsets) == 10
    assert seq.end_with_interval is False

    # from and to integer ratios
    integer_ratios = [1, 5, 8, 2, 5, 4, 4, 2, 1]
    seq = thebeat.core.Sequence.from_integer_ratios(numerators=integer_ratios, value_of_one=500)
    assert np.all(seq.integer_ratios == integer_ratios)

    # test whether copy of IOIs is returned instead of object itself
    s = thebeat.core.Sequence([1, 2, 3, 4])
    iois = s.iois
    iois[0] = -42
    assert s.iois[0] != -42

    seq = thebeat.core.Sequence.generate_isochronous(4, 500, end_with_interval=True)
    with pytest.raises(ValueError):
        seq.onsets = [0, 50, 100]


def test_iois_property(rng):
    seq = thebeat.core.Sequence.generate_random_normal(10, 500, 25, rng=rng)
    assert len(seq.iois) == 9

    iois = seq.iois.copy()
    seq.iois[0] = 42
    assert np.all(seq.iois == iois)

    seq.iois = [5, 6, 7]
    assert isinstance(seq.iois, np.ndarray)
    assert len(seq.iois) == 3
    assert len(seq.onsets) == 4
    assert np.all(seq.onsets == [0, 5, 11, 18])

    with pytest.raises(ValueError, match=r"Inter-onset intervals \(IOIs\) cannot be zero or negative"):
        seq.iois = [1, 2, 3, 0, 4, 5]

    with pytest.raises(ValueError, match=r"Inter-onset intervals \(IOIs\) cannot be zero or negative"):
        seq.iois = [1, 2, 3, -1, 4, 5]


def test_onsets_property(rng):
    seq = thebeat.core.Sequence.generate_random_normal(10, 500, 25, rng=rng)
    assert len(seq.onsets) == 10

    onsets = seq.onsets.copy()
    seq.onsets[0] = 42
    assert np.all(seq.onsets == onsets)

    seq.onsets = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    assert isinstance(seq.onsets, np.ndarray)
    assert len(seq.onsets) == 9
    assert len(seq.iois) == 8
    assert np.all(seq.iois == [1, 1, 2, 3, 5, 8, 13, 21])

    with pytest.raises(ValueError, match="Onsets are not ordered strictly monotonically"):
        seq.onsets = [1, 1, 2, 3, 5]

    with pytest.raises(ValueError, match="Onsets are not ordered strictly monotonically"):
        seq.onsets = [1, -1, 1, -1, 1]


def test_exception():
    seq = thebeat.core.Sequence.generate_isochronous(n=10, ioi=500)
    seq.change_tempo(0.5)
    with pytest.raises(ValueError):
        seq.change_tempo(-1)

    with pytest.raises(ValueError):
        thebeat.core.Sequence.from_onsets([20, 20, 20])


def test_onset_not_zero():
    seq = thebeat.core.Sequence.from_onsets([20, 50, 100])
    assert np.all(seq.onsets == [20, 50, 100])
    assert np.all(seq.iois == [30, 50])

    with pytest.raises(ValueError):
        thebeat.core.Sequence([50, 50, 50], end_with_interval=True, first_onset=50)


def test_multiplication():
    seq = thebeat.core.Sequence([500, 500, 500])
    with pytest.raises(ValueError):
        _ = seq * 10
    seq = thebeat.core.Sequence([500, 500, 500], end_with_interval=True)
    seq *= 10
    assert len(seq.iois) == 30


def test_plot():
    # simple case
    seq = thebeat.core.Sequence([500, 1000, 200])
    fig, ax = seq.plot_sequence(suppress_display=True)
    assert fig, ax

    # plot onto existing Axes
    fig, axs = plt.subplots(1, 2)
    seq.plot_sequence(ax=axs[0])
    assert fig, axs

