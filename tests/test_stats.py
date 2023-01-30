from thebeat.stats import *
from thebeat.core import Sequence
import numpy as np
import pytest
import matplotlib.pyplot as plt


def test_ugof():
    seq = Sequence.generate_isochronous(n_events=10, ioi=500)
    assert get_ugof(seq, 500, 'median') == 0.0


def test_ks():
    rng = np.random.default_rng(seed=123)
    seq = Sequence.generate_random_uniform(n_events=10, a=400, b=600, rng=rng)
    assert ks_test(seq)[0] == 0.2724021511351798


def test_npvi():
    seq = Sequence.generate_isochronous(n_events=10, ioi=500)
    assert get_npvi(seq) == 0.0


def test_ccf_values():
    seq = Sequence([500, 500, 500, 500])
    seq2 = Sequence([250, 500, 500, 500])

    values = ccf_values(seq, seq2, 1)

    # normalize
    values = values / np.max(values)

    # Check whether the correlation is 1 at lag 250 (because there's 250 diff between the seqs)
    assert values[250] == 1.0


@pytest.mark.mpl_image_compare
def test_ccf_plot():
    seq = Sequence([500, 500, 500, 500])
    seq2 = Sequence([250, 500, 500, 500])

    fig, ax = ccf_plot(seq, seq2, 1, suppress_display=True)

    return fig


@pytest.mark.mpl_image_compare
def test_acf_image_seconds():
    rng = np.random.default_rng(seed=123)
    seq = Sequence.generate_random_normal(100, 500 / 1000, 25 / 1000, rng=rng)
    fig, ax = acf_plot(seq, 1 / 1000, max_lag=1000 / 1000, smoothing_window=50 / 1000, smoothing_sd=10 / 1000)
    return fig


@pytest.mark.mpl_image_compare
def test_acf_image_milliseconds():
    rng = np.random.default_rng(seed=123)
    seq = Sequence.generate_random_normal(100, 500, 25, rng=rng)
    fig, ax = acf_plot(seq, 1, max_lag=1000, smoothing_window=50, smoothing_sd=10)
    return fig
