# Copyright (C) 2022-2023  Jelle van der Werff
#
# This file is part of thebeat.
#
# thebeat is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# thebeat is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with thebeat.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import pytest

from thebeat.core import Sequence
from thebeat.music import Rhythm
from thebeat.stats import acf_plot, ccf_plot, fft_plot, get_npvi, get_rhythmic_entropy, get_ugof_isochronous, ks_test


def test_ugof():
    seq = Sequence.generate_isochronous(n_events=10, ioi=500)
    assert get_ugof_isochronous(seq, 500, "median") == 0.0


def test_ks():
    rng = np.random.default_rng(seed=123)
    seq = Sequence.generate_random_uniform(n_events=10, a=400, b=600, rng=rng)
    assert ks_test(seq)[0] == 0.2724021511351798


def test_npvi():
    seq = Sequence.generate_isochronous(n_events=10, ioi=500)
    assert get_npvi(seq) == 0.0


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
    fig, ax = acf_plot(
        seq,
        1 / 1000,
        max_lag=1000 / 1000,
        smoothing_window=50 / 1000,
        smoothing_sd=10 / 1000,
        suppress_display=True,
    )
    return fig


@pytest.mark.mpl_image_compare
def test_acf_image_milliseconds():
    rng = np.random.default_rng(seed=123)
    seq = Sequence.generate_random_normal(100, 500, 25, rng=rng)
    fig, ax = acf_plot(
        seq, 1, max_lag=1000, smoothing_window=50, smoothing_sd=10, suppress_display=True
    )
    return fig


def test_entropy():
    seq = Sequence([500, 502, 500, 500])

    with pytest.raises(ValueError):
        print(get_rhythmic_entropy(seq, resolution=500))

    seq = seq.quantize_iois(500)

    assert get_rhythmic_entropy(seq, 500) == 0.0


def test_fft():
    rng = np.random.default_rng(seed=123)
    r = Rhythm.generate_random_rhythm(1, 500, allowed_note_values=[8, 16], rng=rng)
    seq = r.to_sequence()
    _, ax = fft_plot(seq, 1000, suppress_display=True)
    x_data, y_data = ax.lines[0].get_data()
    x_data, y_data = x_data[1:], y_data[1:]
    max_y_index = np.argmax(y_data)
    max_x = x_data[max_y_index]

    assert 1000 / max_x == 125

    seq = Sequence([500, 250, 250, 250])
    _, ax = fft_plot(seq, 1000, suppress_display=True)
    x_data, y_data = ax.lines[0].get_data()
    x_data, y_data = x_data[1:], y_data[1:]
    max_y_index = np.argmax(y_data)
    max_x = x_data[max_y_index]
    assert 1000 / max_x == 250
