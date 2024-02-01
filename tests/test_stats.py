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
from thebeat.stats import (
    acf_plot,
    acf_values,
    ccf_plot,
    ccf_values,
    fft_plot,
    fft_values,
    get_npvi,
    get_phase_differences,
    get_rhythmic_entropy,
    get_ugof_isochronous,
    ks_test
)


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
        print(get_rhythmic_entropy(seq, smallest_unit=500))

    seq.quantize_iois(500)

    assert get_rhythmic_entropy(seq, 500) == 0.0


def test_fft_values():
    seq = Sequence([500, 500, 500, 500])
    x, y = fft_values(seq, unit_size=1000, x_min=None, x_max=None)
    assert np.argmax(y) == 0

    x, y = fft_values(seq, unit_size=1000, x_min=1, x_max=3)
    assert x[np.argmax(y)] == 2

    seq = Sequence([500, 501, 500, 500])
    x, y = fft_values(seq, unit_size=1000)

    seq = Sequence([501.1, 500.3, 500, 500])
    x, y = fft_values(seq, unit_size=1000)

    seq = Sequence.from_onsets([-10, 40, 100, 1000])
    x, y = fft_values(seq, unit_size=1000)


@pytest.mark.mpl_image_compare
def test_fft_plot():
    seq = Sequence([500, 500, 500, 500])
    fig, ax = fft_plot(seq, unit_size=1000, x_min=None, x_max=10, suppress_display=True)
    return fig


def test_ccf_values():
    seq = Sequence([500, 250, 500, 250], end_with_interval=True)
    seq2 = Sequence([250, 500, 500, 250, 500], end_with_interval=True)
    assert ccf_values(seq, seq2, 1)[0] == 1
    assert np.argmax(ccf_values(seq, seq2, 1)) == 250

    seq = Sequence([500, 250, 250])
    seq2 = Sequence([250, 500, 500, 250, 500], end_with_interval=True)
    assert ccf_values(seq, seq2, 1)[0] == 0

    seq = Sequence([500, 250, 250])
    seq2 = Sequence([250, 500, 500, 250, 500])

    assert ccf_values(seq, seq2, 1)[0] != 0

    seq = Sequence([500, 250, 250])
    seq2 = Sequence([250, 500, 500, 250, 500], end_with_interval=True)

    assert ccf_values(seq, seq2, 50)[0] == 0
    assert ccf_values(seq, seq2, 1, 10, 2)[0] != 0


def test_acf_values():
    seq = Sequence([500, 250, 500, 250], end_with_interval=True)
    assert acf_values(seq, 1)[0] != 0

    seq = Sequence([500, 250, 250])
    assert acf_values(seq, 1)[0] != 0

    seq = Sequence([500, 250, 250])

    assert acf_values(seq, 50)[0] != 0
    assert acf_values(seq, 1, 10, 2)[0] != 0


@pytest.mark.parametrize("end_with_interval", [False, True])
def test_phase_differences(end_with_interval):
    ref_sequence = Sequence.from_onsets([500, 1500, 2000, 3000])
    ref_sequence.end_with_interval = end_with_interval
    test_sequence = Sequence.from_onsets(
        [-100, 100, 500, 600, 1000, 1400, 1500, 1600, 2000, 2500, 2800, 3000, 3500, 4000]
    )

    # Containing
    phase_diffs = get_phase_differences(
        test_sequence, ref_sequence, reference_ioi="containing", unit="fraction"
    )
    expected_phase_diffs = [
        np.nan,
        np.nan,
        0.0,
        0.1,
        0.5,
        0.9,
        0,
        0.2,
        0,
        0.5,
        0.8,
        np.nan,
        np.nan,
        np.nan,
    ]
    assert phase_diffs == pytest.approx(expected_phase_diffs, nan_ok=True)

    phase_diffs = get_phase_differences(
        test_sequence, ref_sequence, reference_ioi="containing", unit="degrees"
    )
    expected_phase_diffs = [
        np.nan,
        np.nan,
        0,
        36,
        180,
        324,
        0,
        72,
        0,
        180,
        288,
        np.nan,
        np.nan,
        np.nan,
    ]
    assert phase_diffs == pytest.approx(expected_phase_diffs, nan_ok=True)

    phase_diffs = get_phase_differences(
        test_sequence, ref_sequence, reference_ioi="containing", unit="radians"
    )
    expected_phase_diffs = [
        np.nan,
        np.nan,
        0,
        0.2 * np.pi,
        np.pi,
        1.8 * np.pi,
        0,
        0.4 * np.pi,
        0,
        np.pi,
        1.6 * np.pi,
        np.nan,
        np.nan,
        np.nan,
    ]
    assert phase_diffs == pytest.approx(expected_phase_diffs, nan_ok=True)

    with pytest.raises(
        ValueError, match="reference_ioi must be either 'containing' or 'preceding'"
    ):
        get_phase_differences(test_sequence, ref_sequence, reference_ioi="invalid", unit="fraction")

    with pytest.raises(ValueError, match="unit must be either 'degrees', 'radians' or 'fraction'"):
        get_phase_differences(
            test_sequence, ref_sequence, reference_ioi="containing", unit="invalid"
        )

    with pytest.raises(
        ValueError, match="window_size cannot be used with reference_ioi='containing'"
    ):
        get_phase_differences(
            test_sequence, ref_sequence, reference_ioi="containing", window_size=1, unit="fraction"
        )

    # Preceding
    phase_diffs = get_phase_differences(
        test_sequence, ref_sequence, reference_ioi="preceding", unit="fraction", modulo=False
    )
    expected_phase_diffs = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0,
        0.1,
        0,
        1,
        1.6,
        0,
        0.5,
        1,
    ]
    assert phase_diffs == pytest.approx(expected_phase_diffs, nan_ok=True)
    phase_diffs = get_phase_differences(
        test_sequence,
        ref_sequence,
        reference_ioi="preceding",
        window_size=1,
        unit="fraction",
        modulo=False,
    )
    assert phase_diffs == pytest.approx(expected_phase_diffs, nan_ok=True)

    phase_diffs = get_phase_differences(
        test_sequence, ref_sequence, reference_ioi="preceding", unit="fraction", modulo=True
    )
    expected_phase_diffs = np.array(
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0.1, 0, 0, 0.6, 0, 0.5, 0]
    )
    assert phase_diffs == pytest.approx(expected_phase_diffs, nan_ok=True)

    phase_diffs = get_phase_differences(
        test_sequence,
        ref_sequence,
        reference_ioi="preceding",
        window_size=2,
        unit="fraction",
        modulo=False,
    )
    expected_phase_diffs = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0,
        2 / 3,
        16 / 15,
        0,
        2 / 3,
        4 / 3,
    ]
    assert phase_diffs == pytest.approx(expected_phase_diffs, nan_ok=True)

    phase_diffs = get_phase_differences(
        test_sequence,
        ref_sequence,
        reference_ioi="preceding",
        window_size=3,
        unit="fraction",
        modulo=False,
    )
    expected_phase_diffs = [
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        0,
        3 / 5,
        6 / 5,
    ]
    assert phase_diffs == pytest.approx(expected_phase_diffs, nan_ok=True)

    window_size = len(ref_sequence.iois) + 1
    phase_diffs = get_phase_differences(
        test_sequence,
        ref_sequence,
        reference_ioi="preceding",
        window_size=window_size,
        unit="fraction",
        modulo=False,
    )
    assert np.all(np.isnan(phase_diffs))

    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        get_phase_differences(
            test_sequence,
            ref_sequence,
            reference_ioi="preceding",
            window_size=0,
            unit="fraction",
            modulo=False,
        )

    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        get_phase_differences(
            test_sequence,
            ref_sequence,
            reference_ioi="preceding",
            window_size=-1,
            unit="fraction",
            modulo=False,
        )

    # Single values and lists
    assert get_phase_differences(
        2100, ref_sequence, reference_ioi="containing", unit="fraction"
    ) == pytest.approx(0.1)
    assert get_phase_differences(
        [2200, 2300], ref_sequence, reference_ioi="containing", unit="fraction"
    ) == pytest.approx([0.2, 0.3])

    assert get_phase_differences(
        2100, ref_sequence, reference_ioi="preceding", unit="fraction"
    ) == pytest.approx(0.2)
    assert get_phase_differences(
        [2200, 2300], ref_sequence, reference_ioi="preceding", unit="fraction"
    ) == pytest.approx([0.4, 0.6])


def test_phase_differences_moving_average():
    ref_sequence = Sequence([1, 2, 4, 1, 2])
    test_events = [1.5, 3.5, 7.5, 8.5, 10.5]

    phase_diffs = get_phase_differences(
        test_events, ref_sequence, reference_ioi="preceding", window_size=1, unit="fraction"
    )
    assert phase_diffs == pytest.approx([1 / 2, 1 / 4, 1 / 8, 1 / 2, 1 / 4], nan_ok=True)

    phase_diffs = get_phase_differences(
        test_events, ref_sequence, reference_ioi="preceding", window_size=2, unit="fraction"
    )
    assert phase_diffs == pytest.approx([np.nan, 1 / 3, 1 / 6, 1 / 5, 1 / 3], nan_ok=True)

    phase_diffs = get_phase_differences(
        test_events, ref_sequence, reference_ioi="preceding", window_size=3, unit="fraction"
    )
    assert phase_diffs == pytest.approx([np.nan, np.nan, 3 / 14, 3 / 14, 3 / 14], nan_ok=True)

    phase_diffs = get_phase_differences(
        test_events, ref_sequence, reference_ioi="preceding", window_size=4, unit="fraction"
    )
    assert phase_diffs == pytest.approx([np.nan, np.nan, np.nan, 1 / 4, 2 / 9], nan_ok=True)

    phase_diffs = get_phase_differences(
        test_events, ref_sequence, reference_ioi="preceding", window_size=5, unit="fraction"
    )
    assert phase_diffs == pytest.approx([np.nan, np.nan, np.nan, np.nan, 1 / 4], nan_ok=True)

    phase_diffs = get_phase_differences(
        test_events, ref_sequence, reference_ioi="preceding", window_size=6, unit="fraction"
    )
    assert phase_diffs == pytest.approx([np.nan, np.nan, np.nan, np.nan, np.nan], nan_ok=True)
