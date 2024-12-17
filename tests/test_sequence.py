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

import matplotlib.pyplot as plt
import numpy as np
import pytest

import thebeat.core


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
    seq = thebeat.core.Sequence.generate_isochronous(n_events=10, ioi=500)
    seq.change_tempo(0.5)
    with pytest.raises(ValueError):
        seq.change_tempo(-1)

    with pytest.raises(ValueError):
        thebeat.core.Sequence.from_onsets([20, 20, 20])


def test_onset_not_zero():
    seq = thebeat.core.Sequence.from_onsets([20, 50, 100])
    assert np.all(seq.onsets == [20, 50, 100])
    assert np.all(seq.iois == [30, 50])


def test_multiplication():
    seq = thebeat.core.Sequence([500, 500, 500])
    with pytest.raises(ValueError):
        _ = seq * 10
    seq = thebeat.core.Sequence([500, 500, 500], end_with_interval=True)
    seq *= 10
    assert len(seq.iois) == 30


@pytest.custom_mpl_image_compare
def test_plot_sequence():
    # simple case
    seq = thebeat.core.Sequence([500, 1000, 200])
    fig, ax = seq.plot_sequence()
    return fig


@pytest.custom_mpl_image_compare
def test_plot_sequence_on_ax():
    # plot onto existing Axes
    seq = thebeat.core.Sequence([500, 1000, 200], name="TestSequence")
    fig, axs = plt.subplots(2, 1)
    seq.plot_sequence(ax=axs[0])
    return fig


@pytest.custom_mpl_image_compare
def test_plot_sequence_title():
    # simple case
    seq = thebeat.core.Sequence([500, 1000, 200], name="TestSequence")
    fig, ax = seq.plot_sequence(title="This title, not the name")
    return fig


def test_concat():
    seq1 = thebeat.Sequence.generate_isochronous(5, 500, True)
    seq2 = thebeat.Sequence.generate_isochronous(5, 1000)
    s1 = seq1 + seq2
    s2 = thebeat.utils.concatenate_sequences([seq1, seq2])
    assert s1, s2


def test_concat_silence_endswithinterval():
    seq = thebeat.Sequence.generate_isochronous(5, 500, True)
    assert seq.onsets[0] == 0
    assert seq.onsets[-1] == 2000
    assert seq.iois[-1] == 500
    assert len(seq.iois) == 5
    assert seq.end_with_interval

    s1 = seq + 1000
    assert s1.onsets[0] == 0
    assert s1.onsets[-1] == 2000
    assert s1.iois[-1] == 1500
    assert len(s1.iois) == 5
    assert s1.end_with_interval

    s2 = 1000 + seq
    assert s2.onsets[0] == 1000
    assert s2._first_onset == 1000
    assert s2.onsets[-1] == 3000
    assert s2.iois[-1] == 500
    assert len(s2.iois) == 5
    assert s2.end_with_interval

    s3 = s2 + 1000
    assert s3.onsets[0] == 1000
    assert s3.onsets[-1] == 3000
    assert s3.iois[-1] == 1500
    assert len(s3.iois) == 5
    assert s3.end_with_interval


def test_concat_silence_endswithevent():
    seq = thebeat.Sequence.generate_isochronous(5, 500, False)
    assert seq.onsets[0] == 0
    assert seq.onsets[-1] == 2000
    assert seq.iois[-1] == 500
    assert len(seq.iois) == 4
    assert not seq.end_with_interval

    s1 = seq + 1000
    assert s1.onsets[0] == 0
    assert s1.onsets[-1] == 2000
    assert s1.iois[-2] == 500
    assert s1.iois[-1] == 1000
    assert len(s1.iois) == 5
    assert s1.end_with_interval

    s2 = 1000 + seq
    assert s2.onsets[0] == 1000
    assert s2._first_onset == 1000
    assert s2.onsets[-1] == 3000
    assert s2.iois[-1] == 500
    assert len(s2.iois) == 4
    assert not s2.end_with_interval

    s3 = s2 + 1000
    assert s3.onsets[0] == 1000
    assert s3.onsets[-1] == 3000
    assert s1.iois[-2] == 500
    assert s3.iois[-1] == 1000
    assert len(s3.iois) == 5
    assert s3.end_with_interval

    chained = seq + 1000 + s1 + 2000
    assert len(chained.onsets) == 10


def test_concat_exceptions():
    seq_endswithinterval = thebeat.Sequence.generate_isochronous(5, 500, True)
    seq_endswithevent = thebeat.Sequence.generate_isochronous(5, 500, False)

    # E.g. a string is not allowed
    with pytest.raises(TypeError):
        _ = seq_endswithinterval + '1000'

    with pytest.raises(TypeError):
        _ = '1000' + seq_endswithinterval

    with pytest.raises(TypeError):
        _ = seq_endswithevent + '1000'

    with pytest.raises(TypeError):
        _ = '1000' + seq_endswithevent

    class Dummy:
        def __add__(self, other):
            return '__add__'

        def __radd__(self, other):
            return '__radd__'

    assert Dummy() + seq_endswithevent == '__add__'
    assert seq_endswithevent + Dummy() == '__radd__'

    with pytest.raises(ValueError,
                       match="When concatenating sequences the sequence on the left-hand side must end with an interval."):
        _ = thebeat.Sequence.generate_isochronous(5, 500, False) + seq_endswithevent

    with pytest.raises(ValueError,
                       match="When concatenating sequences the sequence on the left-hand side must end with an interval."):
        _ = thebeat.Sequence.generate_isochronous(5, 500, False) + seq_endswithinterval

    with pytest.raises(ValueError):
        _ = seq_endswithinterval + 0

    with pytest.raises(ValueError):
        _ = seq_endswithinterval + -1


def test_merge():
    seq1 = thebeat.Sequence.from_onsets([0, 500, 1000])
    seq2 = thebeat.Sequence.from_onsets([250, 750, 1250])
    seq_merged = seq1.merge(seq2)
    seq_merged2 = thebeat.utils.merge_sequences([seq1, seq2])

    assert len(seq1.onsets) + len(seq2.onsets) == len(seq_merged.onsets)
    assert len(seq_merged2.onsets) == len(seq_merged.onsets)


def test_copy():
    seq = thebeat.Sequence([500, 500], name='test')
    seq2 = seq.copy()
    seq.name = "test2"
    assert seq2.name == 'test'
    assert seq.name == 'test2'


def test_frombinarystring():
    pattern = '10101100'  # output: 2 2 1 3
    seq = thebeat.Sequence.from_binary_string(pattern, 250)
    assert np.all(seq.iois == [500, 500, 250, 750])


def test_quantization():
    to = 125
    iois = [523, 111, 798, 512]

    seq_interval = thebeat.Sequence(iois, end_with_interval=True)
    seq_noninterval = thebeat.Sequence(iois, end_with_interval=False)

    seq_interval.quantize_iois(to)
    seq_noninterval.quantize_iois(to)

    assert np.all(seq_interval.iois == [500, 125, 750, 500])
    assert np.all(seq_noninterval.iois == [500, 125, 750, 500])
