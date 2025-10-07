# Copyright (C) 2022-2025  Jelle van der Werff
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

from fractions import Fraction

import numpy as np
import pytest

import thebeat


def test_concatenate_rhythms():
    r1 = thebeat.music.Rhythm.from_note_values([1/4, 1/4, 1/4, 1/4])
    r2 = thebeat.music.Rhythm.from_note_values([1/3, 1/3, 1/3])
    r3 = thebeat.music.Rhythm([500, 1000, 500], (4, 4), 500)

    rhythm = thebeat.utils.concatenate_rhythms([r1, r2, r3], name='concatenated')
    assert np.all(rhythm.iois == [500, 500, 500, 500, 2000 / 3, 2000 / 3, 2000 / 3, 500, 1000, 500])
    assert rhythm.name == 'concatenated'

    with pytest.raises(ValueError, match="At least one Rhythm object is required to concatenate"):
        thebeat.utils.concatenate_rhythms([])

    s = thebeat.core.Sequence([500, 1000, 500])
    with pytest.raises(TypeError, match="Please pass only Rhythm objects"):
        thebeat.utils.concatenate_rhythms([r1, s])

    r4 = thebeat.music.Rhythm.from_note_values([1/4, 1/4], time_signature=(2, 4))
    with pytest.raises(ValueError, match="Provided rhythms should have the same time signatures"):
        thebeat.utils.concatenate_rhythms([r1, r4])

    r5 = thebeat.music.Rhythm.from_note_values([1/4, 1/4, 1/4, 1/4], beat_ms=250)
    with pytest.raises(ValueError, match=r"Provided rhythms should have same tempo \(beat_ms\)"):
        thebeat.utils.concatenate_rhythms([r1, r5])


def test_sequence_to_binary():
    seq = thebeat.core.Sequence([500, 1000, 250, 500, 250], end_with_interval=False)
    binary = thebeat.utils.sequence_to_binary(seq, resolution=1)
    assert len(binary) == 2501
    assert list(np.flatnonzero(binary)) == [0, 500, 1500, 1750, 2250, 2500]

    seq = thebeat.core.Sequence([500, 1000, 250, 500, 250], end_with_interval=True)
    binary = thebeat.utils.sequence_to_binary(seq, resolution=1)
    assert len(binary) == 2500
    assert list(np.flatnonzero(binary)) == [0, 500, 1500, 1750, 2250]

    seq = thebeat.core.Sequence([500, 1000, 250, 500, 250], first_onset=1000, end_with_interval=False)
    binary = thebeat.utils.sequence_to_binary(seq, resolution=1)
    assert len(binary) == 3501
    assert list(np.flatnonzero(binary)) == [1000, 1500, 2500, 2750, 3250, 3500]

    seq = thebeat.core.Sequence([500, 1000, 250, 500, 250], first_onset=1000, end_with_interval=True)
    binary = thebeat.utils.sequence_to_binary(seq, resolution=1)
    assert len(binary) == 3500
    assert list(np.flatnonzero(binary)) == [1000, 1500, 2500, 2750, 3250]


def test_sequence_to_binary_resolution():
    seq = thebeat.core.Sequence([500, 1000, 250, 500, 250], end_with_interval=False)
    binary = thebeat.utils.sequence_to_binary(seq, resolution=1)
    assert len(binary) == 2501
    assert list(np.flatnonzero(binary)) == [0, 500, 1500, 1750, 2250, 2500]

    binary = thebeat.utils.sequence_to_binary(seq, resolution=0.5)
    assert len(binary) == 5001
    assert list(np.flatnonzero(binary)) == [0, 1000, 3000, 3500, 4500, 5000]

    binary = thebeat.utils.sequence_to_binary(seq, resolution=2)
    assert len(binary) == 1251
    assert list(np.flatnonzero(binary)) == [0, 250, 750, 875, 1125, 1250]

    binary = thebeat.utils.sequence_to_binary(seq, resolution=250)
    assert len(binary) == 11
    assert list(np.flatnonzero(binary)) == [0, 2, 6, 7, 9, 10]
    assert list(binary.astype(int)) == [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1]

    binary = thebeat.utils.sequence_to_binary(seq, resolution=3)
    assert len(binary) == 834
    assert list(np.flatnonzero(binary)) == [0, 166, 500, 583, 750, 833]


def test_rhythm_to_binary():
    # Should raise error because there are 1/8th notes but the provides smallest note value is a 1/4th note
    with pytest.raises(ValueError):
        rhythm = thebeat.music.Rhythm.from_note_values([1/4, 1/8, 1/8, 1/4, 1/4])
        print(thebeat.utils.rhythm_to_binary(rhythm, smallest_note_value=Fraction(1, 4)))

    # Should not raise error
    binary = thebeat.utils.rhythm_to_binary(rhythm, smallest_note_value=Fraction(1, 8))
    assert np.all(binary == [1., 0., 1., 1., 1., 0., 1., 0.])
