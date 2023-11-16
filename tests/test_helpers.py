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
import scipy

import thebeat


def test_sequence_to_binary():
    seq = thebeat.core.Sequence([500, 1000, 250, 500, 250], end_with_interval=False)
    binary = thebeat.helpers.sequence_to_binary(seq, resolution=1)
    assert len(binary) == 2501
    assert list(np.flatnonzero(binary)) == [0, 500, 1500, 1750, 2250, 2500]

    seq = thebeat.core.Sequence([500, 1000, 250, 500, 250], end_with_interval=True)
    binary = thebeat.helpers.sequence_to_binary(seq, resolution=1)
    assert len(binary) == 2500
    assert list(np.flatnonzero(binary)) == [0, 500, 1500, 1750, 2250]

    seq = thebeat.core.Sequence([500, 1000, 250, 500, 250], first_onset=1000, end_with_interval=False)
    binary = thebeat.helpers.sequence_to_binary(seq, resolution=1)
    assert len(binary) == 3501
    assert list(np.flatnonzero(binary)) == [1000, 1500, 2500, 2750, 3250, 3500]

    seq = thebeat.core.Sequence([500, 1000, 250, 500, 250], first_onset=1000, end_with_interval=True)
    binary = thebeat.helpers.sequence_to_binary(seq, resolution=1)
    assert len(binary) == 3500
    assert list(np.flatnonzero(binary)) == [1000, 1500, 2500, 2750, 3250]


def test_sequence_to_binary_resolution():
    seq = thebeat.core.Sequence([500, 1000, 250, 500, 250], end_with_interval=False)
    binary = thebeat.helpers.sequence_to_binary(seq, resolution=1)
    assert len(binary) == 2501
    assert list(np.flatnonzero(binary)) == [0, 500, 1500, 1750, 2250, 2500]

    binary = thebeat.helpers.sequence_to_binary(seq, resolution=0.5)
    assert len(binary) == 5001
    assert list(np.flatnonzero(binary)) == [0, 1000, 3000, 3500, 4500, 5000]

    binary = thebeat.helpers.sequence_to_binary(seq, resolution=2)
    assert len(binary) == 1251
    assert list(np.flatnonzero(binary)) == [0, 250, 750, 875, 1125, 1250]

    binary = thebeat.helpers.sequence_to_binary(seq, resolution=250)
    assert len(binary) == 11
    assert list(np.flatnonzero(binary)) == [0, 2, 6, 7, 9, 10]
    assert list(binary.astype(int)) == [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1]

    binary = thebeat.helpers.sequence_to_binary(seq, resolution=3)
    assert len(binary) == 834
    assert list(np.flatnonzero(binary)) == [0, 166, 500, 583, 750, 833]


def test_rhythm_to_binary():
    # Should raise error because there are 1/8th notes but the provides smallest note value is a 1/4th note
    with pytest.raises(ValueError):
        rhythm = thebeat.music.Rhythm.from_note_values([4, 8, 8, 4, 4])
        print(thebeat.helpers.rhythm_to_binary(rhythm, smallest_note_value=4))

    # Should not raise error
    binary = thebeat.helpers.rhythm_to_binary(rhythm, smallest_note_value=8)
    assert np.all(binary == [1., 0., 1., 1., 1., 0., 1., 0.])


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32, np.float64])
def test_dtypes(tmp_path, dtype):
    stim = thebeat.SoundStimulus.generate()
    seq = thebeat.Sequence.generate_isochronous(n_events=10, ioi=500)
    trial = thebeat.SoundSequence(stim, seq)

    trial.write_wav(tmp_path / 'test.wav', metronome=True, dtype=dtype)

    # test datatype
    _, samples = scipy.io.wavfile.read(tmp_path / 'test.wav')
    assert samples.dtype == dtype
