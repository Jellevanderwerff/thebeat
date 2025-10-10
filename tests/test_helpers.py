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

import numpy as np
import pytest
import scipy

import thebeat


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.float32, np.float64])
def test_dtypes(tmp_path, dtype):
    stim = thebeat.SoundStimulus.generate()
    seq = thebeat.Sequence.generate_isochronous(n_events=10, ioi=500)
    trial = thebeat.SoundSequence(stim, seq)

    trial.write_wav(tmp_path / 'test.wav', metronome=True, dtype=dtype)

    # test datatype
    _, samples = scipy.io.wavfile.read(tmp_path / 'test.wav')
    assert samples.dtype == dtype
