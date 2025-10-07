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
