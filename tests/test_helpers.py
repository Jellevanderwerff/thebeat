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

import thebeat


def test_rhythm_to_binary():
    # Should raise error because there are 1/8th notes but the provides smallest note value is a 1/4th note
    with pytest.raises(ValueError):
        rhythm = thebeat.music.Rhythm.from_note_values([4, 8, 8, 4, 4])
        print(thebeat.helpers.rhythm_to_binary(rhythm, smallest_note_value=4))

    # Should not raise error
    binary = thebeat.helpers.rhythm_to_binary(rhythm, smallest_note_value=8)
    assert np.all(binary == [1., 0., 1., 1., 1., 0., 1., 0.])
