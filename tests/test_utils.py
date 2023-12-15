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

import pytest

import thebeat.utils
from thebeat import Sequence


def test_phasedifferences():
    seq = Sequence([499, 501, 505, 501])

    with pytest.warns(UserWarning, match="The first onset of the test sequence was at t=0"):
        diffs = list(thebeat.stats.get_phase_differences(seq, 500))

    assert diffs == [359.2814371257485, 0.0, 3.5643564356435644, 4.2772277227722775]
