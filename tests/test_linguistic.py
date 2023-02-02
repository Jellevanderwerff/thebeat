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

import thebeat.linguistic


def test_stress():
    seq = thebeat.linguistic.generate_stress_timed_sequence(10)
    assert len(seq.onsets) == 10


def test_mora():
    seq = thebeat.linguistic.generate_moraic_sequence(10, foot_ioi=600)
    assert seq.duration == 6000
