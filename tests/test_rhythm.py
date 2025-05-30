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


def test_rhythm():
    # generate random rhythm
    rrhythm = thebeat.music.Rhythm.generate_random_rhythm(allowed_note_values=[4, 8])
    assert rrhythm

    # combine two rhythms
    rhythm1 = thebeat.music.Rhythm.generate_random_rhythm(allowed_note_values=[4, 8])
    rhythm2 = thebeat.music.Rhythm.generate_random_rhythm(allowed_note_values=[4, 8])
    combined_rhythm = rhythm1 + rhythm2
    assert sum(combined_rhythm.iois) == sum(rhythm1.iois) + sum(rhythm2.iois)

    rhythm = thebeat.music.Rhythm.from_note_values([4, 4, 4, 4])
    assert np.all(rhythm.iois == [500, 500, 500, 500])

    rhythm = thebeat.music.Rhythm([500, 1000, 500], (4, 4), 500)
    assert rhythm.beat_ms == 500

    with pytest.raises(ValueError):
        rhythm = thebeat.music.Rhythm([250, 250, 500, 250, 250, 250], (4, 4), 500)

    rhythm = thebeat.music.Rhythm([500, 500, 500, 500])
    rhythm *= 4
    assert rhythm.n_bars == 4
    assert len(rhythm.iois) == 16


def test_rhythm_from_fractions():
    r1 = thebeat.music.Rhythm.from_fractions([1/4, 3/4, 1/8, 3/8, 1/2], beat_ms=500, time_signature=(4, 4))
    assert np.all(r1.iois == [500, 1500, 250, 750, 1000])

    r2 = thebeat.music.Rhythm.from_fractions([1/2, 3/2, 2/8, 6/8, 2/2], beat_ms=500, time_signature=(4, 2))
    assert np.all(r2.iois == [500, 1500, 250, 750, 1000])

    r3 = thebeat.music.Rhythm.from_fractions([1/4, 1/2, 1/8, 3/8, 1/4], beat_ms=500, time_signature=(3, 4))
    assert np.all(r3.iois == [500, 1000, 250, 750, 500])

    r4 = thebeat.music.Rhythm.from_fractions([1/8, 1/4, 1/16, 3/16, 1/8], beat_ms=500, time_signature=(3, 8))
    assert np.all(r4.iois == [500, 1000, 250, 750, 500])


@pytest.custom_mpl_image_compare(tolerance=2)
def test_rhythm_plot():
    rhythm = thebeat.music.Rhythm([500, 500, 500, 500])
    fig, ax = rhythm.plot_rhythm(suppress_display=True)
    return fig


def test_copy():
    r = thebeat.music.Rhythm([500, 500, 500, 500], name="test")
    r2 = r.copy()
    r.name = "test2"
    assert r2.name == 'test'
    assert r.name == 'test2'
