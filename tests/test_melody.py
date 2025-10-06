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

import thebeat.music


def test_melody(tmp_path):
    rng = np.random.default_rng(seed=123)
    r = thebeat.music.Rhythm.from_note_values([1/4, 1/4, 1/4, 1/4])
    m = thebeat.music.Melody(r, 'CEGC', octave=3)
    assert list(m.note_values) == [1/4, 1/4, 1/4, 1/4]

    mel = thebeat.music.Melody.generate_random_melody(n_bars=2, key='G', octave=5, rng=rng)
    samples, fs = mel.synthesize_and_return(n_channels=2, metronome=True)
    assert isinstance(samples, np.ndarray)
    assert fs

    mel.synthesize_and_write(tmp_path / 'test_melody.wav', n_channels=2, metronome=True)


@pytest.custom_mpl_image_compare(tolerance=3)
def test_melody_plot():
    rng = np.random.default_rng(seed=123)
    mel = thebeat.music.Melody.generate_random_melody(n_bars=2, key='G', octave=4, rng=rng)
    fig, ax = mel.plot_melody()
    return fig


# Not sure why, but these two tests come out slightly differently on other platforms
# Hence, put tolerance=4, until one day LilyPond would be more consistent across OSes
@pytest.custom_mpl_image_compare(tolerance=4)
def test_melody_note_ties():
    r = thebeat.music.Rhythm.from_note_values([1/4, 2, 3/4], beat_ms=500, time_signature=(4, 4))
    m = thebeat.music.Melody(r, 'CDE')
    fig, ax = m.plot_melody()
    return fig


@pytest.custom_mpl_image_compare(tolerance=4)
def test_melody_note_ties2():
    r = thebeat.music.Rhythm.from_note_values([1/16, 1/16, 1/16, 5/16, 5/16, 1/16, 1/16, 1/16], beat_ms=500, time_signature=(4, 4))
    m = thebeat.music.Melody(r, 'CDEFGABC')
    fig, ax = m.plot_melody()
    return fig


def test_melody_copy():
    m = thebeat.music.Melody(thebeat.music.Rhythm([500, 500, 500, 500]), 'CEGC', name="test")
    m2 = m.copy()
    m.name = "test2"
    assert m2.name == 'test'
    assert m.name == 'test2'
