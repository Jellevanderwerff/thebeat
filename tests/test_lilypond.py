# Copyright (C) 2023  Jelle van der Werff
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

import re
import shutil
import sys

import numpy as np
import pytest

import thebeat


def test_lilypond_unavailable(monkeypatch):
    _shutil_which = shutil.which
    monkeypatch.setattr(shutil, 'which', lambda x: None if x == 'lilypond' else _shutil_which(x))
    monkeypatch.setitem(sys.modules, 'lilypond', None)

    with pytest.raises(ImportError, match="This function or method requires lilypond for plotting notes"):
        rng = np.random.default_rng(42)
        melody = thebeat.music.Melody.generate_random_melody(n_bars=2, key='G', octave=4, rng=rng)
        _ = melody.plot_melody(suppress_display=True)


@pytest.mark.mpl_image_compare
def test_lilypond_package(monkeypatch):
    lilypond = pytest.importorskip('lilypond')

    _shutil_which = shutil.which
    monkeypatch.setattr(shutil, 'which', lambda x: None if x == 'lilypond' else _shutil_which(x))

    which_lilypond = thebeat._decorators.requires_lilypond(lambda: _shutil_which('lilypond'))()
    which_lilypond = re.sub(r'.EXE$', '', which_lilypond, flags=re.IGNORECASE)
    assert which_lilypond == str(lilypond.executable())

    rng = np.random.default_rng(42)
    melody = thebeat.music.Melody.generate_random_melody(n_bars=2, key='G', octave=4, rng=rng)
    fig, _ = melody.plot_melody(suppress_display=True)
    return fig


@pytest.mark.mpl_image_compare
def test_lilypond_system(monkeypatch):
    if shutil.which('lilypond') is None:
        pytest.skip("lilypond not found on PATH")

    monkeypatch.setitem(sys.modules, 'lilypond', None)

    system_lilypond = shutil.which('lilypond')
    which_lilypond = thebeat._decorators.requires_lilypond(lambda: shutil.which('lilypond'))()
    assert which_lilypond == system_lilypond

    rng = np.random.default_rng(42)
    melody = thebeat.music.Melody.generate_random_melody(n_bars=2, key='G', octave=4, rng=rng)
    fig, _ = melody.plot_melody(suppress_display=True)
    return fig


def test_lilypond_precedence(monkeypatch):
    lilypond = pytest.importorskip('lilypond')
    if shutil.which('lilypond') is None:
        pytest.skip("lilypond not found on PATH")

    which_lilypond = thebeat._decorators.requires_lilypond(lambda: shutil.which('lilypond'))()
    which_lilypond = re.sub(r'.EXE$', '', which_lilypond, flags=re.IGNORECASE)
    assert which_lilypond == str(lilypond.executable())
