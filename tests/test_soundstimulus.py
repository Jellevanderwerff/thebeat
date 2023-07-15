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

import matplotlib.pyplot as plt
import numpy as np
import pytest

import thebeat.core


@pytest.fixture
def rng():
    return np.random.default_rng(123)


def test_soundstimulus():
    stim = thebeat.core.soundstimulus.SoundStimulus.generate()
    assert stim.duration_ms == 50
    stim = thebeat.core.SoundStimulus.from_note('G6', duration=1000, offramp_ms=10)
    assert stim.duration_ms == 1000


def test_ramps():
    stim = thebeat.core.SoundStimulus.generate(freq=440, duration_ms=100, n_channels=1, onramp_ms=50)
    assert stim.duration_ms == 100
    stim = thebeat.core.SoundStimulus.generate(freq=440, duration_ms=100, n_channels=2, onramp_ms=50)
    assert stim.duration_ms == 100
    stim = thebeat.core.SoundStimulus.generate(freq=440, duration_ms=100, n_channels=2, onramp_ms=50, offramp_ms=50,
                                               ramp_type='raised-cosine')
    assert stim.duration_ms == 100


def test_whitenoise(rng):
    stim = thebeat.core.SoundStimulus.generate_white_noise(duration_ms=1000, rng=rng)
    assert stim.duration_ms == 1000


@pytest.mark.mpl_image_compare
def test_plot_stimulus_waveform_0(rng):  # Plot new plot
    # regular example
    stim = thebeat.core.SoundStimulus.generate_white_noise(duration_ms=1000, rng=rng)
    fig, ax = stim.plot_waveform(suppress_display=True)
    return fig


@pytest.mark.mpl_image_compare
def test_plot_stimulus_waveform_1(rng):  # Plot onto existing plot
    fig, axs = plt.subplots(1, 2)
    stim = thebeat.core.SoundStimulus.generate_white_noise(duration_ms=1000, rng=rng)
    stim.plot_waveform(ax=axs[0])
    return fig


def test_concat():
    stim1 = thebeat.SoundStimulus.generate(freq=440)
    stim2 = thebeat.SoundStimulus.generate(freq=880)
    st1 = stim1 + stim2
    st2 = thebeat.utils.concatenate_soundstimuli([stim1, stim2])
    assert st1.duration_ms == st2.duration_ms


def test_merge():
    sound = thebeat.SoundStimulus.generate(amplitude=0.5, freq=440)
    sound2 = thebeat.SoundStimulus.generate(amplitude=0.25, freq=880)
    new_sound = sound.merge(sound2)
    assert new_sound.duration_ms == sound.duration_ms
    new_sound = sound.merge([sound2, sound2])
    assert new_sound.duration_ms == sound.duration_ms
    with pytest.warns(UserWarning, match='Sound was normalized to avoid distortion'):
        new_sound = sound.merge([sound2, sound2, sound2])
    assert new_sound.duration_ms == sound.duration_ms


def test_copy():
    s = thebeat.SoundStimulus.generate(name='test')
    s2 = s.copy()
    s.name = "test2"
    assert s.name == 'test2'
    assert s2.name == 'test'
