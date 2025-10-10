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

import thebeat._warnings
import thebeat.core


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_soundsequence(tmp_path):
    seq = thebeat.core.Sequence.generate_isochronous(10, 500)
    stim = thebeat.core.SoundStimulus.generate()
    trial = thebeat.core.SoundSequence(stim, seq)

    assert len(trial.onsets) == 10

    # make sure error is raised when provided stimuli differ in sampling frequency
    seq = thebeat.core.Sequence.generate_isochronous(10, 500)
    stim1 = thebeat.core.SoundStimulus.generate(fs=48000)
    stim2 = thebeat.core.SoundStimulus.generate(fs=44100)
    stims = [stim1, stim2] * 5
    with pytest.raises(Exception):
        _ = thebeat.core.SoundSequence(stims, seq)

    stim = thebeat.core.SoundStimulus.generate()
    seq = thebeat.core.Sequence.generate_isochronous(n_events=2, ioi=500)
    trial = thebeat.core.SoundSequence(stim, seq)

    trial.write_wav(tmp_path / 'test.wav', metronome=True)

    # test datatype

    _, samples = scipy.io.wavfile.read(tmp_path / 'test.wav')
    assert samples.dtype == np.int16

    # compare seconds and milliseconds
    trial_s = thebeat.core.SoundSequence(thebeat.core.SoundStimulus.generate(),
                                         thebeat.core.Sequence.generate_isochronous(n_events=10, ioi=0.5),
                                         sequence_time_unit="s")
    trial_ms = thebeat.core.SoundSequence(thebeat.core.SoundStimulus.generate(),
                                          thebeat.core.Sequence.generate_isochronous(n_events=10, ioi=500),
                                          sequence_time_unit="ms")

    assert np.all(trial_s.iois == trial_ms.iois)


def test_multiplication():
    trial = thebeat.core.SoundSequence(thebeat.core.SoundStimulus.generate(),
                                       thebeat.core.Sequence.generate_isochronous(n_events=5, ioi=100,
                                                                                  end_with_interval=True))
    trial *= 10

    assert len(trial.onsets) == 50


@pytest.custom_mpl_image_compare
def test_soundsequence_plot():
    ss = thebeat.core.SoundSequence(thebeat.core.SoundStimulus.generate(),
                                    thebeat.core.Sequence.generate_isochronous(n_events=10, ioi=500))

    fig, ax = ss.plot_waveform()

    return fig


def test_concat():
    # SoundStimulus
    sndseq = thebeat.SoundSequence(thebeat.SoundStimulus.generate(),
                                   thebeat.Sequence.generate_isochronous(5, 500, True))

    ss1 = sndseq + sndseq
    ss2 = thebeat.utils.concatenate_soundsequences([sndseq, sndseq])

    assert ss1, ss2


def test_merge():
    sndseq = thebeat.SoundSequence(thebeat.SoundStimulus.generate(),
                                   thebeat.Sequence.generate_isochronous(5, 500, True))
    sndseq2 = thebeat.SoundSequence(thebeat.SoundStimulus.generate(freq=880),
                                    thebeat.Sequence.from_onsets([250, 750, 1250]))

    ss1 = sndseq + sndseq2
    ss2 = thebeat.utils.merge_soundsequences([sndseq, sndseq2])

    assert ss1, ss2


def test_copy():
    ss = thebeat.SoundSequence(thebeat.SoundStimulus.generate(), thebeat.Sequence([500, 500]), name='test')
    ss2 = ss.copy()
    ss.name = "test2"
    assert ss2.name == 'test'
    assert ss.name == 'test2'


def test_multichannel(tmp_path):
    s1 = thebeat.SoundSequence(thebeat.SoundStimulus.generate(n_channels=2), thebeat.Sequence.generate_isochronous(4, 500))
    s2 = thebeat.SoundSequence(thebeat.SoundStimulus.generate(), thebeat.Sequence.generate_isochronous(10, 400))

    thebeat.SoundSequence.write_multichannel_wav([s1, s2], tmp_path / 'test.wav', dtype=np.float32)

    _, samples = scipy.io.wavfile.read(tmp_path / 'test.wav')
    assert samples.ndim == 2
    assert samples.shape[1] == 3
    assert samples.dtype == np.float32


def test_soundsequence_rounding_warning(rng):
    with pytest.warns(UserWarning, match=r"the exact start or end positions in frames \(i.e. samples\) were rounded off to the neirest integer ceiling"):
        seq = thebeat.core.Sequence.generate_random_normal(10, 500, 50, rng=rng)
        stim = thebeat.core.SoundStimulus.generate()
        thebeat.core.SoundSequence(stim, seq)


@pytest.mark.mpl_image_compare
def test_soundsequence_plot_sequence(rng):
    seq = thebeat.core.Sequence.generate_random_normal(10, 500, 50, rng=rng)
    seq.round_onsets()
    stim = thebeat.core.SoundStimulus.generate()
    sound_sequence = thebeat.core.SoundSequence(stim, seq, name='TestSoundSequence')
    fig, _ = sound_sequence.plot_sequence()
    return fig


@pytest.mark.mpl_image_compare
def test_soundsequence_plot_waveform(rng):
    seq = thebeat.core.Sequence.generate_random_normal(10, 500, 50, rng=rng)
    seq.round_onsets()
    stim = thebeat.core.SoundStimulus.generate()
    sound_sequence = thebeat.core.SoundSequence(stim, seq, name='TestSoundSequence')
    fig, _ = sound_sequence.plot_waveform()
    return fig
