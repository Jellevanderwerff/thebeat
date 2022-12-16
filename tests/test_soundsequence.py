import pytest
import thebeat.core
import os
import numpy as np


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
