import pytest
import thebeat.core
import os
import numpy as np

def test_stimsequence(tmp_path):
    seq = thebeat.core.Sequence.generate_isochronous(10, 500)
    stim = thebeat.core.Stimulus.generate()
    trial = thebeat.core.StimSequence(stim, seq)

    assert len(trial.onsets) == 10

    # make sure error is raised when provided stimuli differ in sampling frequency
    seq = thebeat.core.Sequence.generate_isochronous(10, 500)
    stim1 = thebeat.core.Stimulus.generate(fs=48000)
    stim2 = thebeat.core.Stimulus.generate(fs=44100)
    stims = [stim1, stim2] * 5
    with pytest.raises(Exception):
        _ = thebeat.core.StimSequence(stims, seq)

    stim = thebeat.core.Stimulus.generate()
    seq = thebeat.core.Sequence.generate_isochronous(n=2, ioi=500)
    trial = thebeat.core.StimSequence(stim, seq)

    trial.write_wav(tmp_path / 'test.wav', metronome=True)

    # compare seconds and milliseconds
    trial_s = thebeat.core.StimSequence(thebeat.core.Stimulus.generate(),
                                        thebeat.core.Sequence.generate_isochronous(n=10, ioi=0.5),
                                        sequence_time_unit="s")
    trial_ms = thebeat.core.StimSequence(thebeat.core.Stimulus.generate(),
                                         thebeat.core.Sequence.generate_isochronous(n=10, ioi=500),
                                         sequence_time_unit="ms")

    assert np.all(trial_s.iois == trial_ms.iois)


def test_multiplication():
    trial = thebeat.core.StimSequence(thebeat.core.Stimulus.generate(),
                                      thebeat.core.Sequence.generate_isochronous(n=5, ioi=100, metrical=True))
    trial *= 10

    assert len(trial.onsets) == 50
