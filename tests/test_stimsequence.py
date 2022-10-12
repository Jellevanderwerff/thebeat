import pytest
import thebeat.core
import os


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
