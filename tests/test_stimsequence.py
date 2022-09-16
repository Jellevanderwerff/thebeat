import pytest
import combio.core
import os


def test_stimsequence():
    seq = combio.core.Sequence.generate_isochronous(10, 500)
    stim = combio.core.Stimulus.generate()
    trial = combio.core.StimSequence(stim, seq)

    assert len(trial.onsets) == 10

    # make sure error is raised when provided stimuli differ in sampling frequency
    seq = combio.core.Sequence.generate_isochronous(10, 500)
    stim1 = combio.core.Stimulus.generate(fs=48000)
    stim2 = combio.core.Stimulus.generate(fs=44100)
    stims = [stim1, stim2] * 5
    with pytest.raises(Exception):
        _ = combio.core.StimSequence(stims, seq)

    stim = combio.core.Stimulus.generate()
    seq = combio.core.Sequence.generate_isochronous(n=2, ioi=500)
    trial = combio.core.StimSequence(stim, seq)

    trial.write_wav('test.wav', metronome=True)
    os.remove('test.wav')
