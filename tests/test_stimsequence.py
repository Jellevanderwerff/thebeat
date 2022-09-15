import pytest
from combio.core import *
import os


def test_stimsequence():
    seq = Sequence.generate_isochronous(10, 500)
    stim = Stimulus.generate()
    trial = StimSequence(stim, seq)

    assert len(trial.onsets) == 10

    # make sure error is raised when provided stimuli differ in sampling frequency
    seq = Sequence.generate_isochronous(10, 500)
    stim1 = Stimulus.generate(fs=48000)
    stim2 = Stimulus.generate(fs=44100)
    stims = [stim1, stim2] * 5
    with pytest.raises(Exception):
        trial = StimSequence(stims, seq)

    stim = Stimulus.generate()
    seq = Sequence.generate_isochronous(n=2, ioi=500)
    trial = StimSequence(stim, seq)

    trial.write_wav('test.wav', metronome=True)
    os.remove('test.wav')