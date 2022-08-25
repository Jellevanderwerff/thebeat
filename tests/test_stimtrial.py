import pytest
from combio.core import *


def test_stimtrial():
    seq = Sequence.generate_isochronous(10, 500)
    stim = Stimulus.generate()
    trial = StimTrial(stim, seq)

    assert len(trial.onsets) == 10

    # make sure error is raised when provided stimuli differ in sampling frequency
    seq = Sequence.generate_isochronous(10, 500)
    stim1 = Stimulus.generate(fs=48000)
    stim2 = Stimulus.generate(fs=44100)
    stims = [stim1, stim2] * 5
    with pytest.raises(Exception):
        trial = StimTrial(stims, seq)
