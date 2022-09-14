from combio.core import *
from combio.rhythm import *
import numpy as np


def test_rhythmtrial():
    r = Rhythm.from_iois([500, 500, 500, 500], (4, 4), 500)
    stim = Stimulus.generate()

    trial = RhythmTrial(stim, r)
    trial.plot_rhythm(suppress_display=True)


