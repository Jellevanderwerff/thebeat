import combio
import numpy as np


def test_rhythmtrial():
    r = combio.rhythm.Rhythm.from_iois([500, 500, 500, 500], (4, 4), 500)
    stim = combio.core.Stimulus.generate()

    trial = combio.rhythm.RhythmTrial(stim, r)
    trial.plot_rhythm(suppress_display=True)


