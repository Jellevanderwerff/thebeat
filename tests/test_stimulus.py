from combio.core import *


def test_stimulus():
    stim = Stimulus.generate()
    assert stim.duration_ms == 50

