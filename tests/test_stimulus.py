import combio.core


def test_stimulus():
    stim = combio.core.stimulus.Stimulus.generate()
    assert stim.duration_ms == 50

