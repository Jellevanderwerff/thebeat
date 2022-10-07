import combio.core


def test_stimulus():
    stim = combio.core.stimulus.Stimulus.generate()
    assert stim.duration_ms == 50


def test_ramps():
    stim = combio.core.Stimulus.generate(freq=440, duration_ms=100, n_channels=1, onramp=50)
    assert stim.duration_ms == 100
    stim = combio.core.Stimulus.generate(freq=440, duration_ms=100, n_channels=2, onramp=50)
    assert stim.duration_ms == 100
    stim = combio.core.Stimulus.generate(freq=440,
                                         duration_ms=100,
                                         n_channels=2,
                                         onramp=50,
                                         offramp=50,
                                         ramp_type='raised-cosine')
    assert stim.duration_ms == 100
