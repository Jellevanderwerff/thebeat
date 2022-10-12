import thebeat.core


def test_stimulus():
    stim = thebeat.core.stimulus.Stimulus.generate()
    assert stim.duration_ms == 50
    stim = thebeat.core.Stimulus.from_note('G6', duration=1000, offramp=10)
    assert stim.duration_ms == 1000


def test_ramps():
    stim = thebeat.core.Stimulus.generate(freq=440, duration=100, n_channels=1, onramp=50)
    assert stim.duration_ms == 100
    stim = thebeat.core.Stimulus.generate(freq=440, duration=100, n_channels=2, onramp=50)
    assert stim.duration_ms == 100
    stim = thebeat.core.Stimulus.generate(freq=440, duration=100, n_channels=2, onramp=50, offramp=50,
                                          ramp_type='raised-cosine')
    assert stim.duration_ms == 100
