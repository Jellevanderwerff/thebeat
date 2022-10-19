import thebeat.core


def test_stimulus():
    stim = thebeat.core.stimulus.Stimulus.generate()
    assert stim.duration_ms == 50
    stim = thebeat.core.Stimulus.from_note('G6', duration=1000, offramp_ms=10)
    assert stim.duration_ms == 1000


def test_ramps():
    stim = thebeat.core.Stimulus.generate(freq=440, duration_ms=100, n_channels=1, onramp_ms=50)
    assert stim.duration_ms == 100
    stim = thebeat.core.Stimulus.generate(freq=440, duration_ms=100, n_channels=2, onramp_ms=50)
    assert stim.duration_ms == 100
    stim = thebeat.core.Stimulus.generate(freq=440, duration_ms=100, n_channels=2, onramp_ms=50, offramp_ms=50,
                                          ramp_type='raised-cosine')
    assert stim.duration_ms == 100


def test_whitenoise():
    stim = thebeat.core.Stimulus.generate_white_noise(duration_ms=1000)
    assert stim.duration_ms == 1000