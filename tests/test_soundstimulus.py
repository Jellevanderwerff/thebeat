import thebeat.core
import matplotlib.pyplot as plt


def test_soundstimulus():
    stim = thebeat.core.soundstimulus.SoundStimulus.generate()
    assert stim.duration_ms == 50
    stim = thebeat.core.SoundStimulus.from_note('G6', duration=1000, offramp_ms=10)
    assert stim.duration_ms == 1000


def test_ramps():
    stim = thebeat.core.SoundStimulus.generate(freq=440, duration_ms=100, n_channels=1, onramp_ms=50)
    assert stim.duration_ms == 100
    stim = thebeat.core.SoundStimulus.generate(freq=440, duration_ms=100, n_channels=2, onramp_ms=50)
    assert stim.duration_ms == 100
    stim = thebeat.core.SoundStimulus.generate(freq=440, duration_ms=100, n_channels=2, onramp_ms=50, offramp_ms=50,
                                               ramp_type='raised-cosine')
    assert stim.duration_ms == 100


def test_whitenoise():
    stim = thebeat.core.SoundStimulus.generate_white_noise(duration_ms=1000)
    assert stim.duration_ms == 1000


def test_plot_waveform():
    # regular example
    stim = thebeat.core.SoundStimulus.generate_white_noise(duration_ms=1000)
    fig, ax = stim.plot_waveform(suppress_display=True)
    assert fig, ax

    # plot onto existing Axes
    fig, axs = plt.subplots(1, 2)
    stim.plot_waveform(ax=axs[0])
    assert fig, axs[0]
