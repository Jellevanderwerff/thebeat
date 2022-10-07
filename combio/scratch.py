from combio.core import Stimulus

stim = Stimulus.generate(n_channels=2, offramp=50, ramp_type='raised-cosine')

stim.plot_waveform()
