from combio.core import *
from random import shuffle

seq = Sequence.generate_random_uniform(5, 400, 600)
stims = [Stimulus.generate(freq=freq) for freq in [123, 424, 595, 142, 911]]
shuffle(stims)

trial = StimTrial(stims, seq)

trial.plot_waveform()
