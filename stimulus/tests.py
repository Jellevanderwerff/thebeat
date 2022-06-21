from stimulus import *
from numpy.random import default_rng


stim_u = Stimulus.from_wav('test_files/u.wav', name='u', extract_pitch=True)
stim_e = Stimulus.from_wav('test_files/e.wav', name='e', extract_pitch=True)

