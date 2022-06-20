from stimulus import *
import parselmouth
from scipy.io import wavfile
import numpy as np
import os


stim = Stimulus.generate(freq=500)
stims = Stimuli.from_stim(stim, 10)
