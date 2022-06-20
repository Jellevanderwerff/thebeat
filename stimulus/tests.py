from stimulus import *
import parselmouth
from scipy.io import wavfile
import numpy as np
import os
import matplotlib.pyplot as plt

seq = Rhythm.from_note_values(note_values=[4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2],
                              time_signature=(4, 4),
                              quarternote_ms=500)

stim = Stimulus.generate(onramp=10, offramp=10, ramp='raised-cosine')
stim.plot()