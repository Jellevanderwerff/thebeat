from stimulus import *
import parselmouth
from scipy.io import wavfile
import numpy as np
import os

seq = Rhythm.from_note_values(note_values=[4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2],
                              time_signature=(4, 4),
                              quarternote_ms=500)

stim = Stimulus.generate(freq=440)
stim.plot()
