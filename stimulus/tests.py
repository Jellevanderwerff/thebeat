from stimulus import *
import parselmouth
from scipy.io import wavfile
import numpy as np
import os
import matplotlib.pyplot as plt

seq = Sequence.generate_isochronous(n=10, ioi=500)

stim1 = Stimulus.generate(freq=440, offramp=25, ramp='raised-cosine')
stim1.plot()