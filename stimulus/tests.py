from stimulus import *
import parselmouth
from scipy.io import wavfile
import numpy as np
import os
import matplotlib.pyplot as plt


stim = Stimulus.from_wav('metronome stereo.wav', name="My metronome")
stims = Stimuli.from_stim(stim, repeats=3)
seq = Sequence.generate_isochronous(n=3, ioi=1000)

stimseq = StimSequence(stims, seq, name="poep")
stimseq.plot_waveform(title="BELANGRIJk")
stimseq.play()
