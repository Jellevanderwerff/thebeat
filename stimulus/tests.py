from stimulus import *
from numpy.random import default_rng
from scipy.io import wavfile

stims = Stimuli.from_notes('AAGG')

rhythm = Rhythm.generate_isochronous(1, time_sig=(4, 4), quarternote_ms=500)


trial = RhythmTrial(rhythm, stims)





