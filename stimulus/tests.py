from stimulus import *
from numpy.random import default_rng
from scipy.io import wavfile

stims = Stimuli.from_notes('AAAAAAAAGG')

seq = Sequence.generate_isochronous(n=10, ioi=500)

stimtrial = StimTrial(stims, seq, name="Trial 1")

print(stimtrial)

