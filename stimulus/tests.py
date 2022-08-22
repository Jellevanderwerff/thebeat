from stimulus import *
from numpy.random import default_rng
from scipy.io import wavfile
from mingus.containers import Track, Bar, Note, NoteContainer
from fractions import Fraction

from stimulus import *

seq = Rhythm.from_note_values([4, 4, 4, 4], time_signature=(4, 4), beat_ms=500)
seq.plot_rhythm()
stim = Stimulus.generate()
stims = Stimuli.from_stims([stim, None, stim, None])
trial = RhythmTrial(stims, seq)


trial.plot_rhythm()
