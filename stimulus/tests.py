from stimulus import *
from numpy.random import default_rng
from scipy.io import wavfile
from mingus.containers import Track, Bar, Note, NoteContainer

from stimulus import *

seq = Sequence.from_integer_ratios([4, 4, 2, 1], value_of_one_in_ms=500)

print(seq)
