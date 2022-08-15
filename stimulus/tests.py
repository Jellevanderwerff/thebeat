from stimulus import *
from numpy.random import default_rng
from scipy.io import wavfile
from mingus.containers import Track, Bar, Note, NoteContainer

from stimulus import *

seq = Sequence([500, 250, 250, 500, 500])

print(seq.get_integer_ratios_from_dyads(output='floats'))

