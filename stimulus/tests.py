from stimulus import *
from numpy.random import default_rng
from scipy.io import wavfile
from mingus.containers import Track, Bar, Note, NoteContainer

from stimulus import *


seq = Sequence(iois=[500, 500, 1000, 1000])
values = seq.get_integer_ratios_from_dyads(output='floats')  # output = [0.5, 0.333334, 0.5]


