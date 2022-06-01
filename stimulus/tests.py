from stimulus import *
from mingus.core.scales import Major
from mingus.containers import Note, Bar
from mingus.extra import lilypond
import random
import numpy as np

ran_seq = random_metrical_sequence(1, [4, 8, 16], (4, 4), 500)
note_values = iois_to_notevalues(ran_seq.iois, (4, 4), 500)

plot_note_values('./ritme.png', note_values, (4, 4))


