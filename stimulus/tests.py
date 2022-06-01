from stimulus import *
from mingus.core.scales import Major
from mingus.containers import Note, Bar
from mingus.extra import lilypond
import random
import numpy as np

iois = [500, 250, 250, 500, 500]
time_signature = (4, 4)
quarternote_ms = 500

note_values = iois_to_notevalues(iois, time_signature, quarternote_ms)

plot_note_values('./ritme.png', note_values, time_signature)

