from stimulus import *
from mingus.core.value import determine
from mingus.containers import Bar, Track
from mingus.core.value import tuplet



seq = random_rhythmic_sequence(2, [2, 4, 8, 16], time_signature=(3, 4), quarternote_ms=500)

seq.plot_rhythm()

seq.change_tempo(factor=0.66)

seq.plot_rhythm()