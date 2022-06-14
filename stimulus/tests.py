from stimulus import *

seq = random_rhythmic_sequence(n_bars=2, allowed_note_values=[2, 4, 8, 16], time_signature=(3, 4), quarternote_ms=500)
seq.plot_rhythm()

