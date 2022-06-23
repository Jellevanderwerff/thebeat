from stimulus import *
from numpy.random import default_rng
from scipy.io import wavfile
from mingus.containers import Track, Bar, Note, NoteContainer

rhythm = Rhythm.generate_random_rhythm(allowed_note_values=[4, 8, 16], beat_ms=1000)
rhythm.plot_rhythm()

stim = Stimulus.generate()

stims = Stimuli.from_stim(stim, repeats=len(rhythm.onsets))

trial = RhythmTrial(rhythm, stims)
trial.play()

