from stimulus import *
import random
from mingus.containers import Note

notes = get_major_scale_freqs('G')

seq = random_rhythmic_sequence(1, [2, 4, 8, 16], (3, 4), quarternote_ms=500)
stims = [Stimulus.generate(freq=random.choice(notes), offramp=10) for onset in seq.onsets]

stimseq = StimulusSequence(stims, seq)

stimseq.plot_music(key='G')
stimseq.play(metronome=True)
