from stimulus import *
from numpy.random import default_rng
from scipy.io import wavfile
from mingus.containers import Track, Bar, Note, NoteContainer

from stimulus import *

"""
Here we create a drum beat with kicks, snares, and hi-hats.
"""


stim = Stimulus.from_note('A')
stims = Stimuli.from_stims([stim, stim, None, stim])
for stim in stims:
    print(stims)

"""
# get stims
kick = Stimulus.from_wav('examples/resources/kick.wav')
snare = Stimulus.from_wav('examples/resources/snare.wav')
hihat = Stimulus.from_wav('examples/resources/hihat.wav')

# first layer
rhythm_kick = Rhythm.from_note_values([4] * 4)
stims_kick = Stimuli.from_stims([kick, None, kick, kick])
trial = RhythmTrial(rhythm_kick, stims_kick)
print(trial.events)
"""


"""
# second layer
rhythm_snare = Rhythm.from_note_values([4, 4, 4, 4])
stims_snare = Stimuli.from_stims([None, snare, None, snare])
trial.add_layer(rhythm_snare, stims_snare)
"""

# trial.plot_waveform()
# trial.play()
