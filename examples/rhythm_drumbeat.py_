from stimulus import *

"""
Here we create a drum beat with kicks, snares, and hi-hats.
"""

# get stims
kick = Stimulus.from_wav('resources/kick.wav')
snare = Stimulus.from_wav('resources/snare.wav')
hihat = Stimulus.from_wav('resources/hihat.wav')


# first layer
rhythm_kick = Rhythm.from_note_values([4] * 8)
stims_kick = Stimuli.from_stims([kick] * 8)
trial = RhythmTrial(stims_kick, rhythm_kick)

# second layer
rhythm_snare = Rhythm.from_note_values([4] * 8)
stims_snare = Stimuli.from_stims([None, snare, None, snare] * 2)
trial.add_layer(rhythm_snare, stims_snare)

# third layer
rhythm_hihat = Rhythm.from_note_values([16] * 32)
stims_hihat = Stimuli.from_stims([hihat] * 32)
trial.add_layer(rhythm_hihat, stims_hihat)

trial.plot_rhythm()
trial.play()

