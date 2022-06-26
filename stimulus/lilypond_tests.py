from stimulus import *
from mingus.extra import lilypond

# get stims
kick = Stimulus.from_wav('examples/resources/kick.wav')
snare = Stimulus.from_wav('examples/resources/snare.wav')
hihat = Stimulus.from_wav('examples/resources/hihat.wav')

# first layer
rhythm_kick = Rhythm.from_note_values([4] * 4)
stims_kick = Stimuli.from_stims([kick] * 4)
trial = RhythmTrial(rhythm_kick, stims_kick)

# second layer
rhythm_snare = Rhythm.from_note_values([4] * 4)
stims_snare = Stimuli.from_stims([None, snare, None, snare])
trial.add_layer(rhythm_snare, stims_snare)

# third layer
rhythm_hihat = Rhythm.from_note_values([16] * 16)
stims_hihat = Stimuli.from_stims([hihat] * 16)
trial.add_layer(rhythm_hihat, stims_hihat)


trial.plot_rhythm(print_staff=True, lilypond_percussion_notes=['bd', 'snare', 'hihat'])
