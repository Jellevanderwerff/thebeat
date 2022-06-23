from stimulus import *

# get stims
kick = Stimulus.from_wav('test_files/kick mono.wav')
snare = Stimulus.from_wav('test_files/snare mono.wav')
hihat = Stimulus.from_wav('test_files/hihat.wav')

# first layer
rhythm_kick = Rhythm.from_note_values([4] * 8)
stims_kick = Stimuli.from_stims([kick] * 8)
trial = RhythmTrial(rhythm_kick, stims_kick)

# second layer
rhythm_snare = Rhythm.from_note_values([4, 4, 4, 4] * 2)
stims_snare = Stimuli.from_stims([None, snare, None, snare] * 2)
trial.add_layer(rhythm_snare, stims_snare)

# third layer
rhythm_hihat = Rhythm.from_note_values([16] * 32)
stims_hihat = Stimuli.from_stims([hihat] * 32)
trial.add_layer(rhythm_hihat, stims_hihat)

trial.play()
