from stimulus import *
from mingus.extra import lilypond

# get stims
stim = Stimulus.generate(freq=440)

# first layer
rhythm = Rhythm.generate_random_rhythm(n_bars=8, allowed_note_values=[4, 8])
stims = Stimuli.from_stim(stim, repeats=len(rhythm.onsets))

trial = RhythmTrial(stims, rhythm)


# second layer
rhythm2 = Rhythm.generate_random_rhythm(n_bars=8, allowed_note_values=[4, 8])
stims2 = Stimuli.from_stim(stim, repeats=len(rhythm2.onsets))
trial.add_layer(rhythm2, stims2)


# third layer
rhythm_hihat = Rhythm.from_note_values([8] * 8 * 8)
stims_hihat = Stimuli.from_stims([stim] * 8 * 8)


trial.add_layer(rhythm_hihat, stims_hihat)

trial.plot_rhythm()
