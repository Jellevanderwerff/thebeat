from stimulus import *
from numpy.random import default_rng
from scipy.io import wavfile

kick = Stimulus.from_wav('test_files/kick.wav')
snare = Stimulus.from_wav('test_files/snare.wav')

rhythm_kick = Rhythm.from_note_values([4, 4, 4, 4])
stims_kick = Stimuli.from_stims([kick, kick, kick, kick])

rhythm_snare = Rhythm.from_note_values([4, 4, 4, 4])
stims_snare = Stimuli.from_stims([None, snare, None, snare])

trial = RhythmTrial(rhythm_kick, stims_kick)

trial.add_layer(rhythm_snare, stims_snare, layer_id=1)
trial.plot_waveform()
trial.play()
