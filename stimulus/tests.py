from stimulus import *
from numpy.random import default_rng
from scipy.io import wavfile

kick = Stimulus.from_wav('test_files/kick.wav')

stims1 = Stimuli.from_notes('AAAA')
rhythm1 = Rhythm.from_note_values([4, 4, 4, 4], time_signature=(4, 4), beat_ms=500)

stims2 = Stimuli.from_notes('XGXGXGXG')
rhythm2 = Rhythm.from_note_values([8, 8, 8, 8, 8, 8, 8, 8], time_signature=(4, 4), beat_ms=500)

rhythmtrial = RhythmTrial(rhythm1, stims1, name="test")

#rhythmtrial.add_layer(rhythm2, stims2, layer_id=1)

rhythmtrial.play()
