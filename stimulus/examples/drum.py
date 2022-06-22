from stimulus import *

kick = Stimulus.from_wav('../test_files/kick.wav')
snare = Stimulus.from_wav('../test_files/snare.wav')


rhythm_kick = Rhythm.from_note_values([4, 4, 4, 4])
stims_kick = Stimuli.from_stim(kick, 4)

rhythm_snare = Rhythm.from_note_values([4, 4, 4, 4], played=[False, True, False, True])
stims_snare = Stimuli.from_stim(snare, 4)

trial = RhythmTrial(rhythm_kick, stims_kick)

trial.add_layer(rhythm_snare, stims_snare, layer_id=1)

trial.plot_waveform()
trial.play()

