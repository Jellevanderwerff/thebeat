from stimulus import *

seq = random_rhythmic_sequence(n_bars=2, allowed_note_values=[2, 4, 8],
                               time_signature=(4, 4),
                               quarternote_ms=500)
seq.plot_rhythm()
stim = Stimulus.generate()

stim_seq = StimulusSequence(stim, seq)

stim_seq.play(loop=True, metronome=True, metronome_amplitude=0.5)


