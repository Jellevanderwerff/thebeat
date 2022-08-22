from stimulus import *


seq = Sequence.generate_isochronous(n=10, ioi=500)
stim = Stimulus.generate()
stims = Stimuli.from_stim(stim, repeats=10)
stims.randomize()

trial = StimTrial(stims, seq)
trial.plot_waveform()