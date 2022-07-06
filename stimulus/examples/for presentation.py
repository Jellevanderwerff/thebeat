from stimulus import *

seq = Sequence.generate_isochronous(n=10, ioi=1000)
seq.change_tempo_linearly(total_change=4)

stim = Stimulus.generate(freq=440,
                         duration=75,
                         onramp=10,
                         offramp=10,
                         ramp='raised-cosine')

stims = Stimuli.from_stim(stim, repeats=10)

trial = StimTrial(stims, seq)
trial.plot_waveform(title="Ritardando")

print(seq)
