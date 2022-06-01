from stimulus import *

stim = Stimulus.generate(freq=440, duration=50, onramp=10, offramp=10)
seq = random_metrical_sequence(1, [3], (4, 4), quarternote_ms=500)

print(seq)

stim_seq = StimulusSequence(stim, seq)

stim_seq.plot()
stim_seq.play()
