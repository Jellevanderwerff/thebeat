from stimulus import *

ratios = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5] * 2
seq = metrical_sequence(ratios, (4, 4), quarternote_ms=500)

notes = notes_to_freqs('CCGGAAGFFEEDDC')
stims = [Stimulus.generate(freq=note, onramp=10, offramp=10) for note in notes]

stim_seq = StimulusSequence(stims, seq)
stim_seq.play()
