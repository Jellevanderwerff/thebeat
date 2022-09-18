from combio.core import *
from combio.visualization import plot_sequence_single
from random import randint


stimseq = StimSequence(Stimulus.generate(), Sequence.generate_isochronous(n=100, ioi=500))
plot_sequence_single(stimseq)
