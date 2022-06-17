from stimulus import *
import random


seq1 = Sequence.generate_isochronous(n=10, ioi=500, metrical=True)
seq2 = Sequence.generate_isochronous(n=10, ioi=400, metrical=True)

seq_joined = seq1 + seq2

