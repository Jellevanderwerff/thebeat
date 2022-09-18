from combio.core import Sequence
from combio.visualization import plot_multiple_sequences

seq = Sequence.generate_isochronous(n=5, ioi=500)
seq2 = [0, 500, 1000]

plot_multiple_sequences([seq, seq2])
