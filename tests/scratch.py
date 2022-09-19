from combio.core import Sequence
from combio.visualization import plot_multiple_sequences, plot_single_sequence

seq = [2000, 5000, 10001]
plot_single_sequence(seq, figsize=(4, 2), linewidths=[200, 500, 10])
