from combio.core import *
from combio.visualization import plot_single_sequence, plot_multiple_sequences
from random import randint

sequence_one = Sequence.generate_isochronous(n=8, ioi=500)
sequence_two = Sequence.generate_random_uniform(n=8, a=200, b=800)
plot_multiple_sequences([sequence_one, sequence_two], style='seaborn', sequence_names=['Isochronous', 'Random uniform'],
                        title="My awesome plot")


