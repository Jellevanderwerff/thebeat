from combio.core import *
from combio.visualization import plot_sequence_single, plot_sequence_multiple
from random import randint

sequence_one = Sequence.generate_isochronous(n=8, ioi=500)
sequence_two = Sequence.generate_random_uniform(n=8, a=200, b=800)
plot_sequence_multiple([sequence_one, sequence_two],
                       sequence_names=['Isochronous', 'Random uniform'],
                       title="My awesome plot", style='seaborn')
