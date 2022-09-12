from combio.core import *
from combio.rhythm import *
from typing import Union, Iterable
import matplotlib.pyplot as plt
import random
from combio.visualization import plot_sequence_single, plot_sequence_multiple
import os


seq1 = Sequence.generate_isochronous(n=10, ioi=500)
seq2 = Sequence.generate_random_normal(n=10, mu=500, sigma=50)

plot_sequence_multiple([seq1, seq2],
                       sequence_names=['Isochronous', 'Random normal'],
                       figsize=(8, 4))
