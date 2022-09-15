from combio.core import *
from combio.rhythm import *
from typing import Union, Iterable
import matplotlib.pyplot as plt
import random
from combio.visualization import plot_sequence_single, plot_sequence_multiple
import os

seq = Sequence([464., 435., 492., 500., 457., 475., 472., 597., 527.])

trials = []

for x in range(1000):
    seq = Sequence.generate_random_uniform(n=10, a=400, b=600)  # = 10 stimuli, 9 IOIs
    stims = [Stimulus.generate(duration=random.randint(10, 350)) for y in range(10)]  # = 10 stimuli
    try:
        trials.append(StimSequence(stims, seq))
    except ValueError as e:
        raise Exception from e
