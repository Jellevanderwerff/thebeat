from combio.core import *
from combio.rhythm import *
from typing import Union, Iterable
import matplotlib.pyplot as plt
import random
from combio.visualization import event_plot_multiple, event_plot_single


seq = Sequence.generate_random_normal(10, 500, 200)
event_plot_single(seq)



