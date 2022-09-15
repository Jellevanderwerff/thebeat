from combio.core import *
from combio.rhythm import *
from combio.stats import *
from combio.linguistic import *
from typing import Union, Iterable
import matplotlib.pyplot as plt
import random
from combio.visualization import plot_sequence_single, plot_sequence_multiple
import os
import scipy.signal

mora = generate_trimoraic_sequence(10)
acf_plot(mora)
print(acf_df(mora))