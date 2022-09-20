from combio.stats import *
import numpy as np
from combio.core import Sequence

rng = np.random.default_rng(seed=123)
seq = Sequence.generate_random_uniform(n=10, a=4500, b=5500, rng=rng)
acf_plot(seq, 1, smoothing_window=500, smoothing_sd=200)
