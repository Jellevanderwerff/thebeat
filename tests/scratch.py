from combio.stats import *
import numpy as np
from combio.core import Sequence

rng = np.random.default_rng(seed=123)  # for reproducability
seq = Sequence.generate_random_uniform(n=10, a=4000, b=6000, rng=rng)
df = acf_plot(seq, resolution_ms=10, smoothing_window=500, smoothing_sd=10)
