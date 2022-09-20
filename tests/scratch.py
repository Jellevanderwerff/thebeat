from combio.stats import *
import numpy as np
from combio.core import Sequence, Stimulus, StimSequence

data = np.array([0.1, 1.1, 2.1, 3.1,4.1,5.1,6.1,7.1,8.1, 9.1])
data = data * 1000

ugof = get_ugof(data, theoretical_ioi=(1000 / 14.6597))
print(ugof)
