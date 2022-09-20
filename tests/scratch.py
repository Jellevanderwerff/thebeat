from combio.stats import *
import numpy as np
from combio.core import Sequence, Stimulus, StimSequence

stimseq = StimSequence(Stimulus.generate(offramp=10), Sequence.generate_isochronous(n=10, ioi=500))
stimseq.play()
