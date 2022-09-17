import combio.core
from combio.visualization import plot_sequence_single
from random import randint


seq = combio.core.Sequence.generate_isochronous(n=10, ioi=500)
stims = [combio.core.Stimulus.generate(duration=randint(100, 400)) for _ in range(10)]

trial = combio.core.StimSequence(stims, seq)
plot_sequence_single(trial)
