from combio.core import Stimulus, StimTrial, Sequence
from combio.visualization import event_plot_single, event_plot_multiple
import random


def test_event_plot_single():
    seq = Sequence([500, 200, 1000])
    stims = [Stimulus.generate(duration=150),
             Stimulus.generate(duration=10),
             Stimulus.generate(duration=200),
             Stimulus.generate(duration=200)]
    trial = StimTrial(stims, seq)
    event_plot_single(trial, style='seaborn')


def test_event_plot_multiple():
    trials = []

    for x in range(10):
        seq = Sequence.generate_random_uniform(n=10, a=400, b=600)
        stims = [Stimulus.generate(duration=random.randint(10, 350)) for y in range(10)]
        trials.append(StimTrial(stims, seq))

    fig, ax = event_plot_multiple(trials, style='ggplot')

    assert fig
    assert ax

