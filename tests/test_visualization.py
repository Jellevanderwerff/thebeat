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
    event_plot_single(trial, style='seaborn', suppress_display=True)


def test_event_plot_multiple():
    trials = []

    for x in range(10):
        seq = Sequence.generate_random_uniform(n=10, a=400, b=600)
        stims = [Stimulus.generate(duration=random.randint(10, 250)) for y in range(10)]
        trials.append(StimTrial(stims, seq))

    fig, ax = event_plot_multiple(trials, style='ggplot', suppress_display=True)

    assert fig
    assert ax

    seqs = [Sequence.generate_random_normal(10, mu=500, sigma=25) for x in range(10)]
    event_plot_multiple(seqs, suppress_display=True)




