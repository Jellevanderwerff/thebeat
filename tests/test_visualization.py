from combio.core import Stimulus, StimSequence, Sequence
from combio.visualization import plot_single_sequence, plot_multiple_sequences
import random


def test_event_plot_single():
    seq = Sequence([500, 200, 1000])
    stims = [Stimulus.generate(),
             Stimulus.generate(),
             Stimulus.generate(),
             Stimulus.generate()]
    trial = StimSequence(stims, seq)
    plot_single_sequence(trial, style='seaborn', suppress_display=True)

    fig, ax = plot_single_sequence([50, 500, 1000], linewidths=50)
    assert fig, ax


def test_event_plot_multiple():
    trials = []

    for x in range(10):
        seq = Sequence.generate_random_uniform(n=10, a=400, b=600)  # = 10 stimuli, 9 IOIs
        stims = [Stimulus.generate() for _ in range(10)]  # = 10 stimuli
        trials.append(StimSequence(stims, seq))

    fig, ax = plot_multiple_sequences(trials, style='ggplot', suppress_display=True)

    assert fig
    assert ax

    seqs = [Sequence.generate_random_normal(10, mu=500, sigma=25) for _ in range(10)]
    plot_multiple_sequences(seqs, suppress_display=True)
