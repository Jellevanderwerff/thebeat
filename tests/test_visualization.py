from combio.core import Stimulus, StimSequence, Sequence
from combio.visualization import plot_sequence_single, plot_sequence_multiple
import random


def test_event_plot_single():
    seq = Sequence([500, 200, 1000])
    stims = [Stimulus.generate(duration=150),
             Stimulus.generate(duration=10),
             Stimulus.generate(duration=200),
             Stimulus.generate(duration=200)]
    trial = StimSequence(stims, seq)
    plot_sequence_single(trial, style='seaborn', suppress_display=True)


def test_event_plot_multiple():
    trials = []

    for x in range(10):
        seq = Sequence.generate_random_uniform(n=10, a=400, b=600)  # = 10 stimuli, 9 IOIs
        stims = [Stimulus.generate(duration=random.randint(10, 350)) for y in range(10)]  # = 10 stimuli
        trials.append(StimSequence(stims, seq))

    fig, ax = plot_sequence_multiple(trials, style='ggplot', suppress_display=True)

    assert fig
    assert ax

    seqs = [Sequence.generate_random_normal(10, mu=500, sigma=25) for x in range(10)]
    plot_sequence_multiple(seqs, suppress_display=True)
