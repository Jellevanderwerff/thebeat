from matplotlib import pyplot as plt

import thebeat.visualization
from thebeat.core import Stimulus, StimSequence, Sequence
from thebeat.visualization import plot_multiple_sequences, recurrence_plot


def test_event_plot_multiple():
    trials = []

    for x in range(10):
        seq = Sequence.generate_random_uniform(n=10, a=400, b=600)  # = 10 stimuli, 9 IOIs
        stims = [Stimulus.generate() for _ in range(10)]  # = 10 stimuli
        trials.append(StimSequence(stims, seq))

    fig, ax = plot_multiple_sequences(trials, style='ggplot', suppress_display=True)

    assert fig, ax

    seqs = [Sequence.generate_random_normal(10, mu=500, sigma=25) for _ in range(10)]
    plot_multiple_sequences(seqs, suppress_display=True)

    seq1 = Sequence.generate_random_normal(n=5, mu=500, sigma=25, end_with_interval=True)
    seq2 = Sequence.generate_random_normal(n=5, mu=500, sigma=25, end_with_interval=True)
    fig, ax = plot_multiple_sequences([seq1, seq2],
                                      figsize=(10, 5),
                                      colors=['red', 'blue'],
                                      suppress_display=True)
    assert fig, ax

    fig, ax = plot_multiple_sequences([seq1, seq2],
                                      figsize=(10, 5),
                                      colors=[(1, 0, 0), (0, 0, 1)],
                                      suppress_display=True)
    assert fig, ax


def test_recurrence_plot():
    seq = Sequence.generate_random_normal(n=10, mu=500, sigma=20, end_with_interval=True) * 5
    fig, ax = recurrence_plot(seq, 0.03)
    assert fig, ax


def test_plot_phase_differences():
    seq = Sequence.generate_random_normal(n=10, mu=500, sigma=20, end_with_interval=True) * 5
    fig, ax = thebeat.visualization.plot_phase_differences(seq, 500, binwidth=10, title="My first circular plot")
    assert fig, ax

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    thebeat.visualization.plot_phase_differences(seq, 500, ax=ax)

    assert fig, ax


def test_interval_ratios_plots():
    seqs = [Sequence.generate_random_normal(n=10, mu=500, sigma=100) for _ in range(100)]

    fig, ax = thebeat.visualization.plot_interval_ratios_density(seqs, suppress_display=True,
                                                                 title="My first density plot", resolution=0.1)
    assert fig, ax
    fig, ax = thebeat.visualization.plot_interval_ratios_hist(seqs, bins=10, suppress_display=True)
    assert fig, ax
