from matplotlib import pyplot as plt
import numpy as np
import thebeat.visualization
from thebeat.core import SoundStimulus, SoundSequence, Sequence
from thebeat.visualization import plot_multiple_sequences, recurrence_plot
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.mark.mpl_image_compare
def test_plot_multiple_sequences_0(rng):
    trials = []

    for x in range(10):
        seq = Sequence.generate_random_uniform(n_events=10, a=400, b=600, rng=rng)  # = 10 stimuli, 9 IOIs
        stims = [SoundStimulus.generate() for _ in range(10)]  # = 10 stimuli
        trials.append(SoundSequence(stims, seq))

    fig, ax = plot_multiple_sequences(trials, style='ggplot', suppress_display=True)

    return fig


@pytest.mark.mpl_image_compare
def test_plot_multiple_sequences_1(rng):
    seqs = [Sequence.generate_random_normal(10, mu=500, sigma=25, rng=rng) for _ in range(10)]
    plot_multiple_sequences(seqs, suppress_display=True)

    seq1 = Sequence.generate_random_normal(n_events=5, mu=500, sigma=25, end_with_interval=True, rng=rng)
    seq2 = Sequence.generate_random_normal(n_events=5, mu=500, sigma=25, end_with_interval=True, rng=rng)
    fig, ax = plot_multiple_sequences([seq1, seq2],
                                      figsize=(10, 5),
                                      colors=['red', 'blue'],
                                      suppress_display=True)

    return fig


@pytest.mark.mpl_image_compare
def test_plot_multiple_sequences_2(rng):
    seq1 = Sequence.generate_random_normal(n_events=5, mu=500, sigma=25, end_with_interval=True, rng=rng)
    seq2 = Sequence.generate_random_normal(n_events=5, mu=500, sigma=25, end_with_interval=True, rng=rng)
    fig, ax = plot_multiple_sequences([seq1, seq2],
                                      figsize=(10, 5),
                                      colors=[(1, 0, 0), (0, 0, 1)],
                                      suppress_display=True)
    return fig


@pytest.mark.mpl_image_compare
def test_recurrence_plot_threshold(rng):
    seq = Sequence.generate_random_normal(n_events=10, mu=500, sigma=20, end_with_interval=True, rng=rng) * 5
    fig, ax = recurrence_plot(seq, 0.03, suppress_display=True)

    return fig


@pytest.mark.mpl_image_compare
def test_recurrence_plot_nothreshold(rng):
    seq = Sequence.generate_random_normal(n_events=10, mu=500, sigma=20, end_with_interval=True, rng=rng) * 5
    fig, ax = recurrence_plot(seq, suppress_display=True)

    return fig


@pytest.mark.mpl_image_compare
def test_plot_phase_differences(rng):
    seq = Sequence.generate_random_normal(n_events=10, mu=500, sigma=20, end_with_interval=True, rng=rng) * 5
    fig, ax = thebeat.visualization.plot_phase_differences(seq, 500, binwidth=10, title="My first circular plot")
    assert fig, ax

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    thebeat.visualization.plot_phase_differences(seq, 500, ax=ax, suppress_display=True)

    return fig


@pytest.mark.mpl_image_compare
def test_interval_ratios_plot_density(rng):
    seqs = [Sequence.generate_random_normal(n_events=10, mu=500, sigma=100, rng=rng) for _ in range(100)]

    fig, ax = thebeat.visualization.plot_interval_ratios_density(seqs,
                                                                 suppress_display=True,
                                                                 title="My first density plot",
                                                                 resolution=0.1)
    return fig


@pytest.mark.mpl_image_compare
def test_interval_ratios_plot_histogram(rng):
    seqs = [Sequence.generate_random_normal(n_events=10, mu=500, sigma=100, rng=rng) for _ in range(100)]

    fig, ax = thebeat.visualization.plot_interval_ratios_histogram(seqs,
                                                                   suppress_display=True,
                                                                   title="My first density plot")
    return fig
