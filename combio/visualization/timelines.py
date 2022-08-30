from combio.core import Sequence, StimTrial, Stimulus
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Iterable


def event_plot_single(sequence: Union[Sequence, StimTrial],
                      style: str = 'seaborn',
                      linewidth=10,
                      suppress_display: bool = False):
    # Input validation and setting line widths
    if isinstance(sequence, Sequence):
        linewidths = linewidth
    elif isinstance(sequence, StimTrial):
        linewidths = sequence.event_durations
    else:
        raise ValueError("Pass either a Sequence or a StimTrial object as the first argument.")

    # Make plot
    with plt.style.context(style):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 1)
        ax.barh(0.5, width=linewidths, height=0.3, left=sequence.onsets)
        ax.axes.yaxis.set_visible(False)
        ax.axes.set_xlabel("Time (ms)")

    # Show plot
    if not suppress_display:
        plt.show()

    # Additionally return plot
    return fig, ax


def event_plot_multiple(sequences,
                        style: str = 'seaborn',
                        bar_names: Iterable[str] = None,
                        linewidth: int = None,
                        suppress_display: bool = False):

    # Input validation
    if not all(isinstance(sequence, Sequence) for sequence in sequences) and not all(
            isinstance(sequence, StimTrial) for sequence in sequences):
        raise ValueError("Please pass an iterable with either only Sequence objects or only StimTrial objects.")

    # Make names for the bars
    n_bars = len(sequences)
    if bar_names is None:
        bar_names = [str(i) for i in range(1, n_bars+1)]
    elif len(bar_names) != len(sequences):
        raise ValueError("Please provide an equal number of bar names as sequences.")

    # Make line widths (these are either the event durations in case StimTrials were passed, in case of Sequences these
    # default to 10 points).
    if linewidth is None:
        if isinstance(sequences[0], Sequence):
            linewidths = [10] * len(sequences)
        elif isinstance(sequences[0], StimTrial):
            linewidths = [trial.event_durations for trial in sequences]

    # Plot
    with plt.style.context(style):
        fig, ax = plt.subplots()

        for seq, label, linewidths in zip(sequences, bar_names, linewidths):
            ax.barh(y=label, width=linewidths, left=seq.onsets)

        if not suppress_display:
            plt.show()

    return fig, ax


