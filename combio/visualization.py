from combio.core import Sequence, StimSequence
import matplotlib.pyplot as plt
from typing import Union, Iterable
import numpy as np
import combio.core.helpers


def plot_sequence_single(sequence: Union[Sequence, StimSequence, Iterable],
                         style: str = 'seaborn',
                         title: str = None,
                         linewidth=None,
                         figsize=None,
                         suppress_display: bool = False):

    fig, ax = combio.core.helpers.plot_sequence_single(sequence=sequence, style=style, title=title,
                                                       linewidth=linewidth, figsize=figsize,
                                                       suppress_display=suppress_display)
    return fig, ax


def plot_sequence_multiple(sequences: Union,
                           style: str = 'seaborn',
                           sequence_names: Iterable[str] = None,
                           title: str = None,
                           linewidth: int = None,
                           figsize: tuple = None,
                           suppress_display: bool = False):
    # Input validation
    if not all(isinstance(sequence, Sequence) for sequence in sequences) and not all(
            isinstance(sequence, StimSequence) for sequence in sequences):
        raise ValueError("Please pass an iterable with either only Sequence objects or only StimSequence objects.")

    # Make names for the bars
    n_bars = len(sequences)
    if sequence_names is None:
        sequence_names = [str(i) for i in range(1, n_bars + 1)]
    elif len(sequence_names) != len(sequences):
        raise ValueError("Please provide an equal number of bar names as sequences.")

    # Make line widths (these are either the event durations in case StimTrials were passed, in case of Sequences these
    # default to 50 points).
    if linewidth is None:
        if isinstance(sequences[0], Sequence):
            linewidths = [50] * len(sequences)
        elif isinstance(sequences[0], StimSequence):
            linewidths = [trial.event_durations for trial in sequences]
    else:
        linewidths = [linewidth] * len(sequences)

    # Plot
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        # Labels
        if title is not None:
            ax.axes.set_title(title)
        ax.axes.set_xlabel("Time (ms)")

        for seq, label, linewidths in zip(sequences, sequence_names, linewidths):
            ax.barh(y=label, width=linewidths, left=seq.onsets)

        if not suppress_display:
            plt.show()

    return fig, ax
