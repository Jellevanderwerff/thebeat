from combio.core import Sequence, StimSequence
import matplotlib.pyplot as plt
from typing import Union, Iterable
import combio.core.helpers


def plot_sequence_single(sequence: Union[Sequence, StimSequence, Iterable],
                         style: str = 'seaborn',
                         title: str = None,
                         linewidth=None,
                         figsize=None,
                         suppress_display: bool = False):

    """
    This function may be used to plot a sequence of event onsets.
    Either pass it a Sequence or StimSequence object, or an iterable (e.g. list) of event onsets.

    Parameters
    ----------
    sequence : Sequence or StimSequence or iterable
        Either a Sequence or StimSequence object, or an iterable (e.g. list) of event onsets, e.g. [0, 500, 1000].
        Here, 0 is not required.
    style : str, optional
        A matplotlib style. See the matplotlib docs for options. Defaults to 'seaborn'.
    title : str, optional
        Here, one can provide a title for the plot.
    linewidth
    figsize
    suppress_display

    Returns
    -------

    """

    # Input validation, get onsets, set linewidths (the widths of the bars) and title.
    if isinstance(sequence, Sequence):
        # onsets
        onsets = sequence.onsets
        # linewidths
        if linewidth:
            linewidths = [linewidth] * len(onsets)
        else:
            linewidths = [50] * len(onsets)
        # title
        if sequence.name and title is None:
            title = sequence.name

    elif isinstance(sequence, StimSequence):
        # onsets
        onsets = sequence.onsets
        # linewidths
        linewidths = sequence.event_durations
        # title
        if sequence.name and title is None:
            title = sequence.name

    elif hasattr(sequence, '__iter__'):
        # onsets
        onsets = sequence
        # linewidths
        if linewidth:
            linewidths = [linewidth] * len(onsets)
        else:
            linewidths = [50] * len(onsets)
        # title
        title = title

    else:
        raise ValueError("Please provide a Sequence object, a StimSequence object, "
                         "or an iterable of event onsets.")

    fig, ax = combio.core.helpers.plot_sequence_single(onsets=onsets, style=style, title=title,
                                                       linewidths=linewidths, figsize=figsize,
                                                       suppress_display=suppress_display)
    return fig, ax


def plot_sequence_multiple(sequences: Iterable,
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
