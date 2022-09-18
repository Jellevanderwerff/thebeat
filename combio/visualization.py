from combio.core import Sequence, StimSequence
import matplotlib.pyplot as plt
from typing import Union, Iterable
import combio.core.helpers
import numpy as np


def plot_sequence_single(sequence: Union[Sequence, StimSequence, Iterable],
                         style: str = 'seaborn',
                         title: str = None,
                         linewidths: Iterable = None,
                         figsize=None,
                         suppress_display: bool = False):
    """
    This function may be used to plot a single sequence of event onsets.
    Either pass it a Sequence or StimSequence object, or an iterable (e.g. list) of event onsets.

    This function is similarly used by the Sequence.plot() and StimSequence.plot_sequence() methods.

    Parameters
    ----------
    sequence : Sequence or StimSequence or iterable
        Either a Sequence or StimSequence object, or an iterable (e.g. list) of event onsets, e.g. [0, 500, 1000].
        Here, 0 is not required.
    style : str, optional
        A matplotlib style. See the matplotlib docs for options. Defaults to 'seaborn'.
    title : str, optional
        If desired, one can provide a title for the plot. This takes precedence over using the
        Sequence or StimSequence name as the title of the plot (if passed and the object has one).
    linewidths : iterable, optional
        An iterable containing the desired width of the bars (events) in milliseconds.
        Defaults to 50 milliseconds if not provided.
    figsize : tuple, optional
        The desired figure size in inches as a tuple: (width, height).
    suppress_display : bool, optional
        If True, the plot is only returned, and not displayed via plt.show()

    Returns
    -------
    fig : Figure
        A matplotlib Figure object
    ax : Axes
        A matplotlib Axes object

    Examples
    --------
    >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
    >>> plot_sequence_single(seq)  # doctest: +SKIP

    >>> onsets = [0, 500, 1000, 1500]
    >>> plot_sequence_single(onsets)  # doctest: +SKIP

    """

    # Input validation, get onsets, set linewidths (the widths of the bars) and title.
    if isinstance(sequence, Sequence):
        # onsets
        onsets = sequence.onsets
        # linewidths
        if linewidths is None:
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
        if linewidths is None:
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
                           linewidths: int = None,
                           figsize: tuple = None,
                           suppress_display: bool = False):
    """

    Plot multiple sequences in one plot. Either pass it an iterable (e.g. list) of Sequence objects,
    StimSequence objects, or iterables of event onsets (so iterable of iterables, e.g. list of lists).

    If the total duration is shorter than 10 seconds, the x axis values are in milliseconds. If the total duration is longer
    than 10 seconds, the x axis values are in seconds.

    Parameters
    ----------
    sequences : iterable
        An iterable (e.g. list) of Sequence or StimSequence objects. Alternatively, one can provide a list of lists
        containing event onsets, for instance: [[0, 500, 1000], [0, 600, 800], [0, 400, 550]]
    style : str, optional
        A matplotlib style used for the plot. Defaults to 'seaborn'.
    sequence_names : iterable
        An iterable containing names for the sequences as strings. For instance ['Sequence 1', 'Sequence 2'] etc.
        Must be of the same length as the number of sequences passed.
        If no names are provided, defaults to numbering the sequences.
    title : str, optional
        If desired, one can provide a title for the plot. This takes precedence over using the
        Sequence or StimSequence name as the title of the plot (if passed and the object has one).
    linewidths : iterable, optional
        An iterable of ints (e.g. [50, 50, 50] ) or an iterable of iterables containing the desired width of the bars
        (event durations) in milliseconds, so e.g.: [[50, 30, 60], [20, 40, 10], [60, 30, 10]]
        If StimSequence objects are passed, the event durations are used.
        In other cases, a default of 50 milliseconds is used throughout.
    figsize : tuple, optional
        The desired figure size in inches as a tuple: (width, height).
    suppress_display : bool, optional
        If True, the plot is only returned, and not displayed via plt.show()

    Returns
    -------
    fig : Figure
        A matplotlib Figure object
    ax : Axes
        A matplotlib Axes object

    Examples
    --------
    >>> generator = np.random.default_rng(seed=123)
    >>> seqs = [Sequence.generate_random_normal(n=5, mu=5000, sigma=50, rng=generator) for _ in range(10)]
    >>> plot_sequence_multiple(seqs, linewidths=50)

    """

    # Input validation
    if not all(isinstance(sequence, Sequence) for sequence in sequences) and not all(
            isinstance(sequence, StimSequence) for sequence in sequences) and not all(
        hasattr(sequence, '__iter__') for sequence in sequences):
        raise ValueError("Please pass an iterable with only Sequence objects, StimSequence objects "
                         "or iterables containing the event onsets.")

    # Make names for the bars
    n_seqs = len(sequences)
    if sequence_names is None:
        sequence_names = [str(i) for i in range(1, n_seqs + 1)]
    elif len(sequence_names) != len(sequences):
        raise ValueError("Please provide an equal number of bar names as sequences.")

    # Get onsets
    if isinstance(sequences[0], (Sequence, StimSequence)):
        onsets = [sequence.onsets for sequence in sequences]
    else:
        onsets = sequences

    # Make line widths (these are either the event durations in case StimTrials were passed, in case of Sequences these
    # default to 50 milliseconds).
    if linewidths is None:
        if isinstance(sequences[0], StimSequence):
            linewidths = [trial.event_durations for trial in sequences]
        else:
            linewidths = [50] * len(sequences)
    elif not hasattr(linewidths, '__iter__'):
        linewidths = [linewidths] * len(sequences)
    else:
        raise ValueError("Please provide an iterable for the linewidths.")

    # Above 10s we want seconds on the x axis, otherwise milliseconds

    max_onset_ms = np.max(np.max(onsets))

    if max_onset_ms > 10000:
        onsets = np.array(onsets) / 1000
        linewidths = np.array(linewidths) / 1000
        x_label = "Time (s)"
    else:
        onsets = onsets
        x_label = "Time (ms)"

    # Plot
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        # Labels
        ax.axes.set_title(title)
        ax.axes.set_xlabel(x_label)

        for onsets, label, linewidths in zip(onsets, sequence_names, linewidths):
            ax.barh(y=label, width=linewidths, left=onsets)

        if not suppress_display:
            plt.show()

    return fig, ax
