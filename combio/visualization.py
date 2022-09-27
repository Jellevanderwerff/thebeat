from __future__ import annotations
import combio.core
import matplotlib.pyplot as plt
from typing import Union, Iterable, Optional
import combio._helpers
import numpy as np


def plot_single_sequence(sequence: Union[combio.core.Sequence, combio.core.StimSequence, list, np.ndarray],
                         style: str = 'seaborn',
                         title: Optional[str] = None,
                         linewidths: Optional[Union[list[int], np.ndarray[int], int]] = None,
                         figsize: Optional[tuple] = None,
                         suppress_display: bool = False):
    """
    This function may be used to plot a single sequence of event onsets.
    Either pass it a :py:class:`~combio.core.Sequence` or :py:class:`~combio.core.StimSequence` object,
    or a list or array of event onsets in milliseconds.

    This function is internally used by the :meth:`combio.core.Sequence.plot` and
    :meth:`combio.core.StimSequence.plot_sequence` methods.

    Parameters
    ----------
    sequence
        Either a :py:class:`~combio.core.Sequence` or :py:class:`~combio.core.StimSequence` object, or an list
        or an array of event onsets, e.g. ``[0, 500, 1000]``. If using a list or array, 0 is not required.
    style
        Matplotlib style to use for the plot. See `matplotlib style sheets reference
        <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    title
        If desired, one can provide a title for the plot. This takes precedence over using the
        StimSequence name as the title of the plot (if the object has one).
    linewidths
        A list or array containing the desired width of the bars (events) in milliseconds.
        If a single int is provided, this width is used for all the bars.
    figsize
        A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
        This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.figure`.
    suppress_display
        If ``True``, the plot is only returned, and not displayed via :func:`matplotlib.pyplot.show`.

    Examples
    --------
    >>> seq = combio.core.Sequence.generate_isochronous(n=5, ioi=500)
    >>> plot_single_sequence(seq)  # doctest: +SKIP

    >>> seq = [0, 500, 1000, 1500]
    >>> plot_single_sequence(seq)  # doctest: +SKIP

    """

    # Input validation, get onsets, set linewidths (the widths of the bars) and title.
    if isinstance(sequence, combio.core.Sequence):
        # onsets
        onsets = sequence.onsets
        # linewidths
        if linewidths is None:
            linewidths = [50] * len(onsets)
        # title
        if sequence.name and title is None:
            title = sequence.name

    elif isinstance(sequence, combio.core.StimSequence):
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

    # If single value linewidth is passed, use that for each event.
    if isinstance(linewidths, int):
        linewidths = [linewidths] * len(onsets)

    fig, ax = combio._helpers.plot_sequence_single(onsets=onsets, style=style, title=title, linewidths=linewidths,
                                                       figsize=figsize, suppress_display=suppress_display)
    return fig, ax


def plot_multiple_sequences(sequences: Union[list, np.ndarray],
                            style: str = 'seaborn',
                            sequence_names: Optional[list[str], np.ndarray[str]] = None,
                            title: Optional[str] = None,
                            linewidths: Optional[int] = None,
                            figsize: Optional[tuple] = None,
                            suppress_display: bool = False):
    """

    Plot multiple sequences in one plot. Either pass it a list or array of :py:class:`~combio.core.Sequence`
    objects, :py:class:`~combio.core.StimSequence` objects, or list or array of event onsets (so e.g. list of lists).

    If the total duration is shorter than 10 seconds, the `x` axis values are in milliseconds. If the total duration is
    longer than 10 seconds, the `x` axis values are in seconds.

    Parameters
    ----------
    sequences
        A list or array of :py:class:`~combio.core.Sequence` or :py:class:`~combio.core.StimSequence` objects.
        Alternatively, one can provide e.g. a list of lists containing event onsets, for instance:
        ``[[0, 500, 1000], [0, 600, 800], [0, 400, 550]]``.
    style
        Matplotlib style to use for the plot. See `matplotlib style sheets reference
        <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    sequence_names
        A list or array containing names for the sequences as strings. For instance ``['Sequence 1', 'Sequence 2']``
        etc. Must be of the same length as the number of sequences passed. If no names are provided, defaults to
        numbering the sequences.
    title
        If desired, one can provide a title for the plot. This takes precedence over using the
        Sequence or StimSequence name as the title of the plot (if passed and the object has one).
    linewidths
        An array or list of ints (e.g. ``[50, 50, 50]`` ) or nested array or list containing the desired widths of the
        bars (event durations) in milliseconds, for instance: ``[[50, 30, 60], [20, 40, 10], [60, 30, 10]]``.
        By default, if :py:class:`~combio.core.StimSequence` objects are passed, the event durations are used.
        In other cases, a default of 50 milliseconds is used throughout.
    figsize
        A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
        This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.figure`.
    suppress_display
        If ``True``, the plot is only returned, and not displayed via :func:`matplotlib.pyplot.show`.

    Examples
    --------
    >>> generator = np.random.default_rng(seed=123)
    >>> seqs = [combio.core.Sequence.generate_random_normal(n=5, mu=5000, sigma=50, rng=generator) for _ in range(10)]
    >>> plot_multiple_sequences(seqs,linewidths=50)

    """

    onsets = []

    for seq in sequences:
        if isinstance(seq, (combio.core.Sequence, combio.core.StimSequence)):
            onsets.append(seq.onsets)
        else:
            onsets.append(np.array(seq))

    # Make names for the bars
    n_seqs = len(sequences)
    if sequence_names is None:
        sequence_names = [str(i) for i in range(1, n_seqs + 1)]
    elif len(sequence_names) != len(sequences):
        raise ValueError("Please provide an equal number of bar names as sequences.")

    # Make line widths (these are either the event durations in case StimTrials were passed, in case of Sequences these
    # default to 50 milliseconds).
    if linewidths is None:
        if isinstance(sequences[0], combio.core.StimSequence):
            linewidths = [trial.event_durations for trial in sequences]
        else:
            linewidths = [50] * len(sequences)
    elif not hasattr(linewidths, '__iter__'):
        linewidths = [linewidths] * len(sequences)
    else:
        raise ValueError("Please provide an iterable for the linewidths.")

    # Get the highest value
    temp_onsets = np.concatenate(onsets)
    max_onsets_ms = np.max(np.max(temp_onsets))

    if max_onsets_ms > 10000:
        onsets = np.array(onsets) / 1000
        linewidths = np.array(linewidths) / 1000
        x_label = "Time (s)"
    else:
        onsets = onsets
        x_label = "Time (ms)"

    # Calculate x lims
    # The reasoning here is that linewidths is either an iterable containing integers
    # or an iterable containing iterables with individual linewidths

    # Plot
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)

        # Labels
        ax.axes.set_title(title)
        ax.axes.set_xlabel(x_label)

        for onsets, label, linewidths in zip(onsets, sequence_names, linewidths):
            ax.barh(y=label, width=linewidths, left=onsets)

        # Make sure we always have 0 on the left side of the x axis
        current_lims = ax.get_xlim()
        ax.set_xlim(0, current_lims[1])

        if not suppress_display:
            plt.show()

    return fig, ax
