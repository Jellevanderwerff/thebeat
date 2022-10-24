from __future__ import annotations
import thebeat.core
import matplotlib.pyplot as plt
from typing import Union, Optional
import thebeat.helpers
import numpy as np
from matplotlib.colors import ListedColormap
import scipy.spatial.distance


def plot_multiple_sequences(sequences: Union[list, np.ndarray],
                            style: str = 'seaborn',
                            title: Optional[str] = None,
                            x_axis_label: str = "Time",
                            y_axis_labels: Optional[list[str], np.ndarray[str]] = None,
                            linewidths: Optional[list[float], np.typing.NDArray[float], float] = None,
                            figsize: Optional[tuple] = None,
                            suppress_display: bool = False,
                            dpi: int = 100,
                            colors: Union[list, np.ndarray] = None) -> tuple[plt.Figure, plt.Axes]:
    """Plot multiple sequences in one plot. Either pass it a list or array of :py:class:`~thebeat.core.Sequence`
    objects, :py:class:`~thebeat.core.StimSequence` objects, or list or array of event onsets (so e.g. list of lists).

    Parameters
    ----------
    sequences
        A list or array of :py:class:`~thebeat.core.Sequence` or :py:class:`~thebeat.core.StimSequence` objects.
        Alternatively, one can provide e.g. a list of lists containing event onsets, for instance:
        ``[[0, 500, 1000], [0, 600, 800], [0, 400, 550]]``.
    style
        Matplotlib style to use for the plot. See `matplotlib style sheets reference
        <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    x_axis_label
        A label for the x axis.
    y_axis_labels
        A list or array containing names for the sequences as strings. For instance ``['Sequence 1', 'Sequence 2']``
        etc. Must be of the same length as the number of sequences passed. If no names are provided, defaults to
        numbering the sequences. This is because matplotlib needs a label there.
    title
        If desired, one can provide a title for the plot. This takes precedence over using the
        Sequence or StimSequence name as the title of the plot (if passed and the object has one).
    linewidths
        An array or list of ints (e.g. ``[50, 50, 50]`` ) or nested array or list containing the desired widths of the
        bars (event durations), for instance: ``[[50, 30, 60], [20, 40, 10], [60, 30, 10]]``.
        By default, if :py:class:`~thebeat.core.StimSequence` objects are passed, the event durations are used.
        In other cases, a default of 1/10th of the smallest IOI is used.
    figsize
        A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
        This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.figure`.
    suppress_display
        If ``True``, the plot is only returned, and not displayed via :func:`matplotlib.pyplot.show`.
    dpi
        The resolution of the plot in dots per inch. This refers to the ``dpi`` parameter in
        :func:`matplotlib.pyplot.figure`.
    colors
        A list or array of colors to use for the plot. If not provided, the default matplotlib colors are used.
        Colors may be provided as strings (e.g. ``'red'``) or as RGB tuples (e.g. ``(1, 0, 0)``).

    Examples
    --------
    >>> generator = np.random.default_rng(seed=123)
    >>> seqs = [thebeat.core.Sequence.generate_random_normal(n=5, mu=5000, sigma=50, rng=generator) for _ in range(10)]
    >>> plot_multiple_sequences(seqs,linewidths=50)  # doctest: +SKIP

    """

    onsets = []

    for seq in sequences:
        if isinstance(seq, (thebeat.core.Sequence, thebeat.core.StimSequence)):
            onsets.append(seq.onsets)
        else:
            onsets.append(np.array(seq))

    # Make names for the bars
    n_seqs = len(sequences)
    if y_axis_labels is None:
        y_axis_labels = ([str(i) for i in range(1, n_seqs + 1)])
    elif len(y_axis_labels) != len(sequences):
        raise ValueError("Please provide an equal number of bar names as sequences.")

    # Make line widths (these are either the event durations in case StimTrials were passed, in case of Sequences these
    # default to 50 milliseconds).
    if linewidths is None:
        if isinstance(sequences[0], thebeat.core.StimSequence):
            linewidths = [trial.event_durations for trial in sequences]
        else:
            all_iois = [sequence.iois for sequence in sequences]
            smallest_ioi = np.min(np.concatenate(all_iois))
            linewidths = [smallest_ioi / 10] * len(sequences)
    else:
        linewidths = [linewidths] * len(sequences)

    # Plot
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True, dpi=dpi)

        # Labels
        ax.axes.set_title(title)
        ax.axes.set_xlabel(x_axis_label)

        # Colors
        if colors:
            if len(colors) != len(sequences):
                raise ValueError("Please provide an equal number of colors as sequences.")
        else:
            colors = [None] * len(sequences)

        for onsets, label, linewidths, color in zip(reversed(onsets),
                                                    reversed(y_axis_labels),
                                                    reversed(linewidths),
                                                    reversed(colors)):
            ax.barh(y=label, width=linewidths, left=onsets, color=color)

        # Make sure we always have 0 on the left side of the x axis
        current_lims = ax.get_xlim()
        ax.set_xlim(0, current_lims[1])

        if not suppress_display:
            plt.show()

    return fig, ax


def recurrence_plot(sequence: thebeat.core.Sequence,
                    threshold: Optional[float] = None,
                    colorbar: bool = False,
                    colorbar_label: Optional[str] = "Distance",
                    cmap: Optional[str] = None,
                    style: str = 'seaborn',
                    title: Optional[str] = None,
                    x_axis_label: str = "IOI number",
                    y_axis_label: str = "IOI number",
                    figsize: tuple = (4, 4),
                    suppress_display: bool = False,
                    dpi: int = 100) -> tuple[plt.Figure, plt.Axes]:
    """

    Parameters
    ----------
    sequence
        A :py:class:`~thebeat.core.Sequence` object.
    threshold
        The threshold to use for the recurrence plot. If a threshold is given, the plot is binary,
        with color (e.g. black) representing the inter-onset intervals that are below the threshold.
        If no threshold is given, the plot is colored according to the distance between the inter-onset intervals.
    colorbar
        If ``True``, a colorbar is added to the plot.
    colorbar_label
        A label for the colorbar.
    cmap
        The colormap to use for the plot. See
        `matplotlib colormaps reference <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
        for the different options. For binary plots, the default is ``Greys``, for colored plots, the default is
        ``viridis``.
    style
        Matplotlib style to use for the plot. See `matplotlib style sheets reference
        <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    title
        If desired, one can provide a title for the plot.
    x_axis_label
        A label for the x axis.
    y_axis_label
        A label for the y axis.
    figsize
        A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
    suppress_display
        If ``True``, the plot is only returned, and not displayed via :func:`matplotlib.pyplot.show`.
    dpi
        The resolution of the plot in dots per inch.

    Examples
    --------
    >>> from thebeat.core import Sequence
    >>> from thebeat.visualization import recurrence_plot
    >>> seq = Sequence.generate_random_normal(n=3, mu=5000, sigma=50, metrical=True) * 10
    >>> recurrence_plot(seq)  # doctest: +SKIP

    >>> from thebeat.core import Sequence
    >>> from thebeat.visualization import recurrence_plot
    >>> seq = Sequence.generate_random_normal(n=3, mu=5000, sigma=50, metrical=True) * 10
    >>> fig, ax = recurrence_plot(seq, threshold=5, dpi=300, suppress_display=True)
    >>> fig.savefig('recurrence_plot.png')  # doctest: +SKIP

    Notes
    -----
    The binary recurrence plot is based on :footcite:t:`ravignaniMeasuringRhythmicComplexity2017`.
    The coloured recurrence plot is based on :footcite:t:`ravignaniMeasuringRhythmicComplexity2017`.

    """
    # Make title
    title = sequence.name if sequence.name else title

    # Calculate distance matrix
    iois = sequence.iois
    distance_matrix = np.abs(iois[:, None] - iois[None, :])

    # Make either 0's or 1's (if threshold) and set default cmaps
    if threshold:
        distance_matrix = (distance_matrix < threshold).astype(int)
        cmap = cmap if cmap else 'Greys'
    else:
        cmap = cmap if cmap else 'viridis'

    # Plot
    with plt.style.context(style):
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize, tight_layout=True)
        pcm = ax.pcolormesh(distance_matrix, cmap=cmap)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)

        if colorbar is True:
            fig.colorbar(pcm, ax=ax, label=colorbar_label)

    if not suppress_display:
        fig.show()

    return fig, ax
