from __future__ import annotations
import thebeat.core
import matplotlib.pyplot as plt
from typing import Union, Optional
import thebeat.helpers
import numpy as np
import matplotlib.ticker as ticker
import scipy.stats


def plot_interval_ratios_density(sequence: Union[thebeat.core.Sequence,
                                                 list[thebeat.core.Sequence],
                                                 np.ndarray[thebeat.core.Sequence]],
                                 resolution: float = 0.01,
                                 style: str = 'seaborn',
                                 title: Optional[str] = None,
                                 x_axis_label: str = "Interval ratios from dyads",
                                 y_axis_label: str = "Probability density",
                                 figsize: Optional[tuple[int, int]] = None,
                                 suppress_display: bool = False,
                                 dpi: int = 100,
                                 ax: Optional[plt.Axes] = None) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a density plot of the interval ratios from sequential dyads in a sequence.
    Input can either be a single sequence, or a list or array of sequences.

    This function internally uses :func:`thebeat.helpers.interval_ratios_from_dyads` to calculate
    the interval ratios.

    Parameters
    ----------
    sequence
        The sequence or list or array of sequences to plot.
    resolution
        The resolution of the density plot. At each point, the probability density is calculated.
    style
        The matplotlib style to use. See
        `matplotlib style reference <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    title
        A title for the plot.
    x_axis_label
        The label for the x axis.
    y_axis_label
        The label for the y axis.
    figsize
        The size of the figure to be created in inches, width x height, e.g. (4, 4).
    suppress_display
        If True, the figure will not be displayed.
    dpi
        The resolution of the figure in dots per inch.
    ax
        An optional *matplotlib* :class:`~matplotlib.axes.Axes` object to plot on. If not provided,
        a new Axes object will be created. If an :class:`~matplotlib.axes.Axes` object is provided,
        this function returns the original :class:`~matplotlib.axes.Figure` and
        :class:`~matplotlib.axes.Axes` objects.

    """

    interval_ratios = np.array([])

    if isinstance(sequence, (list, np.ndarray)):
        for seq in sequence:
            interval_ratios = np.append(interval_ratios, seq.interval_ratios_from_dyads)
    elif isinstance(sequence, thebeat.core.Sequence):
        interval_ratios = np.append(interval_ratios, sequence.interval_ratios_from_dyads)
    else:
        raise TypeError("'sequence' argument must be a Sequence object, list of Sequence objects, or a numpy array of "
                        "Sequence objects.")

    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax_provided = False
        else:
            fig = ax.get_figure()
            ax_provided = True

        # Get kernel density function
        kde = scipy.stats.gaussian_kde(interval_ratios)

        # Get x values
        x = np.arange(0, 1, resolution)

        # Get y values
        y = kde.evaluate(x)

        # Plot and set texts
        ax.plot(x, y)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)

    if not suppress_display and not ax_provided:
        fig.show()

    return fig, ax


def plot_interval_ratios_hist(sequence: Union[thebeat.core.Sequence,
                                              list[thebeat.core.Sequence], np.ndarray[thebeat.core.Sequence]],
                              bins: int = 100,
                              style: str = 'seaborn',
                              title: Optional[str] = None,
                              x_axis_label: str = "Interval ratios from dyads",
                              y_axis_label: str = "Count",
                              figsize: Optional[tuple[int, int]] = None,
                              suppress_display: bool = False,
                              dpi: int = 100,
                              ax: Optional[plt.Axes] = None) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a histogram of the interval ratios from sequential dyads in a sequence.
    Input can either be a single sequence, or a list or array of sequences.

    This function internally uses :func:`thebeat.helpers.interval_ratios_from_dyads` to calculate
    the interval ratios.

    Parameters
    ----------
    sequence
        The sequence or list or array of sequences to plot.
    bins
        The number of bins to use in the histogram.
    style
        The matplotlib style to use. See
        `matplotlib style reference <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    title
        A title for the plot.
    x_axis_label
        The label for the x axis.
    y_axis_label
        The label for the y axis.
    figsize
        The size of the figure to be created in inches, width x height, e.g. (4, 4).
    suppress_display
        If True, the figure will not be displayed.
    dpi
        The resolution of the figure in dots per inch.
    ax
        An optional *matplotlib* :class:`~matplotlib.axes.Axes` object to plot on. If not provided,
        a new Axes object will be created. If an :class:`~matplotlib.axes.Axes` object is provided,
        this function returns the original :class:`~matplotlib.axes.Figure` and
        :class:`~matplotlib.axes.Axes` objects.

    """

    interval_ratios = np.array([])

    if isinstance(sequence, (list, np.ndarray)):
        for seq in sequence:
            interval_ratios = np.append(interval_ratios, seq.interval_ratios_from_dyads)
    elif isinstance(sequence, thebeat.core.Sequence):
        interval_ratios = np.append(interval_ratios, sequence.interval_ratios_from_dyads)
    else:
        raise TypeError("'sequence' argument must be a Sequence object, list of Sequence objects, or a numpy array of "
                        "Sequence objects.")

    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax_provided = False
        else:
            fig = ax.get_figure()
            ax_provided = True

        ax.hist(interval_ratios, bins=bins)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_xlim(0, 1)
        ax.set_title(title)

    if not suppress_display and not ax_provided:
        fig.show()

    return fig, ax


def plot_phase_differences(test_sequence: Union[thebeat.core.Sequence,
                                                list[thebeat.core.Sequence],
                                                np.ndarray[thebeat.core.Sequence]],
                           reference_sequence: Union[thebeat.core.Sequence,
                                                     float,
                                                     list[thebeat.core.Sequence],
                                                     np.ndarray[thebeat.core.Sequence]],
                           binwidth: int = 10,
                           zero_direction: str = 'N',
                           color: str = None,
                           style: str = 'seaborn',
                           title: Optional[str] = None,
                           figsize: Optional[tuple[int, int]] = None,
                           suppress_display: bool = False,
                           dpi: int = 100,
                           ax: Optional[plt.Axes] = None) -> tuple[plt.Figure, plt.Axes]:
    """Plot the phase differences for ``test_sequence`` compared to ``sequence_2``. ``test_sequence`` can be a single
    sequence, a list of sequences, or a numpy array of sequences.

    The reference sequence can either be a single sequence in which case a comparison will be made between
    each test sequence and the reference sequence, or a float in which case a comparison will be made between
    each test sequence and an isochronous sequence with the provided IOI, or it can be an array or list
    of sequences, in which case a comparison is made between each test and reference sequence element-wise.

    Example
    -------

    .. figure:: images/phase_differences_plot.png
        :width: 100 %

        Example phase differences plot.

    Parameters
    ----------
    test_sequence
        The sequence or sequences to be compared to ``reference_sequence``. Can be a single
        :py:class:`thebeat.core.Sequence` object, or a list or array of :py:class:`thebeat.core.Sequence` objects.
    reference_sequence
        The reference sequence or sequences to be compared to ``test_sequence``. Can be a single
        :py:class:`thebeat.core.Sequence` object, or a list or array of :py:class:`thebeat.core.Sequence` objects.
        If both the test_sequence and reference sequences are lists or arrays, they must be of the same length.
    binwidth
        The width of the bins used to calculate the histogram bars.
    zero_direction
        The direction of the zero angle. Can be 'N', 'E', 'S', or 'W'.
    color
        The color of the bars. Can be a single color or a list of colors, one for each bar.
        See :meth:`matplotlib.axes.Axes.bar` for more information.
    style
        The matplotlib style to use. See
        `matplotlib style reference <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    title
        A title for the plot.
    figsize
        The size of the figure to be created in inches.
    suppress_display
        If True, the figure will not be displayed.
    dpi
        The resolution of the figure in dots per inch.
    ax
        An optional :py:class:`matplotlib.axes.Axes` object to plot on. If not provided, a new Axes object will be
        created. If an :py:class:`matplotlib.axes.Axes` object is provided, the function returns the original
        :py:class:`matplotlib.axes.Figure` and :py:class:`matplotlib.axes.Axes` objects.

    """

    # Input validation for test sequence
    if isinstance(test_sequence, (list, np.ndarray)):
        test_iterable_passed = True
    else:
        test_iterable_passed = False

    # Input validation for reference sequence
    if isinstance(reference_sequence, (list, np.ndarray)):
        ref_iterable_passed = True
    else:
        ref_iterable_passed = False

    # If we have lists on both sides, they must be of equal length
    if isinstance(test_sequence, (list, np.ndarray)) and isinstance(reference_sequence, (list, np.ndarray)):
        if len(test_sequence) != len(reference_sequence):
            raise ValueError("The test and reference sequences must be the same length.")

    # Output array
    phase_diffs = np.array([])

    # Do all the different combinations of input types
    if test_iterable_passed:
        for i, test_seq in enumerate(test_sequence):
            if ref_iterable_passed:
                # we have list of test sequences and list of ref sequences
                ref_seq = reference_sequence[i]
            else:
                # we have list of test sequences and single ref sequence
                ref_seq = reference_sequence

            phase_diffs = np.append(phase_diffs, thebeat.utils.get_phase_differences(test_seq, ref_seq))

    else:
        if ref_iterable_passed:
            for ref_seq in reference_sequence:
                phase_diffs = np.append(phase_diffs, thebeat.utils.get_phase_differences(test_sequence, ref_seq))
        else:
            # we have a single test sequence and a single ref sequences
            phase_diffs = np.append(phase_diffs,
                                    thebeat.utils.get_phase_differences(test_sequence, reference_sequence))

    # Calculate the bins
    a, b = np.histogram(phase_diffs, bins=np.arange(0, 360 + binwidth, binwidth))
    centers = np.deg2rad(np.ediff1d(b) // 2 + b[:-1])

    # Plot the histogram
    with plt.style.context(style):
        # If no Axes was provided, create one
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={'projection': 'polar'})
            axes_provided = False
        # If an Axes object was provided, use that one but check if it is a polar one
        else:
            fig = ax.get_figure()
            if not ax.name == 'polar':
                raise ValueError("Please provide a polar Axes object. Use projection='polar' when creating it.")
            axes_provided = True
        ax.bar(centers, a, width=np.deg2rad(binwidth), bottom=0.0, color=color, alpha=0.5)
        ax.set_title(title)

        # Make the plot face north
        ax.set_theta_zero_location(zero_direction)
        ax.set_theta_direction(-1)

    # Show
    if not suppress_display and axes_provided is False:
        fig.show()

    return fig, ax


def phase_space_plot(sequence: thebeat.core.Sequence,
                     style: str = 'seaborn',
                     linecolor: str = 'black',
                     linewidth: float = 0.5,
                     title: Optional[str] = None,
                     x_axis_label: str = r"$\mathregular{IOI_i}$",
                     y_axis_label: str = r"$\mathregular{IOI_{i+1}}$",
                     figsize: Optional[tuple[int, int]] = None,
                     suppress_display: bool = False,
                     dpi: int = 100,
                     ax: Optional[plt.Axes] = None) -> tuple[plt.Figure, plt.Axes]:
    """Plot the phase space of a sequence. In such a plot we loop over each IOI, and plot a line
    between it on the x axis and the IOI that follows it on the y axis.

    Parameters
    ----------
    sequence
        The sequence to plot.
    style
        The matplotlib style to use. See
        `matplotlib style reference <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    linecolor
        The color of the lines.
    linewidth
        The width of the lines.
    title
        A title for the plot.
    x_axis_label
        The label for the x axis.
    y_axis_label
        The label for the y axis.
    figsize
        The size of the figure to be created in inches, width x height, e.g. (4, 4).
    suppress_display
        If True, the figure will not be displayed.
    dpi
        The resolution of the figure in dots per inch.
    ax
        An optional *matplotlib* :class:`~matplotlib.axes.Axes` object to plot on. If not provided,
        a new Axes object will be created. If an :class:`~matplotlib.axes.Axes` object is provided,
        this function returns the original :class:`~matplotlib.axes.Figure` and
        :class:`~matplotlib.axes.Axes` objects.

    Example
    -------

    .. figure:: images/phase_space_plot.png

        Example of a phase space plot.

    Note
    ----
    Code adapted from :footcite:t:`ravignaniMeasuringRhythmicComplexity2017`.

    Examples
    --------
    >>> from thebeat import Sequence
    >>> seq = Sequence.generate_random_normal(100, 100, 10)
    >>> phase_space_plot(seq)  # doctest: +SKIP

    """

    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi, tight_layout=True)
            ax_provided = False
        else:
            fig = ax.get_figure()
            ax_provided = True
        iois = sequence.iois
        for i in range(len(iois) - 2):
            ax.plot([iois[i], iois[i + 1]], [iois[i + 1], iois[i + 2]],
                    color=linecolor,
                    linewidth=linewidth)

        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
        ax.set_aspect('equal')

    if not suppress_display and not ax_provided:
        fig.show()

    return fig, ax


def plot_multiple_sequences(sequences: Union[list, np.ndarray],
                            style: str = 'seaborn',
                            title: Optional[str] = None,
                            x_axis_label: str = "Time",
                            sequence_labels: Optional[list[str], np.ndarray[str]] = None,
                            linewidths: Optional[list[float], np.typing.NDArray[float], float] = None,
                            figsize: Optional[tuple] = None,
                            suppress_display: bool = False,
                            dpi: int = 100,
                            colors: Union[list, np.ndarray] = None,
                            ax: Optional[plt.Axes] = None) -> tuple[plt.Figure, plt.Axes]:
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
    sequence_labels
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
    ax
        If desired, you can provide an existing :class:`matplotlib.Axes` object onto which to plot.
        See the Examples of the different plotting functions to see how to do this
        (e.g. :py:meth:`~thebeat.core.Sequence.plot_sequence` ).
        If an existing :class:`matplotlib.Axes` object is supplied, this function returns the original
        :class:`matplotlib.Figure` and :class:`matplotlib.Axes` objects.

    Examples
    --------
    >>> from thebeat.core import Sequence
    >>> generator = np.random.default_rng(seed=123)
    >>> seqs = [Sequence.generate_random_normal(n=5, mu=5000, sigma=50, rng=generator) for _ in range(10)]
    >>> plot_multiple_sequences(seqs,linewidths=50)  # doctest: +SKIP

    >>> seq1 = Sequence([500, 100, 200])
    >>> seq2 = Sequence([100, 200, 500])
    >>> fig, ax = plot_multiple_sequences([seq1, seq2], colors=['red', 'blue'])  # doctest: +SKIP
    >>> fig.savefig('test.png')  # doctest: +SKIP

    """

    onsets = []

    for seq in sequences:
        if isinstance(seq, (thebeat.core.Sequence, thebeat.core.StimSequence)):
            onsets.append(seq.onsets)
        else:
            onsets.append(np.array(seq))

    # Make names for the bars
    n_seqs = len(sequences)
    if sequence_labels is None:
        if all(sequence.name for sequence in sequences):
            sequence_labels = [sequence.name for sequence in sequences]
        else:
            sequence_labels = ([str(i) for i in range(1, n_seqs + 1)])
    elif len(sequence_labels) != len(sequences):
        raise ValueError("Please provide an equal number of bar names as sequences.")

    # Make line widths (these are either the event durations in case StimTrials were passed, in case of Sequences these
    # default to 1/10th of the smallest IOI)
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
        # If an existing Axes object was passed, do not create new Figure and Axes.
        # Else, only create a new Figure object (Then,)
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True, dpi=dpi)
            ax_provided = False
        else:
            fig = ax.get_figure()
            ax_provided = True

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
                                                    reversed(sequence_labels),
                                                    reversed(linewidths),
                                                    reversed(colors)):
            ax.barh(y=label, width=linewidths, left=onsets, color=color)

        # Make sure we always have 0 on the left side of the x axis
        current_lims = ax.get_xlim()
        ax.set_xlim(0, current_lims[1])

    # Show plot if desired, and if no existing Axes object was passed.
    if suppress_display is False and ax_provided is False:
        fig.show()

    return fig, ax


def recurrence_plot(sequence: thebeat.core.Sequence,
                    threshold: Optional[float] = None,
                    colorbar: bool = False,
                    colorbar_label: Optional[str] = "Distance",
                    cmap: Optional[str] = None,
                    style: str = 'seaborn',
                    title: Optional[str] = None,
                    x_axis_label: str = "$\mathregular{N_i}$",
                    y_axis_label: str = "$\mathregular{N_i}$",
                    figsize: tuple = (5, 4),
                    suppress_display: bool = False,
                    dpi: int = 100,
                    ax: Optional[plt.Axes] = None) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a recurrence plot of a sequence. A recurrence plot is a plot with the IOI numbers (i.e. their indices) on the
    x and y axis, and the distance between the IOIs on the color scale. For each combination of two IOIs,
    the distance between these IOIs is calculated as their absolute difference (which may for instance
    be in seconds or milliseconds, depending on your input during construction of the :py:class:`Sequence` object).

    If you provide a ``threshold``, the plot will be binary, where color indicates anything below the threshold,
    and where white indicates anything above the threshold.

    Example
    -------

    .. figure:: images/recurrence_plot.png
        :width: 85 %

        Example recurrence plot with ``colorbar=True``.

    Parameters
    ----------
    sequence
        A :py:class:`~thebeat.core.Sequence` object.
    threshold
        The threshold to use for the recurrence plot. If a threshold is given, the plot is binary,
        with color (e.g. black) representing the inter-onset intervals that are below the threshold.
        If no threshold is given, the plot is colored according to the distance between the inter-onset intervals.
    colorbar
        If ``True``, a colorbar is added to the plot. Note that no colorbar is plotted when an existing
        Axes object is provided.
    colorbar_label
        A label for the colorbar.
    cmap
        The colormap to use for the plot. See
        `matplotlib colormaps reference <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
        for the different options. For binary plots, the default is ``'Greys'``, for colored plots, the default is
        ``'viridis'``.
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
    ax
        If desired, you can provide an existing :class:`matplotlib.Axes` object onto which to plot.
        See the Examples of the different plotting functions to see how to do this
        (e.g. :py:meth:`~thebeat.core.Sequence.plot_sequence` ).
        If an existing :class:`matplotlib.Axes` object is supplied, this function returns the original
        :class:`matplotlib.Figure` and :class:`matplotlib.Axes` objects.

    Examples
    --------

    >>> from thebeat.core import Sequence
    >>> from thebeat.visualization import recurrence_plot
    >>> seq = Sequence.generate_random_normal(n=3, mu=5000, sigma=50, metrical=True) * 10

    # No color bar, no threshold
    >>> recurrence_plot(seq)  # doctest: +SKIP

    # Color bar, no threshold
    >>> fig, ax = recurrence_plot(seq, dpi=300, colorbar=True)
    >>> fig.savefig('recurrence_plot.png', bbox_inches='tight')  # doctest: +SKIP

    >>> seq = Sequence.generate_random_normal(n=3, mu=5000, sigma=50, metrical=True) * 10
    >>> fig, ax = recurrence_plot(seq, threshold=5, dpi=300, suppress_display=True)
    >>> fig.savefig('recurrence_plot.png')  # doctest: +SKIP

    Notes
    -----
    The binary recurrence plot is based on :footcite:t:`ravignaniMeasuringRhythmicComplexity2017`.
    The coloured recurrence plot is based on :footcite:t:`burchardtNovelIdeasFurther2021`.

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
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True, dpi=dpi)
            ax_provided = False
        else:
            fig = ax.get_figure()
            ax_provided = True

        pcm = ax.pcolormesh(distance_matrix, cmap=cmap)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
        ax.set_aspect('equal')

        if colorbar is True:
            fig.colorbar(pcm, ax=ax, label=colorbar_label)

    # Show plot if desired, and if no existing Axes object was passed.
    if suppress_display is False and ax_provided is False:
        fig.show()

    return fig, ax
