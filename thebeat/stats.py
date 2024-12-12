# Copyright (C) 2022-2023  Jelle van der Werff
#
# This file is part of thebeat.
#
# thebeat is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# thebeat is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with thebeat.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import numbers

import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.fft
import scipy.signal
import scipy.stats
from scipy.fft import rfft, rfftfreq

import thebeat.core
import thebeat.helpers


def acf_df(
    sequence: thebeat.core.Sequence,
    resolution,
    smoothing_window: float | None = None,
    smoothing_sd: float | None = None,
) -> pd.DataFrame:
    """

    Perform autocorrelation analysis on a :py:class:`Sequence` object,
    and return a :class:`Pandas.DataFrame` object containing the results.

    Parameters
    ----------

    sequence
        A :py:class:`~thebeat.core.Sequence` object.
    resolution
        The temporal resolution. If the used Sequence is in seconds, you might want to use 0.001.
        If the Sequence is in milliseconds, try using 1. Incidentally, the number of lags
        for the autocorrelation function is calculated as
        ``n_lags = sequence_duration / resolution``.
    smoothing_window
        The window within which a normal probability density function is used for
        smoothing out the analysis.
    smoothing_sd
        The standard deviation of the normal probability density function used for smoothing out the analysis.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame containing two columns: the timestamps, and the autocorrelation factor.

    Notes
    -----
    This function is based on the procedure described in :cite:t:`ravignaniMeasuringRhythmicComplexity2017`.
    There, one can also find a more detailed description of the smoothing procedure.

    Examples
    --------
    >>> rng = np.random.default_rng(seed=123)  # for reproducability
    >>> seq = thebeat.core.Sequence.generate_random_uniform(n_events=10,a=400,b=600,rng=rng)
    >>> df = acf_df(seq, smoothing_window=50, smoothing_sd=20, resolution=10)
    >>> print(df.head(3))
       timestamp  correlation
    0          0     1.000000
    1         10     0.824706
    2         20     0.547863
    """

    correlations = acf_values(
        sequence=sequence,
        resolution=resolution,
        smoothing_window=smoothing_window,
        smoothing_sd=smoothing_sd,
    )
    correlations = correlations / np.max(correlations)
    timestamps = np.arange(start=0, stop=len(correlations) * resolution, step=resolution)

    df = pd.DataFrame({"timestamp": timestamps, "correlation": correlations})

    return df


def acf_plot(
    sequence: thebeat.core.Sequence,
    resolution,
    min_lag: float | None = None,
    max_lag: float | None = None,
    smoothing_window: float | None = None,
    smoothing_sd: float | None = None,
    style: str = "seaborn-v0_8",
    title: str = "Autocorrelation",
    x_axis_label: str = "Lag",
    y_axis_label: str = "Correlation",
    figsize: tuple | None = None,
    dpi: int = 100,
    ax: plt.Axes | None = None,
    suppress_display: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    This function can be used for plotting an autocorrelation plot from a :class:`~thebeat.core.Sequence`.

    Parameters
    ----------
    sequence
        A :py:class:`~thebeat.core.Sequence` object.
    resolution
        The temporal resolution. If the used Sequence is in seconds, you might want to use 0.001.
        If the Sequence is in milliseconds, try using 1. Incidentally, the number of lags
        for the autocorrelation function is calculated as
        ``n_lags = sequence_duration_in_ms / resolution``.
    min_lag
        The minimum lag to be plotted. Defaults to 0.
    max_lag
        The maximum lag to be plotted. Defaults to the sequence duration.
    smoothing_window
        The window (in milliseconds) within which a normal probability density function is used for
        smoothing out the analysis.
    smoothing_sd
        The standard deviation of the normal probability density function used for smoothing out the analysis.
    style
        Style used by matplotlib. See `matplotlib style sheets reference
        <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    title
        If desired, one can provide a title for the plot. This takes precedence over using the
        :class:`~thebeat.core.Sequence` or :class:`~thebeat.core.SoundSequence` ``name`` attribute as the title of the
        plot (if the object has one).
    x_axis_label
        A label for the x axis.
    y_axis_label
        A label for the y axis.
    figsize
        A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
        This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.figure`.
    dpi
        The desired output resolution of the plot in dots per inch (DPI). This refers to the ``dpi`` parameter
        in :func:`matplotlib.pyplot.figure`.
    ax
        If desired, one can provide an existing :class:`matplotlib.axes.Axes` object to plot the autocorrelation
        plot on. This is for instance useful if you want to plot multiple autocorrelation plots on the same figure.
    suppress_display
        If ``True``, :func:`matplotlib.pyplot.show` is not run.

    Notes
    -----
    This function is based on the procedure described in :cite:t:`ravignaniMeasuringRhythmicComplexity2017`.
    There, one can also find a more detailed description of the smoothing procedure.

    """

    onsets = sequence.onsets

    correlation = acf_values(
        sequence=sequence,
        resolution=resolution,
        smoothing_window=smoothing_window,
        smoothing_sd=smoothing_sd,
    )

    x_step = resolution
    min_lag = int(min_lag // resolution) if min_lag else 0
    max_lag = (
        int(max_lag // resolution) if max_lag else np.floor(np.max(onsets) / resolution).astype(int)
    )

    # plot
    try:
        y = correlation[min_lag:max_lag]
        y = y / np.max(y)  # normalize
    except ValueError:
        raise ValueError("We end up with an empty y axis. Try changing the resolution.")

    # Make x axis
    x = np.arange(start=min_lag, stop=max_lag * x_step, step=x_step)

    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True, dpi=dpi)
        else:
            fig = ax.get_figure()

        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
        ax.plot(x, y)

    if not suppress_display and ax is not None:
        plt.show()

    return fig, ax


def acf_values(
    sequence: thebeat.core.Sequence,
    resolution,
    smoothing_window: float | None = None,
    smoothing_sd: float | None = None,
) -> np.ndarray:
    """

    Perform autocorrelation. This function takes a :class:`~thebeat.core.Sequence` object, and returns an array with
    steps of ``resolution`` of unstandardized correlation factors.

    Parameters
    ----------
    sequence
        A :class:`~thebeat.core.Sequence` object.
    resolution
        The temporal resolution. If the used Sequence is in seconds, you might want to use 0.001.
        If the Sequence is in milliseconds, try using 1. Incidentally, the number of lags
        for the autocorrelation function is calculated as
        ``n_lags = sequence_duration / resolution``.
    smoothing_window
        The window within which a normal probability density function is used for
        smoothing out the analysis.
    smoothing_sd
        The standard deviation of the normal probability density function used for smoothing out the analysis.

    Notes
    -----
    This function is based on the procedure described in :cite:t:`ravignaniMeasuringRhythmicComplexity2017`. There,
    one can also find a more detailed description of the smoothing procedure.

    This function uses the :func:`numpy.correlate` to calculate the correlations.

    """

    signal = thebeat.helpers.sequence_to_binary(sequence, resolution)

    # npdf
    if smoothing_window and smoothing_sd:
        # Here we make a tiny normal distribution, which has the width of the smoothing window
        x = np.arange(start=-smoothing_window / 2, stop=smoothing_window / 2, step=resolution)
        npdf = scipy.stats.norm.pdf(x, 0, smoothing_sd)
        npdf = npdf / np.max(npdf)
        # Convolve tiny normal distributions with the original signal
        signal = np.convolve(signal, npdf, "same")

    try:
        correlation = np.correlate(signal, signal, "full")
    except ValueError as e:
        raise ValueError(
            "Error! Hint: Most likely your resolution is too large for the chosen smoothing_window"
            "and smoothing_sd. Try choosing a smaller resolution."
        ) from e

    # Now, we remove the negative lags (which have the length of signal)
    correlation = correlation[len(signal) - 1:]

    return correlation


def ccf_df(
    test_sequence: thebeat.core.Sequence,
    reference_sequence: thebeat.core.Sequence,
    resolution,
    smoothing_window: float | None = None,
    smoothing_sd: float | None = None,
) -> pd.DataFrame:
    """

    Perform autocorrelation analysis on a :py:class:`Sequence` object,
    and return a :class:`Pandas.DataFrame` object containing the results.

    Parameters
    ----------

    test_sequence
        The test sequence.
    reference_sequence
        The reference sequence.
    resolution
        The temporal resolution. If the used Sequence is in seconds, you might want to use 0.001.
        If the Sequence is in milliseconds, try using 1. Incidentally, the number of lags
        for the autocorrelation function is calculated as
        ``n_lags = sequence_duration / resolution``.
    smoothing_window
        The window within which a normal probability density function is used for
        smoothing out the analysis.
    smoothing_sd
        The standard deviation of the normal probability density function used for smoothing out the analysis.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame containing two columns: the timestamps, and the cross-correlation factor.

    Notes
    -----
    This function is based on the procedure described in :cite:t:`ravignaniMeasuringRhythmicComplexity2017`.
    There, one can also find a more detailed description of the smoothing procedure.


    """

    correlations = ccf_values(
        test_sequence=test_sequence,
        reference_sequence=reference_sequence,
        resolution=resolution,
        smoothing_window=smoothing_window,
        smoothing_sd=smoothing_sd,
    )
    # normalize
    correlations = correlations / np.max(correlations)
    timestamps = np.arange(start=0, stop=len(correlations) * resolution, step=resolution)

    df = pd.DataFrame({"timestamp": timestamps, "correlation": correlations})

    return df


def ccf_plot(
    test_sequence: thebeat.core.Sequence,
    reference_sequence: thebeat.core.Sequence,
    resolution,
    smoothing_window: float | None = None,
    smoothing_sd: float | None = None,
    style: str = "seaborn-v0_8",
    title: str = "Cross-correlation",
    x_axis_label: str = "Lag",
    y_axis_label: str = "Correlation",
    figsize: tuple | None = None,
    dpi: int = 100,
    ax: plt.Axes | None = None,
    suppress_display: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Calculate and plot the cross-correlation function (CCF) between two :class:`~thebeat.core.Sequence` objects.
    The test sequence is compared to the reference sequence.

    Parameters
    ----------
    test_sequence
        The test sequence.
    reference_sequence
        The reference sequence.
    resolution
        The temporal resolution. If the used Sequence is in milliseconds, you probably want 1. If the Sequence is in
        seconds, try using 0.001.
    smoothing_window
        The window within which a normal probability density function is used for
        smoothing out the analysis.
    smoothing_sd
        The standard deviation of the normal probability density function used for smoothing out the analysis.
    style
        The matplotlib style to use. See
        `matplotlib style reference <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    title
        The title of the plot.
    x_axis_label
        The label of the x axis.
    y_axis_label
        The label of the y axis.
    figsize
        A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
        This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.figure`.
    dpi
        The resolution of the plot in dots per inch. This refers to the ``dpi`` parameter in
        :func:`matplotlib.pyplot.figure`.
    ax
        A :class:`matplotlib.axes.Axes` object. If ``None``, a new Figure and Axes is created.
    suppress_display
        If ``True``, the plot is not displayed. This is useful e.g. if you only want to save the plot to a file

    Returns
    -------
    fig
        The :class:`matplotlib.figure.Figure` object.
    ax
        The :class:`matplotlib.axes.Axes` object.

    Notes
    -----
    This function is based on the procedure described in :cite:t:`ravignaniMeasuringRhythmicComplexity2017`. There,
    one can also find a more detailed description of the smoothing procedure.

    """

    # Get correlation factors
    correlation = ccf_values(
        test_sequence=test_sequence,
        reference_sequence=reference_sequence,
        resolution=resolution,
        smoothing_window=smoothing_window,
        smoothing_sd=smoothing_sd,
    )

    # Make y axis
    x_step = resolution
    max_lag = np.floor(
        np.max(np.concatenate([test_sequence.onsets, reference_sequence.onsets])) / resolution
    ).astype(int)
    try:
        y = correlation[:max_lag]
        y = y / np.max(y)  # normalize
    except ValueError:
        raise ValueError("We end up with an empty y axis. Try changing the resolution.")

    # Make x axis
    x = np.arange(start=0, stop=len(y) * x_step, step=x_step)

    # Plot
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True, dpi=dpi)
        else:
            fig = ax.get_figure()

        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_title(title)
        ax.plot(x, y)

    if not suppress_display and ax is not None:
        plt.show()

    return fig, ax


def ccf_values(
    test_sequence: thebeat.core.Sequence,
    reference_sequence: thebeat.core.Sequence,
    resolution: float,
    smoothing_window: float | None = None,
    smoothing_sd: float | None = None,
) -> np.ndarray:
    """
    Returns the unstandardized cross-correlation function (CCF) for two :class:`~thebeat.core.Sequence` objects.
    The test sequence is compared to the reference sequence.

    Parameters
    ----------
    test_sequence
        The test sequence.
    reference_sequence
        The reference sequence.
    resolution
        The temporal resolution. If the used Sequence is in milliseconds, you probably want 1. If the Sequence is in
        seconds, try using 0.001.
    smoothing_window
        The window within which a normal probability density function is used for
        smoothing out the analysis.
    smoothing_sd
        The standard deviation of the normal probability density function used for smoothing out the analysis.

    Returns
    -------
    correlation
        The unstandardized cross-correlation function.

    """

    # Make into 0's and 1's
    test_signal = thebeat.helpers.sequence_to_binary(test_sequence, resolution)
    ref_signal = thebeat.helpers.sequence_to_binary(reference_sequence, resolution)

    # npdf
    if smoothing_window and smoothing_sd:
        # Here we make a tiny normal distribution, which has the width of the smoothing window
        x = np.arange(start=-smoothing_window / 2, stop=smoothing_window / 2, step=resolution)
        npdf = scipy.stats.norm.pdf(x, 0, smoothing_sd)
        npdf = npdf / np.max(npdf)
        # Convolve tiny normal distributions with the original signal
        test_signal = np.convolve(test_signal, npdf, "same")
        ref_signal = np.convolve(ref_signal, npdf, "same")

    # Calculate cross-correlation
    try:
        correlation = np.correlate(test_signal, ref_signal, "full")
    except ValueError as e:
        raise ValueError(
            "Error! Hint: Most likely your resolution is too large for the chosen smoothing_window"
            "and smoothing_sd. Try choosing a smaller resolution."
        ) from e

    # Now, remove negative lags
    correlation = correlation[len(test_signal) - 1:]

    return correlation


def edit_distance_rhythm(
    test_rhythm: thebeat.music.Rhythm,
    reference_rhythm: thebeat.music.Rhythm,
    smallest_note_value: int = 16,
) -> float:
    """
    Caculates edit/Levenshtein distance between two rhythms. The ``smallest_note_value`` determines
    the underlying grid that is used. If e.g. 16, the underlying grid is composed of 1/16th notes.

    Note
    ----
    Based on the procedure described in :cite:t:`postEditDistanceMeasure2011`.

    Parameters
    ----------
    test_rhythm
        The rhythm to be tested.
    reference_rhythm
        The rhythm to which ``test_rhythm`` will be compared.
    smallest_note_value
        The smallest note value that is used in the underlying grid. 16 means 1/16th notes, 4 means 1/4th notes, etc.

    Examples
    --------
    >>> from thebeat.music import Rhythm
    >>> test_rhythm = Rhythm.from_fractions([1/4, 1/4, 1/4, 1/4])
    >>> reference_rhythm = Rhythm.from_fractions([1/4, 1/8, 1/8, 1/4, 1/4])
    >>> print(edit_distance_rhythm(test_rhythm, reference_rhythm))
    1

    """
    if not isinstance(test_rhythm, thebeat.music.Rhythm) or not isinstance(
        reference_rhythm, thebeat.music.Rhythm
    ):
        raise TypeError("test_rhythm and reference_rhythm must be of type Rhythm")

    test_string = thebeat.helpers.rhythm_to_binary(
        rhythm=test_rhythm, smallest_note_value=smallest_note_value
    )
    reference_string = thebeat.helpers.rhythm_to_binary(
        rhythm=reference_rhythm, smallest_note_value=smallest_note_value
    )

    # calculate edit distance
    edit_distance = Levenshtein.distance(test_string, reference_string)

    return edit_distance


def edit_distance_sequence(
    test_sequence: thebeat.core.Sequence, reference_sequence: thebeat.core.Sequence, resolution: int
) -> float:
    """
    Calculates the edit/Levenshtein distance between two sequences.

    Requires for all the IOIs in a Sequence to be multiples of 'resolution'. If needed, quantize the
    Sequence first, e.g. using Sequence.quantize_iois().

    Note
    ----
    The resolution also represents the underlying grid. If, for example, the resolution is 50, that means that
    a grid will be created with steps of 50. The onsets of the sequence are then placed on the grid for both
    sequences. The resulting sequences consist of ones and zeros, where ones represent the event onsets. This string
    for ``test_sequence`` is compared to the string of the ``reference_sequence``. Note that ``test_sequence`` and
    ``reference_sequence`` can be interchanged without an effect on the results.

    Parameters
    ----------
    test_sequence
        The sequence to be tested.
    reference_sequence
        The sequence to which ``test_sequence`` will be compared.
    resolution
        The used resolution (i.e. bin size).

    """
    if not isinstance(test_sequence, thebeat.core.Sequence) or not isinstance(
        reference_sequence, thebeat.core.Sequence
    ):
        raise TypeError("test_sequence and reference_sequence must be of type Sequence")

    # Check whether we need to quantize the sequences
    if np.any(test_sequence.onsets % resolution != 0) or np.any(
        reference_sequence.onsets % resolution != 0
    ):
        raise ValueError(
            "test_sequence and reference_sequence must be quantized to multiples of {resolution} first,"
            "for instance using Sequence.quantize_iois()"
        )

    test_string = thebeat.helpers.sequence_to_binary(sequence=test_sequence, resolution=resolution)
    reference_string = thebeat.helpers.sequence_to_binary(
        sequence=reference_sequence, resolution=resolution
    )

    # calculate edit distance
    edit_distance = Levenshtein.distance(test_string, reference_string)

    return edit_distance


def fft_values(
    sequence: thebeat.core.Sequence,
    unit_size: float,
    x_min: float | None = None,
    x_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gets the x and y values for a Fourier transform of a :class:`~thebeat.core.Sequence` object.
    The ``unit_size`` parameter is required, because Sequence objects are agnostic about the used time unit.
    You can use 1000 if the Sequence is in milliseconds, and 1 if the Sequence is in seconds.

    The x values indicate the number of cycles per unit, and y values the absolute power.
    The number of cycles per unit can be interpreted as the beat frequency. For instance, 2 cycles for a unit size of
    1000 ms means a beat frequency of 2 Hz.

    Parameters
    ----------
    sequence
        The sequence.
    unit_size
        The size of the unit in which the sequence is measured. If the sequence is in milliseconds,
        you probably want 1000. If the sequence is in seconds, you probably want 1.

    Returns
    -------
    xf
        The x values.
    yf
        The y values. Note that absolute values are returned.

    Examples
    --------
    >>> from thebeat import Sequence
    >>> from thebeat.stats import fft_values
    >>> seq = Sequence.generate_random_normal(n_events=100, mu=500, sigma=25)  # milliseconds
    >>> xf, yf = fft_values(seq, unit_size=1000, x_max=10)

    """
    # Calculate step size
    step_size = unit_size / 1000

    # Make a sequence of ones and zeroes
    timeseries = thebeat.helpers.sequence_to_binary(sequence, resolution=step_size)
    duration = sequence.duration
    x_length = np.ceil(duration / step_size).astype(int)

    # Do the fft
    yf = rfft(timeseries)
    xf = rfftfreq(x_length, d=step_size) * (step_size / 0.001)

    # Slice the data and take absolute values (we don't care about complex numbers)
    min_freq_index = np.min(np.where(xf > x_min)).astype(int) if x_min else None
    max_freq_index = np.min(np.where(xf > x_max)).astype(int) if x_max else None
    yf = np.abs(yf[min_freq_index:max_freq_index])
    xf = xf[min_freq_index:max_freq_index]

    return xf, yf


def fft_plot(
    sequence: thebeat.core.Sequence,
    unit_size: float,
    x_min: float | None = None,
    x_max: float | None = None,
    style: str = "seaborn-v0_8",
    title: str = "Fourier transform",
    x_axis_label: str = "Cycles per unit",
    y_axis_label: str = "Absolute power",
    figsize: tuple | None = None,
    dpi: int = 100,
    ax: plt.Axes | None = None,
    suppress_display: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the Fourier transform of a :class:`~thebeat.core.Sequence` object.
    The ``unit_size`` parameter is required, because Sequence objects are agnostic about the used time unit.
    You can use 1000 if the Sequence is in milliseconds, and 1 if the Sequence is in seconds.

    On the x axis is plotted the number of cycles per unit, and on the y axis the absolute power.
    The number of cycles per unit can be interpreted as the beat frequency. For instance, 2 cycles for a unit size of
    1000 ms means a beat frequency of 2 Hz.

    Note
    ----
    In most beat-finding applications you will want to set the ``x_max`` argument to something reasonable.
    The Fourier transform is plotted for all possible frequencies until the Nyquist frequency (half the
    value of ``unit_size``). However, in most cases you will not be interested in frequencies that are higher than
    the beat frequency. For instance, if you have a sequence with a beat frequency of 2 Hz, you will not be interested
    in frequencies higher than, say, 20 Hz. In that case, you can set ``x_max`` to 20.

    Parameters
    ----------
    sequence
        The sequence.
    unit_size
        The size of the unit in which the sequence is measured. If the sequence is in milliseconds,
        you probably want 1000. If the sequence is in seconds, you probably want 1.
    x_min
        The minimum number of cycles per unit to be plotted.
    x_max
        The maximum number of cycles per unit to be plotted.
    style
        The matplotlib style to use. See
        `matplotlib style reference <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    title
        The title of the plot.
    x_axis_label
        The label of the x axis.
    y_axis_label
        The label of the y axis.
    figsize
        A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
        This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.figure`.
    dpi
        The resolution of the plot in dots per inch.
    ax
        A matplotlib Axes object to plot on. If not provided, a new figure and axes will be created.
    suppress_display
        If True, the plot will not be displayed.

    Returns
    -------
    fig
        The matplotlib Figure object.
    ax
        The matplotlib Axes object.

    Examples
    --------
    >>> from thebeat import Sequence
    >>> from thebeat.stats import fft_plot
    >>> seq = Sequence.generate_random_normal(n_events=100, mu=500, sigma=25)  # milliseconds
    >>> fft_plot(seq, unit_size=1000)  # doctest: +SKIP
    (<Figure size 800x550 with 1 Axes>, <Axes: title={'center': 'Fourier transform'}, \
xlabel='Cycles per unit', ylabel='Absolute power'>)

    >>> seq = Sequence.generate_random_normal(n_events=100, mu=0.5, sigma=0.025)  # seconds
    >>> fft_plot(seq, unit_size=1, x_max=5)  # doctest: +SKIP
    (<Figure size 800x550 with 1 Axes>, <Axes: title={'center': 'Fourier transform'}, \
xlabel='Cycles per unit', ylabel='Absolute power'>)
    """

    # Get values
    xf, yf = fft_values(sequence=sequence, unit_size=unit_size, x_min=x_min, x_max=x_max)

    # Plot
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True, dpi=dpi)
            ax_provided = False
        else:
            fig = ax.get_figure()
            ax_provided = True
        ax.plot(xf, yf)
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_xlim(x_min, None)
        ax.set_title(title)

    if not suppress_display and ax_provided is False:
        fig.show()

    return fig, ax


def ks_test(
    sequence: thebeat.core.Sequence,
    reference_distribution: str = "normal",
    alternative: str = "two-sided",
):
    """
    This function returns the `D` statistic and `p` value of a one-sample Kolmogorov-Smirnov test.
    It calculates how different the distribution of inter-onset intervals (IOIs) is compared to the provided reference
    distribution.

    If `p` is significant it means that the IOIs are not distributed according to the provided reference distribution.

    Parameters
    ----------
    sequence
        A :class:`~thebeat.core.Sequence` object.
    reference_distribution
        Either 'normal' or 'uniform'. The distribution against which the distribution of inter-onset intervals (IOIs)
        is compared.
    alternative
        Either 'two-sided', 'less', or 'greater'. See :func:`scipy.stats.kstest` for more information.

    Returns
    -------
    :class:`scipy.stats._stats_py.KstestResult`
        A SciPy named tuple containing the `D` statistic and the `p` value.

    Notes
    -----
    This function uses :func:`scipy.stats.kstest`. For more information about the use of the
    Kolmogorov-Smirnov test in rhythm research, see :cite:t:`jadoulSeekingTemporalPredictability2016` and
    :cite:t:`ravignaniMeasuringRhythmicComplexity2017`.

    Examples
    --------
    >>> rng = np.random.default_rng(seed=123)
    >>> seq = thebeat.core.Sequence.generate_random_normal(n_events=100,mu=500,sigma=25,rng=rng)
    >>> ks_result = ks_test(seq)
    >>> print(round(ks_result.pvalue, 5))
    0.6608

    """

    sequence = sequence.iois

    if reference_distribution == "normal":
        mean = np.mean(sequence)
        sd = np.std(sequence)
        dist = scipy.stats.norm(loc=mean, scale=sd).cdf
        return scipy.stats.kstest(sequence, dist, alternative=alternative)
    elif reference_distribution == "uniform":
        a = min(sequence)
        b = max(sequence)
        scale = b - a
        dist = scipy.stats.uniform(loc=a, scale=scale).cdf
        return scipy.stats.kstest(sequence, dist, alternative=alternative)
    else:
        raise ValueError("Unknown distribution. Choose 'normal' or 'uniform'.")


def get_interval_ratios_from_dyads(sequence: np.array | thebeat.core.Sequence | list):
    r"""
    Return sequential interval ratios, calculated as:

    :math:`\textrm{ratio}_k = \frac{\textrm{IOI}_k}{\textrm{IOI}_k + \textrm{IOI}_{k+1}}`.

    Note that for *n* IOIs this property returns *n*-1 ratios.

    Parameters
    ----------
    sequence
        The sequence from which to calculate the interval ratios. Can be a Sequence object, or a list or array of
        IOIs.

    Notes
    -----
    The used method is based on the methodology from :cite:t:`roeskeCategoricalRhythmsAre2020`.

    """
    if isinstance(sequence, thebeat.core.Sequence):
        sequence = sequence.iois

    return sequence[:-1] / (sequence[1:] + sequence[:-1])


def get_phase_differences(
    test_events: thebeat.core.Sequence | list[numbers.Real] | np.ndarray | numbers.Real,
    reference_sequence: thebeat.core.Sequence,
    reference_ioi: str = "preceding",
    window_size: int | None = None,
    unit: str = "degrees",
    modulo: bool = True
) -> np.ndarray | float:
    r"""Get the phase differences for ``test_events`` compared to ``reference_sequence``.

    Phase differences are a (circular) measure of temporal alignment. They are calculated as the difference between
    the onset of a test event and the onset of the corresponding reference event, divided by the IOI of the reference
    IOI:

    :math:`\textrm{phase difference} = \frac{\textrm{test onset} - \textrm{reference onset}}{\textrm{reference IOI}}`

    The resulting phase differences are expressed in the unit specified by ``unit``, where the default is in degrees.

    The reference IOI can either be the IOI in the reference sequence that 'contains' an onset in the test sequence
    (``reference_ioi='containing'``), or the IOI in the reference sequences that precedes this onset (``reference_ioi='preceding'``).
    The default is for the reference IOI to be the one that precedes the test onset. Let us consider a reference
    sequence with onsets:

    :math:`t = 0`, :math:`t = 250`, :math:`t = 1000`

    If an onset in the test sequence is at :math:`t = 750`,
    and ``reference_ioi='containing'``, the reference IOI that will be used is the one from :math:`t = 250` to :math:`t = 1000`.
    If ``reference_ioi='preceding'``, the reference IOI that will be used is the one from :math:`t = 0` to :math:`t = 250`.

    In addition, one can specify a window size to get a (moving) average of the preceeding IOIs. This is only used if ``reference_ioi='preceding'``.
    The window size determines the number of reference IOIs (up to and including the immediately 'preceding' one) that are used to calculate a mean
    reference IOI. This can be useful to smoothe out the reference IOI when using irregular (reference) sequences.

    Finally, one can specify whether modular arithmetic should be used. For degrees, if ``modulo=True``, the phase differences will be
    expressed in the range :math:`[0, 360)` degrees, :math:`[0, 2\pi)` radians, or :math:`[0, 1)` (as a plain 'fraction').
    If ``modulo=False``, the phase differences will be expressed in the range :math:`[0, \infty)`, with the event at the start of the containing
    interval corresponding to 0 for both the 'containing' and 'preceeding' values.

    Note
    ----
    In cases where it is not possible to calculate a phase difference, ``np.nan`` is returned.
    This can happen in the following cases:
    - The test event is before the first onset in the reference sequence.
    - The test event is after the last onset in the reference sequence, and ``reference_ioi='containing'``.
    - The test event is in the nth interval of the reference sequence, and ``reference_ioi='preceding'`` and a ``window_size`` is larger than n.

    Examples
    --------
    >>> from thebeat.core import Sequence
    >>> reference = Sequence.from_onsets([0, 250, 1000])
    >>> test = Sequence.from_onsets([250, 1250])
    >>> get_phase_differences(test, reference, reference_ioi='preceding', unit='fraction')
    array([0.        , 0.33333333])


    Parameters
    ----------
    test_events
        A sequence or a single time point to be compared with the reference sequence. Can either be a single event time,
        or a Sequence, list, or NumPy array containing multiple events.
    reference_sequence
        The reference sequence. Can be a Sequence object, a list or array of Sequence objects, or a number.
        In the latter case, the reference sequence will be an isochronous sequence with a constant IOI of that
        number and the same length as ``sequence_1``.
    reference_ioi
        The IOI in the reference sequence that is used as the reference IOI. Can be either "containing" or "preceding".
    window_size
        The window size used for calculating the mean reference IOI. Only used if ``reference_ioi='preceding'``.
    circular_unit
        The unit of the circular unit. Can be "degrees" (the default), "radians", or "fraction".
    modulo
        Return the phase differences modulo 360 degrees or not. Only has an effect if ``reference_ioi='preceding'``.

    Returns
    -------
    :class:`numpy.ndarray`
        An array containing the phase differences if a Sequence, list, or NumPy array was passed.
    :class:`float`
        The phase difference if a single event time was passed.

    """

    # Input validation and processing
    if isinstance(test_events, thebeat.Sequence):
        test_onsets = test_events.onsets
    elif isinstance(test_events, (list, np.ndarray)):
        test_onsets = np.array(test_events)
    elif isinstance(test_events, numbers.Real):
        test_onsets = np.array([test_events])
    else:
        raise TypeError("test_events must be a Sequence, list, NumPy array, or float.")

    if not isinstance(reference_sequence, thebeat.Sequence):
        raise TypeError("reference_sequence must be a Sequence object.")

    if reference_ioi not in ("containing", "preceding"):
        raise ValueError("reference_ioi must be either 'containing' or 'preceding'.")

    if reference_ioi == "containing" and window_size is not None:
        raise ValueError("window_size cannot be used with reference_ioi='containing'.")
    elif reference_ioi == "preceding" and window_size is None:
        window_size = 1
    elif reference_ioi == "preceding" and window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    if unit not in ("degrees", "radians", "fraction"):
        raise ValueError("unit must be either 'degrees', 'radians' or 'fraction'")

    reference_onsets = reference_sequence.onsets
    reference_end = reference_sequence.onsets[0] + reference_sequence.duration
    reference_iois = reference_sequence.iois
    phase_diffs = []

    for test_onset in test_onsets:
        if test_onset >= reference_end:
            containing_ioi_index = len(reference_iois)
        elif test_onset < reference_onsets[0]:
            containing_ioi_index = -1
        else:
            containing_ioi_index = np.flatnonzero(test_onset >= reference_onsets)[-1]

        if reference_ioi == "containing":
            if not 0 <= containing_ioi_index < len(reference_iois):
                phase_diff = np.nan
            else:
                reference_ioi_ = reference_iois[containing_ioi_index]
                phase_diff = (test_onset - reference_onsets[containing_ioi_index]) / reference_ioi_
        else:
            assert reference_ioi == "preceding"
            if containing_ioi_index < window_size:
                phase_diff = np.nan
            else:
                mean_reference_ioi = np.mean(reference_iois[containing_ioi_index - window_size:containing_ioi_index])
                reference_onset = reference_onsets[containing_ioi_index] if containing_ioi_index < len(reference_onsets) else reference_end
                phase_diff = (test_onset - reference_onset) / mean_reference_ioi
        phase_diffs.append(phase_diff)

    phase_diffs = np.array(phase_diffs, dtype=np.float64)
    if modulo:
        phase_diffs = np.fmod(phase_diffs, 1)

    if unit == "degrees":
        return phase_diffs * 360
    elif unit == "radians":
        return phase_diffs * 2 * np.pi

    if isinstance(test_events, numbers.Real):
        return float(phase_diffs[0])
    else:
        return phase_diffs


def get_rhythmic_entropy(
    sequence: thebeat.core.Sequence | thebeat.music.Rhythm, smallest_unit: float
):
    """
    Calculate Shannon entropy from bins. This is a measure of rhythmic complexity.
    If many different 'note durations' are present, entropy is high. If only a few are present, entropy is low.
    A sequence that is completely isochronous has a Shannon entropy of 0.

    The smallest_unit determines the size of the bins/the underlying grid.
    In musical terms, this for instance represents a 1/16th note. Bins will then be made such
    that each bin has a width of one 1/16th note. A 1/4th note will then be contained in one of those bins.
    Sequence needs to be quantized to multiples
    of 'smallest_unit'. If needed, quantize the Sequence first, e.g. using 'Sequence.quantize_iois'.

    Parameters
    ----------
    sequence
        The :py:class:`thebeat.core.Sequence` object for which Shannon entropy is calculated.
    smallest_unit
        The size of the bins/the underlying grid.

    Example
    -------
    >>> seq = thebeat.Sequence.generate_isochronous(n_events=10, ioi=500)
    >>> print(get_rhythmic_entropy(seq, smallest_unit=250))
    0.0

    """

    if not isinstance(sequence, (thebeat.core.Sequence, thebeat.music.Rhythm)):
        raise TypeError("sequence must be a Sequence or Rhythm object")

    if np.any(sequence.iois % smallest_unit != 0):
        raise ValueError(
            f"Sequence needs to be quantized to multiples of {smallest_unit}."
            "If needed, quantize the Sequence first e.g. using 'Sequence.quantize_iois'."
        )

    bins = (
        np.arange(0, np.max(sequence.iois) + 2 * smallest_unit, smallest_unit) - smallest_unit / 2
    )  # shift bins to center
    bin_counts = np.histogram(sequence.iois, bins=bins)[0]

    return scipy.stats.entropy(bin_counts)


def get_cov(sequence: thebeat.core.Sequence) -> np.float64:
    """
    Calculate the coefficient of variantion of the inter-onset intervals (IOIS) in a
    :py:class:`thebeat.core.Sequence` object.

    Parameters
    ----------
    sequence
        A :py:class:`thebeat.core.Sequence` object.

    Returns
    -------
    float
        The covariance of the sequence.

    """
    return np.float64(np.std(sequence.iois) / np.mean(sequence.iois))


def get_npvi(sequence: thebeat.core.Sequence) -> np.float64:
    """

    This function calculates the normalized pairwise variability index (nPVI) for a provided :py:class:`Sequence` or
    :py:class:`SoundSequence` object, or for an interable of inter-onset intervals (IOIs).

    Parameters
    ----------
    sequence
        Either a :py:class:`Sequence` or :py:class:`SoundSequence` object, or an iterable containing inter-onset
        intervals (IOIs).

    Returns
    -------
    :class:`numpy.float64`
        The nPVI for the provided sequence.

    Notes
    -----
    The normalied pairwise variability index (nPVI) is a measure of the variability of adjacent temporal intervals.
    The nPVI is zero for sequences that are perfectly isochronous.
    See :cite:t:`jadoulSeekingTemporalPredictability2016` and :cite:t:`ravignaniMeasuringRhythmicComplexity2017`
    for more information on its use in rhythm research.

    Examples
    --------
    >>> seq = thebeat.core.Sequence.generate_isochronous(n_events=10, ioi=500)
    >>> print(get_npvi(seq))
    0.0

    >>> rng = np.random.default_rng(seed=123)
    >>> seq = thebeat.core.Sequence.generate_random_normal(n_events=10,mu=500,sigma=50,rng=rng)
    >>> print(get_npvi(seq))
    9.40657936323865
    """

    if isinstance(sequence, (thebeat.core.Sequence, thebeat.core.SoundSequence)):
        iois = sequence.iois
    else:
        iois = np.array(sequence)

    npvi_values = []

    for i in range(1, len(iois)):
        diff = iois[i] - iois[i - 1]
        mean = np.mean([iois[i], iois[i - 1]])
        npvi_values.append(np.abs(diff / mean))

    npvi = (100 / (len(iois) - 1)) * np.sum(npvi_values)

    return np.float64(npvi)


def get_ugof_isochronous(
    test_sequence: thebeat.core.Sequence, reference_ioi: float, output_statistic: str = "mean"
) -> np.float64:
    r"""

    This function calculates the universal goodness of fit (``ugof``) measure.
    The ``ugof`` statistic quantifies how well a theoretical sequence describes a sequence at hand
    (the ``test_sequence``). This function can only calculate ``ugof`` using a theoretical sequence that is isochronous.

    The ``reference_ioi`` is the IOI of an isochronous theoretical sequence.

    Parameters
    ----------
    test_sequence
        A :py:class:`~thebeat.core.Sequence` or :py:class:`~thebeat.core.SoundSequence` object that will be compared
        to ``reference_sequence``.
    reference_ioi
        A number (float or int) representing the IOI of an isochronous sequence.
    output_statistic
        Either 'mean' (the default) or 'median'. This determines whether for the individual ugof values we take the mean
        or the median as the output statistic.

    Returns
    -------
    :class:`numpy.float64`
        The ugof statistic.

    Notes
    -----
    This measure is described in :cite:t:`burchardtNovelIdeasFurther2021`.
    Please also refer to `this Github page <https://github.com/LSBurchardt/R_app_rhythm>`_ for an R implementation of
    the *ugof* measure.


    Examples
    --------
    >>> seq = thebeat.core.Sequence.generate_isochronous(n_events=10, ioi=1000)
    >>> ugof = get_ugof_isochronous(seq, reference_ioi=68.21)
    >>> print(ugof)
    0.4646435

    """

    # Input validation
    if not isinstance(test_sequence, thebeat.core.sequence.Sequence):
        raise TypeError("test_sequence must be a Sequence object")
    if not isinstance(reference_ioi, numbers.Real):
        raise TypeError("reference_ioi must be a number (int or float)")

    # Re-calculate onsets based on the sequence's IOIs, this to make sure both the test
    # and reference sequences start with an onset at 0 (of which the ugof will be discarded)
    test_onsets = np.concatenate([[0], np.cumsum(test_sequence.iois)])

    reference_onsets = np.arange(
        start=0, stop=np.max(test_onsets) + reference_ioi + 1, step=reference_ioi
    )

    # For each onset, get the closest theoretical beat and get the absolute difference
    minimal_deviations = np.min(np.abs(test_onsets[:, None] - reference_onsets), axis=1)
    maximal_deviation = reference_ioi / 2

    # calculate ugofs
    ugof_values = minimal_deviations / maximal_deviation

    if output_statistic == "mean":
        return np.float32(
            np.mean(ugof_values[1:])
        )  # discard the first value because that will always be 0
    elif output_statistic == "median":
        return np.float32(
            np.median(ugof_values[1:])
        )  # discard the first value because that will always be 0
    else:
        raise ValueError("The output statistic can only be 'median' or 'mean'.")
