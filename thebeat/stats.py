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
from typing import Optional, Union
import scipy.stats
import scipy.fft
import numpy as np
import thebeat.core
from thebeat.helpers import make_binary_timeseries
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
from scipy.fft import rfft, rfftfreq
import pandas as pd
import thebeat.helpers


def acf_df(sequence: thebeat.core.Sequence,
           resolution,
           smoothing_window: Optional[float] = None,
           smoothing_sd: Optional[float] = None) -> pd.DataFrame:
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
    0          0     0.851373
    1         10     1.000000
    2         20     0.851373

    """

    correlations = acf_values(sequence=sequence, resolution=resolution, smoothing_window=smoothing_window,
                              smoothing_sd=smoothing_sd)
    correlations = correlations / np.max(correlations)
    timestamps = np.arange(start=0, stop=len(correlations) * resolution, step=resolution)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "correlation": correlations
        }
    )

    return df


def acf_plot(sequence: thebeat.core.Sequence,
             resolution,
             max_lag: Optional[float] = None,
             smoothing_window: Optional[float] = None,
             smoothing_sd: Optional[float] = None,
             style: str = 'seaborn-v0_8',
             title: str = 'Autocorrelation',
             x_axis_label: str = 'Lag',
             y_axis_label: str = 'Correlation',
             figsize: Optional[tuple] = None,
             dpi: int = 100,
             ax: Optional[plt.Axes] = None,
             suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:
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

    correlation = acf_values(sequence=sequence, resolution=resolution, smoothing_window=smoothing_window,
                             smoothing_sd=smoothing_sd)

    x_step = resolution
    max_lag = int(max_lag // resolution) if max_lag else np.floor(np.max(onsets) / resolution).astype(int)

    # plot
    try:
        y = correlation[:max_lag]
        y = y / np.max(y)  # normalize
    except ValueError:
        raise ValueError("We end up with an empty y axis. Try changing the resolution.")

    # Make x axis
    x = np.arange(start=0, stop=len(y) * x_step, step=x_step)

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


def acf_values(sequence: thebeat.core.Sequence,
               resolution,
               smoothing_window: Optional[float] = None,
               smoothing_sd: Optional[float] = None) -> np.ndarray:
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

    onsets = sequence.onsets
    signal = thebeat.helpers.make_binary_timeseries(onsets, resolution)

    # npdf
    if smoothing_window and smoothing_sd:
        x = np.arange(start=-smoothing_window / 2, stop=smoothing_window / 2, step=resolution)
        npdf = scipy.stats.norm.pdf(x, 0, smoothing_sd)
        npdf = npdf / np.max(npdf)
        signal_convoluted = np.convolve(signal, npdf)
        signal = signal_convoluted[round(resolution * smoothing_window / 2):]

    try:
        correlation = np.correlate(signal, signal, 'full')
        correlation = correlation[round(len(correlation) / 2) - 1:]
    except ValueError as e:
        raise ValueError("Error! Hint: Most likely your resolution is too large for the chosen smoothing_window"
                         "and smoothing_sd. Try choosing a smaller resolution.") from e

    return correlation


def ccf_df(test_sequence: thebeat.core.Sequence,
           reference_sequence: thebeat.core.Sequence,
           resolution,
           smoothing_window: Optional[float] = None,
           smoothing_sd: Optional[float] = None) -> pd.DataFrame:
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

    correlations = ccf_values(test_sequence=test_sequence, reference_sequence=reference_sequence, resolution=resolution,
                              smoothing_window=smoothing_window, smoothing_sd=smoothing_sd)
    # normalize
    correlations = correlations / np.max(correlations)
    timestamps = np.arange(start=0, stop=len(correlations) * resolution, step=resolution)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "correlation": correlations
        }
    )

    return df


def ccf_plot(test_sequence: thebeat.core.Sequence,
             reference_sequence: thebeat.core.Sequence,
             resolution,
             smoothing_window: Optional[float] = None,
             smoothing_sd: Optional[float] = None,
             style: str = 'seaborn-v0_8',
             title: str = 'Cross-correlation',
             x_axis_label: str = 'Lag',
             y_axis_label: str = 'Correlation',
             figsize: Optional[tuple] = None,
             dpi: int = 100,
             ax: Optional[plt.Axes] = None,
             suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:
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
    correlation = ccf_values(test_sequence=test_sequence, reference_sequence=reference_sequence,
                             resolution=resolution, smoothing_window=smoothing_window,
                             smoothing_sd=smoothing_sd)

    # Make y axis
    x_step = resolution
    max_lag = np.floor(np.max(np.concatenate([test_sequence.onsets, reference_sequence.onsets])) / resolution).astype(
        int)
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


def ccf_values(test_sequence: thebeat.core.Sequence,
               reference_sequence: thebeat.core.Sequence,
               resolution: float,
               smoothing_window: Optional[float] = None,
               smoothing_sd: Optional[float] = None) -> np.ndarray:
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
    # Get event onsets
    test_onsets = test_sequence.onsets
    ref_onsets = reference_sequence.onsets

    # Make into 0's and 1's
    test_signal = thebeat.helpers.make_binary_timeseries(test_onsets, resolution)
    ref_signal = thebeat.helpers.make_binary_timeseries(ref_onsets, resolution)

    # npdf
    if smoothing_window and smoothing_sd:
        x = np.arange(start=-smoothing_window / 2, stop=smoothing_window / 2, step=resolution)
        npdf = scipy.stats.norm.pdf(x, 0, smoothing_sd)
        npdf = npdf / np.max(npdf)
        test_signal_convoluted = np.convolve(test_signal, npdf)
        ref_signal_convoluted = np.convolve(ref_signal, npdf)
        test_signal = test_signal_convoluted[round(resolution * smoothing_window / 2):]
        ref_signal = ref_signal_convoluted[round(resolution * smoothing_window / 2):]

    # Make signals of equal length
    diff = len(ref_signal) - len(test_signal)
    if diff > 0:  # ref_signal is longer
        test_signal = np.concatenate((test_signal, np.zeros(diff)))
    elif diff < 0:  # test_signal is longer
        ref_signal = np.concatenate((ref_signal, np.zeros(-diff)))

    # Calculate cross-correlation
    try:
        correlation = np.correlate(test_signal, ref_signal, 'full')
        correlation = correlation[round(len(correlation) / 2) - 1:]
    except ValueError as e:
        raise ValueError("Error! Hint: Most likely your resolution is too large for the chosen smoothing_window"
                         "and smoothing_sd. Try choosing a smaller resolution.") from e

    return correlation


def fft_plot(sequence: thebeat.core.Sequence,
             unit_size: float,
             x_min: float = 0,
             x_max: Optional[float] = None,
             style: str = 'seaborn-v0_8',
             title: str = 'Fourier transform',
             x_axis_label: str = 'Cycles per unit',
             y_axis_label: str = 'Absolute power',
             figsize: Optional[tuple] = None,
             dpi: int = 100,
             ax: Optional[plt.Axes] = None,
             suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the Fourier transform of a :class:`~thebeat.core.Sequence` object.
    The ``unit_size`` parameter is required, because Sequence objects are agnostic about the used time unit.
    You can use 1000 if the Sequence is in milliseconds, and 1 if the Sequence is in seconds.
    Note that the first frame is discarded since it will always have the highest power, yet is not informative.

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
    >>> fft_plot(seq, unit_size=1000)
    (<Figure size 800x550 with 1 Axes>, <AxesSubplot: title={'center': 'Fourier transform'}, xlabel='Cycles per unit', ylabel='Absolute power'>)

    >>> seq = Sequence.generate_random_normal(n_events=100, mu=0.5, sigma=0.025)  # seconds
    >>> fft_plot(seq, unit_size=1, x_max=5)
    (<Figure size 800x550 with 1 Axes>, <AxesSubplot: title={'center': 'Fourier transform'}, xlabel='Cycles per unit', ylabel='Absolute power'>)

    """

    # Calculate step size
    step_size = unit_size / 1000

    # Make a sequence of ones and zeroes
    timeseries = make_binary_timeseries(sequence.onsets, resolution=step_size)
    duration = np.max(sequence.onsets)
    x_length = np.ceil(duration / step_size).astype(int)

    # Do the fft
    yf = rfft(timeseries)[1:]
    xf = rfftfreq(x_length, d=step_size)[1:] * (step_size / 0.001)

    # Calculate reasonable max_freq
    max_freq_index = np.min(np.where(xf > x_max)) if x_max else len(xf) / 10
    yf = yf[:int(max_freq_index)]
    xf = xf[:int(max_freq_index)]

    # Plot
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True, dpi=dpi)
            ax_provided = False
        else:
            fig = ax.get_figure()
            ax_provided = True
        ax.plot(xf, np.abs(yf))
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel(y_axis_label)
        ax.set_xlim(x_min, None)
        ax.set_title(title)

    if not suppress_display and ax_provided is False:
        fig.show()

    return fig, ax


def ks_test(sequence: thebeat.core.Sequence,
            reference_distribution: str = 'normal',
            alternative: str = 'two-sided'):
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
        Either ‘two-sided’, ‘less’ or ‘greater’. See :func:`scipy.stats.kstest` for more information.

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
    >>> print(ks_test(seq))
    KstestResult(statistic=0.07176677141846549, pvalue=0.6608009345687911)

    """

    sequence = sequence.iois

    if reference_distribution == 'normal':
        mean = np.mean(sequence)
        sd = np.std(sequence)
        dist = scipy.stats.norm(loc=mean, scale=sd).cdf
        return scipy.stats.kstest(sequence, dist, alternative=alternative)
    elif reference_distribution == 'uniform':
        a = min(sequence)
        b = max(sequence)
        scale = b - a
        dist = scipy.stats.uniform(loc=a, scale=scale).cdf
        return scipy.stats.kstest(sequence, dist, alternative=alternative)
    else:
        raise ValueError("Unknown distribution. Choose 'normal' or 'uniform'.")


def get_rhythmic_entropy(sequence: Union[thebeat.core.Sequence, thebeat.music.Rhythm],
                         bin_fraction: float = 0.03125):
    """
    Calculate Shannon entropy from bins. This is a measure of rhythmic complexity.
    If many different 'note durations' are present, entropy is high. If only a few are present, entropy is low.
    A sequence that is completely isochronous has a Shannon entropy of 0.

    The bin size is determined from the average inter-onset interval in the
    :py:class:`thebeat.core.Sequence` object (i.e. the tempo) and the ``bin_fraction``.
    The ``bin_fraction`` corresponds to temporal sensitivity. The default is 1/32th of the average IOI.
    This implies that the smallest note value that can be detected is a 1/32th note.

    Parameters
    ----------
    sequence
        The :py:class:`thebeat.core.Sequence` object for which Shannon entropy is calculated.
    bin_fraction
        The fraction of the average inter-onset interval (IOI) that determines the bin size.
        It is multiplied by the average IOI to get the bin size.

    Example
    -------
    A :py:class:`~thebeat.core.Sequence` has an average IOI of 500 ms. With a bin_fraction of 0.03125
    (corresponding to 1/32th note value) the bins will have a size of 15.625 ms.
    The entropy will be calculated from the number of IOIs in each bin.

    References
    ----------
    #todo add reference here for this type of entropy calculation.

    """
    bin_size = np.mean(sequence.iois) * bin_fraction
    bins = np.arange(0, np.max(sequence.iois) + 2 * bin_size, bin_size) - bin_size / 2  # shift bins to center
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
    37.6263174529546
    """

    if isinstance(sequence, (thebeat.core.Sequence, thebeat.core.SoundSequence)):
        iois = sequence.iois
    else:
        iois = np.array(sequence)

    npvi_values = []

    for i in range(1, len(iois)):
        diff = iois[i] - iois[i - 1]
        mean = np.mean(iois[i] + iois[i - 1])
        npvi_values.append(np.abs(diff / mean))

    npvi = np.mean(npvi_values) * (100 * (len(iois) - 1))

    return np.float64(npvi)


def get_ugof_isochronous(test_sequence: thebeat.core.Sequence,
                         reference_ioi: float,
                         output_statistic: str = 'mean') -> np.float64:
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
    >>> ugof = get_ugof(seq, reference_sequence=68.21)
    >>> print(ugof)
    0.59636414

    """

    # Input validation and getting onsets for test sequence
    if not isinstance(test_sequence, thebeat.core.sequence.Sequence):
        raise TypeError('test_sequence must be a Sequence object')
    test_onsets = test_sequence.onsets

    # Input validation and getting onsets for reference sequence
    if not isinstance(reference_ioi, (int, float)):
        raise TypeError('reference_sequence must be a number (int or float)')

    reference_onsets = np.arange(start=0,
                                 stop=np.max(test_onsets) + reference_ioi + 1,
                                 step=reference_ioi)

    # For each onset, get the closest theoretical beat and get the absolute difference
    minimal_deviations = np.min(np.abs(test_onsets[:, None] - reference_onsets), axis=1)
    maximal_deviation = reference_ioi / 2

    # calculate ugofs
    ugof_values = minimal_deviations / maximal_deviation

    if output_statistic == 'mean':
        return np.float32(np.mean(ugof_values))
    elif output_statistic == 'median':
        return np.float32(np.median(ugof_values))
    else:
        raise ValueError("The output statistic can only be 'median' or 'mean'.")
