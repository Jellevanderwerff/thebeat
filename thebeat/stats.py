from __future__ import annotations
from typing import Union, Optional
import scipy.stats
import scipy.fft
import numpy as np
import thebeat.core
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd


def acf_df(sequence: Union[thebeat.core.Sequence, thebeat.core.StimSequence, np.typing.ArrayLike],
           resolution_ms: int = 1,
           smoothing_window: Optional[float] = None,
           smoothing_sd: Optional[float] = None) -> pd.DataFrame:
    """

    This function takes a :py:class:`Sequence` or :py:class:`StimSequence` object, or a list of event onsets,
    and returns a :class:`pandas.DataFrame` containing timestamps (acf lags), and autocorrelation factors.

    Parameters
    ----------

    sequence
        Either a :py:class:`~thebeat.core.Sequence` or :py:class:`~thebeat.core.StimSequence` object, or a list or an
        array containing event onsets in milliseconds, e.g. ``[0, 500, 1000]``.
    resolution_ms
        The temporal resolution in milliseconds (i.e. sampling frequency/step size). The number of lags
        calculated for the autocorrelation function can be calculated as
        ``n_lags = sequence_duration_in_ms / resolution_ms``.
    smoothing_window
        The window (in milliseconds) within which a normal probability density function is used for
        smoothing out the analysis.
    smoothing_sd
        The standard deviation of the normal probability density function used for smoothing out the analysis.

    Returns
    -------
    :class:`pandas.DataFrame`
        DataFrame containing two columns: the timestamps in milliseconds, and the autocorrelation factor.

    Notes
    -----
    This function is based on the procedure described in :footcite:t:`ravignaniMeasuringRhythmicComplexity2017`.
    There, one can also find a more detailed description of the smoothing procedure.

    Examples
    --------
    >>> rng = np.random.default_rng(seed=123)  # for reproducability
    >>> seq = thebeat.core.Sequence.generate_random_uniform(n=10, a=400, b=600, rng=rng)
    >>> df = acf_df(seq, smoothing_window=50, smoothing_sd=20, resolution_ms=10)
    >>> print(df.head(3))
       time_ms  correlation
    0        0     1.000000
    1       10     0.851373
    2       20     0.590761

    """

    correlations = acf_values(sequence=sequence, resolution=resolution_ms, smoothing_window=smoothing_window,
                              smoothing_sd=smoothing_sd)
    correlations = correlations / max(correlations)
    times_ms = np.arange(start=0, stop=len(correlations) * resolution_ms, step=resolution_ms)

    df = pd.DataFrame(
        {
            "time_ms": times_ms,
            "correlation": correlations
        }
    )

    return df


def acf_plot(sequence: Union[thebeat.core.Sequence, thebeat.core.StimSequence, np.typing.ArrayLike],
             resolution: int = 1,
             smoothing_window: Optional[float] = None,
             smoothing_sd: Optional[float] = None,
             style: str = 'seaborn',
             title: str = 'Autocorrelation',
             x_axis_label: str = 'Lag',
             figsize: Optional[tuple] = None,
             suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:
    """

    This function can be used for plotting an autocorrelation plot from a :class:`~thebeat.core.Sequence` or
    :class:`~thebeat.core.StimSequence` object, or from a list or an array of event onsets.

    Parameters
    ----------
    sequence
        Either a :class:`~thebeat.core.Sequence` or :class:`~thebeat.core.StimSequence` object,
        or a list or an array of event onsets in milliseconds, e.g. ``[0, 500, 1000]``.
    resolution
        The temporal resolution in milliseconds (i.e. sampling frequency/step size). The number of lags
        calculated for the autocorrelation function can be calculated as
        ``n_lags = sequence_duration_in_ms / resolution_ms``
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
        :class:`~thebeat.core.Sequence` or :class:`~thebeat.core.StimSequence` name as the title of the plot
        (if passed and the object has one).
    x_axis_label
        A label for the x axis.
    figsize
        A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
        This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.figure`.
    suppress_display
        If ``True``, :func:`matplotlib.pyplot.show` is not run.

    Notes
    -----
    This function is based on the procedure described in :footcite:t:`ravignaniMeasuringRhythmicComplexity2017`.
    There, one can also find a more detailed description of the smoothing procedure.

    """

    if isinstance(sequence, (thebeat.core.sequence.Sequence, thebeat.core.stimsequence.StimSequence)):
        onsets = sequence.onsets
    else:
        onsets = sequence

    correlation = acf_values(sequence=sequence, resolution=resolution, smoothing_window=smoothing_window,
                             smoothing_sd=smoothing_sd)

    x_step = resolution
    max_lag = np.floor(np.max(onsets))

    # plot
    y = correlation[:int(max_lag)]
    y = y / max(y)  # normalize

    # Make x axis
    x = np.arange(start=0, stop=len(y) * x_step, step=x_step)

    if isinstance(sequence, thebeat.core.StimSequence):
        x /= 1000
        if np.max(x) > 10000:
            x /= 1000

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.axes.set_xlabel(x_axis_label)
        ax.axes.set_title(title)
        ax.plot(x, y)

    if not suppress_display:
        plt.show()

    return fig, ax


def acf_values(sequence: Union[thebeat.core.Sequence, thebeat.core.StimSequence, np.typing.ArrayLike],
               resolution: int = 1,
               smoothing_window: Optional[float] = None,
               smoothing_sd: Optional[float] = None) -> np.ndarray:
    """

    This function takes a :class:`~thebeat.core.Sequence` or :class:`~thebeat.core.StimSequence` object,
    or a list of event onsets, and returns an array with steps of ``resolution_ms`` of unstandardized correlation
    factors.

    Parameters
    ----------
    sequence
        Either a Sequence or StimSequence object, or an iterable containing event onsets in milliseconds,
        e.g. ``[0, 500, 1000]``.
    resolution
        The temporal resolution in milliseconds (i.e. sampling frequency). The number of lags calculated for the
        autocorrelation function can be calculated as ``n_lags = sequence_duration_in_ms / resolution_ms``.
    smoothing_window
        The window (in milliseconds) within which a normal probability density function is used for
        smoothing out the analysis.
    smoothing_sd
        The standard deviation of the normal probability density function used for smoothing out the analysis.

    Notes
    -----
    This function is based on the procedure described in :footcite:t:`ravignaniMeasuringRhythmicComplexity2017`. There,
    one can also find a more detailed description of the smoothing procedure.

    This function uses the :func:`numpy.correlate` to calculate the correlations.

    """

    if isinstance(sequence, (thebeat.core.sequence.Sequence, thebeat.core.stimsequence.StimSequence)):
        onsets_ms = sequence.onsets
    else:
        onsets_ms = sequence

    signal = _make_ones_and_zeros_timeseries(onsets_ms, resolution)

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
                         "and smoothing_sd. Try choosing a smaller resolution_ms.") from e

    return correlation


def ks_test(sequence: Union[thebeat.core.Sequence, thebeat.core.StimSequence, np.typing.ArrayLike],
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
        Either a :class:`~thebeat.core.Sequence` or :class:`~thebeat.core.StimSequence` object or an iterable (e.g. list)
        containing inter-onset intervals (IOIs), e.g. ``[500, 500, 500]``.
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
    Kolmogorov-Smirnov test in rhythm research, see :footcite:t:`jadoulSeekingTemporalPredictability2016` and
    :footcite:t:`ravignaniMeasuringRhythmicComplexity2017`.

    Examples
    --------
    >>> rng = np.random.default_rng(seed=123)
    >>> seq = thebeat.core.Sequence.generate_random_normal(n=100, mu=500, sigma=25, rng=rng)
    >>> print(ks_test(seq))
    KstestResult(statistic=0.07176677141846549, pvalue=0.6608009345687911)

    """

    if isinstance(sequence, (thebeat.core.Sequence, thebeat.core.StimSequence)):
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


def get_npvi(sequence: Union[thebeat.core.Sequence, thebeat.core.StimSequence, np.typing.ArrayLike]) -> np.float64:
    """

    This function calculates the normalized pairwise variability index (nPVI) for a provided :py:class:`Sequence` or
    :py:class:`StimSequence` object, or for an interable of inter-onset intervals (IOIs).

    Parameters
    ----------
    sequence
        Either a :py:class:`Sequence` or :py:class:`StimSequence` object, or an iterable containing inter-onset
        intervals (IOIs).

    Returns
    -------
    :class:`numpy.float64`
        The nPVI for the provided sequence.

    Notes
    -----
    The normalied pairwise variability index (nPVI) is a measure of the variability of adjacent temporal intervals.
    The nPVI is zero for sequences that are perfectly isochronous.
    See :footcite:t:`jadoulSeekingTemporalPredictability2016` and :footcite:t:`ravignaniMeasuringRhythmicComplexity2017`
    for more information on its use in rhythm research.

    Examples
    --------
    >>> seq = thebeat.core.Sequence.generate_isochronous(n=10, ioi=500)
    >>> print(get_npvi(seq))
    0.0

    >>> rng = np.random.default_rng(seed=123)
    >>> seq = thebeat.core.Sequence.generate_random_normal(n=10, mu=500, sigma=50, rng=rng)
    >>> print(get_npvi(seq))
    37.6263174529546
    """

    if isinstance(sequence, (thebeat.core.Sequence, thebeat.core.StimSequence)):
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


def get_ugof(sequence: Union[thebeat.core.Sequence, thebeat.core.StimSequence, np.typing.ArrayLike[float]],
             theoretical_ioi: float,
             output_statistic: str = 'mean') -> np.float64:
    """

    This function calculates the universal goodness of fit (`ugof`) measure for a sequence compared to a
    theoretical/underlying inter-onset interval (IOI). The `ugof` statistic quantifies how well a theoretical IOI
    describes a sequence.

    Parameters
    ----------
    sequence
        Either a :py:class:`~thebeat.core.Sequence` or :py:class:`~thebeat.core.StimSequence` object, or a list or an
        array containing event onsets, e.g. ``[0, 500, 1000]``.
    theoretical_ioi
        The theoretical/underlying inter-onset interval (IOI) to which the sequence is compared.
    output_statistic
        Either 'mean' (the default) or 'median'. This determines whether for the individual ugof values we take the mean
        or the median as the output statistic.

    Returns
    -------
    :class:`numpy.float64`
        The ugof statistic.

    Notes
    -----
    This measure is described in :footcite:t:`burchardtNovelIdeasFurther2021`.


    Examples
    --------
    >>> seq = thebeat.core.Sequence.generate_isochronous(n=10, ioi=1000)
    >>> ugof = get_ugof(seq, theoretical_ioi=68.21)
    >>> print(ugof)
    0.59636414

    """

    # Get the onsets
    if isinstance(sequence, (thebeat.core.sequence.Sequence, thebeat.core.stimsequence.StimSequence)):
        onsets = sequence.onsets  # in ms
    else:
        onsets = np.array(sequence)

    # Make a theoretical sequence that's multiples of the theoretical_ioi (the + one is because arange doesn't include
    # the stop)
    theo_seq = np.arange(start=0, stop=np.max(onsets) + 1, step=theoretical_ioi)

    # For each onset, get the closest theoretical beat and get the absolute difference
    minimal_deviations = np.array([min(abs(onsets[n] - theo_seq)) for n in range(len(onsets))])
    maximal_deviation = theoretical_ioi / 2  # in ms

    # calculate ugofs
    ugof_values = minimal_deviations / maximal_deviation

    if output_statistic == 'mean':
        return np.float32(np.mean(ugof_values))
    elif output_statistic == 'median':
        return np.float32(np.median(ugof_values))
    else:
        raise ValueError("Output can only be 'median' or 'mean'.")


def _make_ones_and_zeros_timeseries(onsets_ms, resolution_ms):
    """
    Converts a sequence of millisecond onsets to a series of zeros and ones.
    Ones for the onsets.
    """
    duration = max(onsets_ms)
    zeros_n = int(np.ceil(duration / resolution_ms)) + 1
    signal = np.zeros(zeros_n)

    for onset in onsets_ms:
        index = int(onset / resolution_ms)
        signal[index] = 1

    return np.array(signal)

