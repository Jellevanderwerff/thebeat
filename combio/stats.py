from __future__ import annotations
from typing import Iterable, Union
import scipy.stats
import scipy.fft
import numpy as np
from combio import core
from combio.core import Sequence
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd


def acf_df(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable],
           resolution_ms: int = 1,
           smoothing_window: Union[int, float] = 0,
           smoothing_sd: Union[int, float] = 0) -> pd.DataFrame:
    """

        This function takes a Sequence or StimSequence object, or a list of event onsets, and returns
        a Pandas dataframe containing timestamps (acf lags), and autocorrelation factors.

        Parameters
        ----------
        sequence : Sequence, StimSequence or iterable
            Either a Sequence or StimSequence object, or an iterable containing event onsets in milliseconds,
            e.g. [0, 500, 1000]
        resolution_ms : int, optional
            The temporal resolution in milliseconds (i.e. sampling frequency/step size). The number of lags
            calculated for the autocorrelation function can be calculated as
            n_lags = sequence_duration_in_ms / resolution_ms
        smoothing_window : int or float, optional
            The window (in milliseconds) within which a normal probability density function is used for
            smoothing out the analysis.
        smoothing_sd : int or float, optional
            The standard deviation of the normal probability density function used for smoothing out the analysis.

        Returns
        -------
        pandas.DataFrame
            A Pandas DataFrame containing two columns: the timestamps in milliseconds, and the autoccorrelation factor.

        Notes
        -----
        This function is based on the procedure described in Ravignani and Norton (2017) [1]_. There, one can also find a
        more detailed description of the smoothing procedure.

        References
        ----------
        .. [1] Ravignani, A., & Norton, P. (2017). Measuring rhythmic complexity:
           a primer to quantify and compare temporal structure in speech, movement, and animal vocalizations.
           Journal of Language Evolution, 2(1), 4-19.
           https://doi.org/10.1093/jole/lzx002

        Examples
        --------
        >>> rng = np.random.default_rng(seed=123)  # for reproducability
        >>> seq = Sequence.generate_random_uniform(n=10, a=400, b=600, rng=rng)
        >>> df = acf_df(seq, smoothing_window=50, smoothing_sd=20, resolution_ms=10)
        >>> print(df.head(3))
           time_ms  correlation
        0        0     1.000000
        1       10     0.851373
        2       20     0.590761


        """

    correlations = acf_values(sequence=sequence,
                              resolution_ms=resolution_ms,
                              smoothing_window=smoothing_window,
                              smoothing_sd=smoothing_sd)
    correlations = correlations / max(correlations)  # normalize
    times_ms = np.arange(start=0, stop=correlations.size * resolution_ms, step=resolution_ms)

    df = pd.DataFrame(
        {
            "time_ms": times_ms,
            "correlation": correlations
        }
    )

    return df


def acf_plot(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable],
             resolution_ms: int = 1,
             smoothing_window: Union[int, float] = 0,
             smoothing_sd: Union[int, float] = 0,
             style: str = 'seaborn',
             title: str = 'Autocorrelation',
             figsize: tuple = None,
             suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:
    """

    This function can be used for plotting an autocorrelation plot from a Sequence or StimSequence object,
    or from a list of event onsets.

    Parameters
    ----------
    sequence : Sequence, StimSequence or iterable
        Either a Sequence or StimSequence object, or an iterable containing event onsets in milliseconds,
        e.g. [0, 500, 1000]
    resolution_ms : int, optional
        The temporal resolution in milliseconds (i.e. sampling frequency/step size). The number of lags
        calculated for the autocorrelation function can be calculated as
        n_lags = sequence_duration_in_ms / resolution_ms
    smoothing_window : int or float, optional
        The window (in milliseconds) within which a normal probability density function is used for
        smoothing out the analysis.
    smoothing_sd : int or float, optional
        The standard deviation of the normal probability density function used for smoothing out the analysis.
    style : str, optional
        A matplotlib style. See the matplotlib docs for options. Defaults to 'seaborn'.
    title : str, optional
        If desired, one can provide a title for the plot. This takes precedence over using the
        Sequence or StimSequence name as the title of the plot (if passed and the object has one).
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

    Notes
    -----
    This function is based on the procedure described in Ravignani and Norton (2017) [2]_. There, one can also find a
    more detailed description of the smoothing procedure.

    References
    ----------
    .. [2] Ravignani, A., & Norton, P. (2017). Measuring rhythmic complexity:
       a primer to quantify and compare temporal structure in speech, movement, and animal vocalizations.
       Journal of Language Evolution, 2(1), 4-19.
       https://doi.org/10.1093/jole/lzx002

    """

    if isinstance(sequence, (core.sequence.Sequence, core.stimsequence.StimSequence)):
        onsets_ms = sequence.onsets
    else:
        onsets_ms = sequence

    correlation = acf_values(sequence=sequence,
                             resolution_ms=resolution_ms,
                             smoothing_window=smoothing_window,
                             smoothing_sd=smoothing_sd)

    x_step = resolution_ms
    max_lag = np.floor(np.max(onsets_ms))

    # plot
    y = correlation[:int(max_lag)]
    y = y / max(y)  # normalize

    # Make x axis
    x = np.arange(start=0, stop=y.size * x_step, step=x_step)

    # Do seconds instead of milliseconds above 10s
    if np.max(x) > 10000:
        x = x / 1000
        x_label = "Lag [s]"
    else:
        x_label = "Lag [ms]"

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.axes.set_xlabel(x_label)
        ax.axes.set_title(title)
        ax.plot(x, y)

    if suppress_display is False:
        plt.show()

    return fig, ax


def acf_values(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable],
               resolution_ms: int = 1,
               smoothing_window: Union[int, float] = 0,
               smoothing_sd: Union[int, float] = 0) -> np.ndarray:
    """

    This function takes a Sequence or StimSequence object, or a list of event onsets, and returns
    an array with steps of resolution_ms of unstandardized correlation factors.

    Parameters
    ----------
    sequence : Sequence, StimSequence or iterable
        Either a Sequence or StimSequence object, or an iterable containing event onsets in milliseconds,
        e.g. [0, 500, 1000]
    resolution_ms : int, optional
        The temporal resolution in milliseconds (i.e. sampling frequency). The number of lags calculated for the
        autocorrelation function can be calculated as n_lags = sequence_duration_in_ms / resolution_ms
    smoothing_window : int or float, optional
        The window (in milliseconds) within which a normal probability density function is used for
        smoothing out the analysis.
    smoothing_sd : int or float, optional
        The standard deviation of the normal probability density function used for smoothing out the analysis.

    Returns
    -------
    numpy.ndarray
        An array containing the autocorrelation function (i.e. the correlation factors by resolution/step size).

    Notes
    -----
    This function is based on the procedure described in Ravignani and Norton (2017) [3]_. There, one can also find a
    more detailed description of the smoothing procedure. This function uses the ``numpy.correlate`` [4]_ function
    to calculate the correlations.

    References
    ----------
    .. [3] Ravignani, A., & Norton, P. (2017). Measuring rhythmic complexity:
       a primer to quantify and compare temporal structure in speech, movement, and animal vocalizations.
       Journal of Language Evolution, 2(1), 4-19.
       https://doi.org/10.1093/jole/lzx002
    .. [4] https://numpy.org/doc/stable/reference/generated/numpy.correlate.html

    """

    if isinstance(sequence, (core.sequence.Sequence, core.stimsequence.StimSequence)):
        onsets_ms = sequence.onsets
    else:
        onsets_ms = sequence

    signal = _make_ones_and_zeros_timeseries(onsets_ms, resolution_ms)

    # npdf
    if not smoothing_window == 0 and not smoothing_sd == 0:
        x = np.arange(start=-smoothing_window / 2, stop=smoothing_window / 2, step=resolution_ms)
        npdf = scipy.stats.norm.pdf(x, 0, smoothing_sd)
        npdf = npdf / np.max(npdf)
        signal_convoluted = np.convolve(signal, npdf)
        signal = signal_convoluted[round(resolution_ms * smoothing_window / 2):]

    try:
        correlation = np.correlate(signal, signal, 'full')
        correlation = correlation[round(len(correlation) / 2) - 1:]
    except ValueError as e:
        raise ValueError("Error! Hint: Most likely your resolution_ms is too large for the chosen smoothing_window"
                         "and smoothing_sd. Try choosing a smaller resolution_ms.") from e

    return correlation


def ks_test(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable],
            reference_distribution: str = 'normal',
            alternative: str = 'two-sided'):
    """
    This function returns the `D` statistic and `p` value of a one-sample Kolmogorov-Smirnov test.
    It calculates how different the distribution of inter-onset intervals (IOIs) is compared to the provided reference
    distribution.

    If `p` is significant it means that the IOIs are not distributed according to the provided reference distribution.

    Parameters
    ----------
    sequence : Sequence, StimSequence or iterable
        Either a Sequence or StimSequence object or an iterable (e.g. list) containing inter-onset intervals
        (IOIs), e.g. [500, 500, 500].
    reference_distribution : str, optional
        Either 'normal' or 'uniform'. The distribution against which the distribution of inter-onset intervals (IOIs)
        is compared. The default is 'normal'.
    alternative: str, optional
        Either ‘two-sided’, ‘less’ or ‘greater’. See the SciPy documentation [5]_ for more information.

    Returns
    -------
    scipy.stats._stats_py.KstestResult
        A SciPy named tuple containing the `D` statistic and the `p` value.

    Notes
    -----
    This function uses the ``scipy.stats.kstest`` function [5]_. For more information about the use of the
    Kolmogorov-Smirnov test in rhythm research, see Jadoul et al. (2016) [6]_ and Ravignani and Norton (2017) [7]_.

    References
    ----------
    .. [5] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html

    .. [6] Jadoul, Y., Ravignani, A., Thompson, B., Filippi, P. and de Boer, B. (2016).
       Seeking Temporal Predictability in Speech: Comparing Statistical Approaches on 18 World Languages’.
       Frontiers in Human Neuroscience, 10(586), 1–15.

    .. [7] Ravignani, A., & Norton, P. (2017). Measuring rhythmic complexity:
       A primer to quantify and compare temporal structure in speech, movement,
       and animal vocalizations. Journal of Language Evolution, 2(1), 4-19.

    Examples
    --------
    >>> rng = np.random.default_rng(seed=123)
    >>> seq = Sequence.generate_random_normal(n=100, mu=500, sigma=25, rng=rng)
    >>> print(ks_test(seq))
    KstestResult(statistic=0.0770618842759333, pvalue=0.5722913668564746)

    """

    if isinstance(sequence, (core.sequence.Sequence, core.stimsequence.StimSequence)):
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


def get_npvi(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable]) -> np.float32:
    """

    This function calculates the normalized pairwise variability index (nPVI) for a provided Sequence or StimSequence
    object, or for an interable of inter-onset intervals (IOIs).

    Parameters
    ----------
    sequence : Sequence, StimSequence or iterable
        Either a Sequence or StimSequence object, or an iterable containing inter-onset intervals (IOIs).

    Returns
    -------
    np.float32
        The nPVI for the provided sequence.

    Notes
    -----
    The normalied pairwise variability index (nPVI) is a measure of the variability of adjacent temporal intervals.
    The nPVI is zero for sequences that are perfectly isochronous. See Jadoul et al. (2016) [8]_ and
    Ravignani and Norton (2017) [9]_ for more information on its use in rhythm research.

    .. [8] Jadoul, Y., Ravignani, A., Thompson, B., Filippi, P. and de Boer, B. (2016).
       Seeking Temporal Predictability in Speech: Comparing Statistical Approaches on 18 World Languages’.
       Frontiers in Human Neuroscience, 10(586), 1–15.

    .. [9] Ravignani, A., & Norton, P. (2017). Measuring rhythmic complexity:
       A primer to quantify and compare temporal structure in speech, movement,
       and animal vocalizations. Journal of Language Evolution, 2(1), 4-19.

    Examples
    --------
    >>> seq = Sequence.generate_isochronous(n=10, ioi=500)
    >>> print(get_npvi(seq))
    0.0

    >>> rng = np.random.default_rng(seed=123)
    >>> seq = Sequence.generate_random_normal(n=10, mu=500, sigma=50, rng=rng)
    >>> print(get_npvi(seq))
    37.481644
    """

    if isinstance(sequence, (core.sequence.Sequence, core.stimsequence.StimSequence)):
        iois = sequence.iois
    else:
        iois = np.array(sequence)

    npvi_values = []

    for i in range(1, len(iois)):
        diff = iois[i] - iois[i - 1]
        mean = np.mean(iois[i] + iois[i - 1])
        npvi_values.append(np.abs(diff / mean))

    npvi = np.mean(npvi_values) * (100 * (iois.size - 1))

    return np.float32(npvi)


def get_ugof(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable],
             theoretical_ioi: Union[int, float],
             output_statistic: str = 'mean') -> np.float32:
    """

    This function calculates the universal goodness of fit (`ugof`) measure for a sequence compared to a
    theoretical/underlying inter-onset interval (IOI). The `ugof` statistic quantifies how well a theoretical IOI
    describes a sequence.

    Parameters
    ----------
    sequence : Sequence, StimSequence or iterable
        Either a Sequence or StimSequence object, or an iterable (e.g. list) containing event onsets,
        e.g. [0, 500, 1000].
    theoretical_ioi : int or float
        The theoretical/underlying inter-onset interval (IOI) to which the sequence is compared.
    output_statistic : str
        Either 'mean' (the default) or 'median'. This determines whether for the individual ugof values we take the mean
        or the median as the output statistic.

    Returns
    -------
    np.float32
        The ugof statistic.

    Notes
    -----
    This measure is described in Burchardt et al. (2021) [10]_.

    References
    ----------
    .. [10] Burchardt, L. S., Briefer, E. F., & Knörnschild, M. (2021).
       Novel ideas to further expand the applicability of rhythm analysis.
       Ecology and evolution, 11(24), 18229-18237.
       https://doi.org/10.1002/ece3.8417

    Examples
    --------
    >>> seq = Sequence.generate_isochronous(n=10, ioi=1000)
    >>> ugof = get_ugof(seq, theoretical_ioi=68.21)
    >>> print(ugof)
    0.59636414

    """

    # Get the onsets
    if isinstance(sequence, (core.sequence.Sequence, core.stimsequence.StimSequence)):
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

