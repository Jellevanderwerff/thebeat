from typing import Iterable, Union
import scipy.stats
import scipy.fft
import numpy as np
from . import core
import matplotlib.pyplot as plt
import scipy.signal
import pandas as pd


def acf_values(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable],
               resolution_ms: int = 1,
               smoothe_width: Union[int, float] = 0,
               smoothe_sd: Union[int, float] = 0):
    """From Ravignani & Norton, 2017"""

    if isinstance(sequence, (core.sequence.Sequence, core.stimsequence.StimSequence)):
        onsets_ms = sequence.onsets
    else:
        onsets_ms = sequence

    signal = _make_ones_and_zeros_timeseries(onsets_ms, resolution_ms)

    # npdf
    if not smoothe_width == 0 and not smoothe_sd == 0:
        x = np.arange(start=-smoothe_width / 2, stop=smoothe_width / 2, step=resolution_ms)
        npdf = scipy.stats.norm.pdf(x, 0, smoothe_sd)
        npdf = npdf / np.max(npdf)

        signal_convoluted = np.convolve(signal, npdf)
        signal = signal_convoluted[round(resolution_ms * smoothe_width / 2):]

    correlation = np.correlate(signal, signal, 'full')
    correlation = correlation[round(len(correlation) / 2) - 1:]

    return correlation


def acf_df(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable],
           resolution_ms: int = 1,
           smoothe_width: Union[int, float] = 0,
           smoothe_sd: Union[int, float] = 0):
    """
    This function returns a Pandas DataFrame with timepoints and correlation factors.

    """

    correlations = acf_values(sequence, resolution_ms, smoothe_width, smoothe_sd)
    correlations = correlations / max(correlations)  # normalize
    times_ms = np.arange(start=0, stop=correlations.size)
    df = pd.DataFrame(
        {
            "time_ms": times_ms,
            "correlation": correlations
        }
    )

    return df


def acf_plot(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable],
             resolution_ms: int = 1,
             smoothe_width: Union[int, float] = 0,
             smoothe_sd: Union[int, float] = 0,
             style: str = 'seaborn',
             title: str = 'Autocorrelation',
             figsize: tuple = None,
             suppress_display: bool = False):
    """
    Based on Ravignani & Norton, 2017
    Please provide a core.sequence.Sequence or core.stimsequence.StimSequence object, or an iterable containing stimulus onsets in milliseconds.
    """

    if isinstance(sequence, (core.sequence.Sequence, core.stimsequence.StimSequence)):
        onsets_ms = sequence.onsets
    else:
        onsets_ms = sequence

    correlation = acf_values(sequence, resolution_ms, smoothe_width, smoothe_sd)

    x_step = resolution_ms
    max_lag = np.floor(max(onsets_ms))

    # plot
    x = np.arange(start=0, stop=max_lag, step=x_step)
    y = correlation[:int(max_lag)]
    y = y / max(y)  # normalize
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.axes.set_xlabel('Lag [ms]')
        ax.axes.set_title(title)
        ax.plot(x, y)

    if suppress_display is False:
        plt.show()

    return fig, ax


def ks_test(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable],
            reference_distribution: str = 'normal',
            alternative: str = 'two-sided'):
    """
    This function returns the D statistic and p value of a one-sample Kolmogorov-Smirnov test.
    It calculates how different the supplied values are from a provided reference distribution.

    If p is significant that means that the iois are not distributed according to the provided reference distribution.

    References
    ----------
    Jadoul, Y., Ravignani, A., Thompson, B., Filippi, P. and de Boer, B. (2016).
    Seeking Temporal Predictability in Speech: Comparing Statistical Approaches on 18 World Languages’.
    Frontiers in Human Neuroscience, 10(586), 1–15.

    Ravignani, A., & Norton, P. (2017). Measuring rhythmic complexity:
    A primer to quantify and compare temporal structure in speech, movement,
    and animal vocalizations. Journal of Language Evolution, 2(1), 4-19.
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
    """Get nPVI
    """

    if isinstance(sequence, (core.sequence.Sequence, core.stimsequence.StimSequence)):
        sequence = sequence.iois

    npvi_values = []

    for i in range(1, len(sequence)):
        diff = sequence[i] - sequence[i - 1]
        mean = np.mean(sequence[i] + sequence[i - 1])
        npvi_values.append(diff / mean)

    return np.float32(np.mean(npvi_values))


def get_ugof(sequence: Union[core.sequence.Sequence, core.stimsequence.StimSequence, Iterable],
             theoretical_ioi: float,
             output: str = 'mean') -> np.float32:
    """Credits to Lara. S. Burchardt, include ref."""

    """ugof is only for isochronous sequences?"""

    # Get the onsets
    if isinstance(sequence, (core.sequence.Sequence, core.stimsequence.StimSequence)):
        onsets = sequence.onsets  # in ms
    else:
        onsets = sequence

    # Make a theoretical sequence that's multiples of the theoretical_ioi (the + one is because arange doesn't include
    # the stop)
    theo_seq = np.arange(start=0, stop=max(onsets) + 1, step=theoretical_ioi)

    # For each onset, get the closest theoretical beat and get the absolute difference
    minimal_deviations = np.array([min(abs(onsets[n] - theo_seq)) for n in range(len(onsets))])
    maximal_deviation = theoretical_ioi / 2  # in ms

    # calculate ugofs
    ugof_values = minimal_deviations / maximal_deviation

    if output == 'mean':
        return np.float32(np.mean(ugof_values))
    elif output == 'median':
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
