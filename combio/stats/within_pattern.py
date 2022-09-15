from typing import Iterable, Union
import scipy.stats
import numpy as np
from combio.core import Sequence, StimSequence
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.tsa.stattools
import matplotlib.pyplot as plt
from .helpers import make_ones_and_zeros_timeseries

# todo All these functions need to be checked!!


def acf_plot(sequence: Union[Sequence, StimSequence, Iterable],
             resolution_ms: int = 1,
             npdf_width=None,
             npdf_sd=None,
             style: str = 'seaborn',
             figsize: tuple = None,
             suppress_display: bool = False):
    """
    Based on Ravignani & Norris, 2017
    Please provide a Sequence or StimSequence object, or an iterable containing stimulus onsets in milliseconds.
    """

    if npdf_width is None or npdf_sd is None:
        raise ValueError("Please provide a width and sd for the normal probability density function that"
                         "is used for smoothing out the found IOIs.")

    if isinstance(sequence, (Sequence, StimSequence)):
        onsets_ms = sequence.onsets
    else:
        onsets_ms = sequence

    signal = make_ones_and_zeros_timeseries(onsets_ms, resolution_ms)

    # npdf
    x = np.arange(start=-npdf_width / 2, stop=npdf_width / 2, step=resolution_ms)
    npdf = scipy.stats.norm.pdf(x, 0, npdf_sd)
    npdf = npdf / np.max(npdf)

    signal_convoluted = np.convolve(signal, npdf)
    signal = signal_convoluted[round(resolution_ms * npdf_width / 2):]

    correlation = np.correlate(signal, signal, 'full')
    correlation = correlation[round(len(correlation)/2):]

    x_step = resolution_ms
    max_lag = np.floor(max(onsets_ms) / 2)

    # plot
    x = np.arange(start=x_step, stop=max_lag+1, step=x_step)
    y = correlation[0:int(np.floor(max_lag*resolution_ms))]
    y = y / max(y)  # normalize
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.axes.set_xlabel('Lag [ms]')
        ax.axes.set_title('Autocorrelation')
        ax.plot(x, y)

    if suppress_display is False:
        plt.show()

    return fig, ax


def ks_test(sequence: Union[Sequence, StimSequence, Iterable],
            reference_distribution: str = 'normal',
            alternative: str = 'two-sided'):
    """
    This function returns the D statistic and p value of a one-sample Kolmogorov-Smirnov test.
    It calculates how different the supplied values are from a provided reference distribution.

    If p is significant that means that the iois are not distributed according to the provided reference distribution.

    For reference, see:
    Jadoul, Y., Ravignani, A., Thompson, B., Filippi, P. and de Boer, B. (2016).
        Seeking Temporal Predictability in Speech: Comparing Statistical Approaches on 18 World Languages’.
        Frontiers in Human Neuroscience, 10(586), 1–15.
    Ravignani, A., & Norton, P. (2017). Measuring rhythmic complexity:
        A primer to quantify and compare temporal structure in speech, movement,
        and animal vocalizations. Journal of Language Evolution, 2(1), 4-19.


    """

    if isinstance(sequence, (Sequence, StimSequence)):
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


def npvi(sequence: Union[Sequence, StimSequence, Iterable]) -> np.float32:
    """Get nPVI
    """

    if isinstance(sequence, (Sequence, StimSequence)):
        sequence = sequence.iois

    npvi_values = []

    for i in range(1, len(sequence)):
        diff = sequence[i] - sequence[i - 1]
        mean = np.mean(sequence[i] + sequence[i - 1])
        npvi_values.append(diff / mean)

    return np.float32(np.mean(npvi_values))


def ugof(sequence: Union[Sequence, StimSequence, Iterable],
         theoretical_ioi: float,
         output: str = 'mean') -> np.float32:
    """Credits to Lara. S. Burchardt, include ref."""

    # Get the onsets
    if isinstance(sequence, (Sequence, StimSequence)):
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

