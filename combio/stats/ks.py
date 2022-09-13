from typing import Iterable
import scipy.stats
import numpy as np
from combio.core import Sequence

def ks_test(iois: Iterable,
            reference_distribution: str = 'normal',
            alternative: str = 'two-sided'):
    """
    This function returns the D statistic and p value of a one-sample Kolmogorov-Smirnov test.
    It calculates how different the supplied values are from a provided distribution.

    If p is significant that means that the iois are not distributed according to the provided reference distribution.

    For reference, see:
    Jadoul, Y., Ravignani, A., Thompson, B., Filippi, P. and de Boer, B. (2016).
        Seeking Temporal Predictability in Speech: Comparing Statistical Approaches on 18 World Languages’.
        Frontiers in Human Neuroscience, 10(586), 1–15.
    Ravignani, A., & Norton, P. (2017). Measuring rhythmic complexity:
        A primer to quantify and compare temporal structure in speech, movement,
        and animal vocalizations. Journal of Language Evolution, 2(1), 4-19.


    """
    if reference_distribution == 'normal':
        mean = np.mean(iois)
        sd = np.std(iois)
        dist = scipy.stats.norm(loc=mean, scale=sd).cdf
        return scipy.stats.kstest(iois, dist, alternative=alternative)
    elif reference_distribution == 'uniform':
        a = min(iois)
        b = max(iois)
        scale = b - a
        dist = scipy.stats.uniform(loc=a, scale=scale).cdf
        return scipy.stats.kstest(iois, dist, alternative=alternative)
    else:
        raise ValueError("Unknown distribution. Choose 'normal' or 'uniform'.")

