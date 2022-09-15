import numpy as np


def make_ones_and_zeros_timeseries(onsets_ms, resolution_ms):
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


