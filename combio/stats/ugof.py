from combio.core import Sequence, StimTrial
from typing import Union
import numpy as np


def ugof(sequence: Union[Sequence, StimTrial],
         theoretical_ioi: float,
         output: str = 'mean') -> np.float32:

    """Credits to Lara. S. Burchardt, include ref."""

    # Get the onsets
    onsets = sequence.onsets  # in ms

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
