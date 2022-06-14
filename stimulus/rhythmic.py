from mingus.containers import Bar
from mingus.extra import lilypond
import numpy as np
from stimulus import Sequence
import random
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def _all_possibilities(nums, target):
    """
    I stole this code
    """
    res = []
    nums.sort()

    def dfs(left, path):
        if not left:
            res.append(path)
        else:
            for val in nums:
                if val > left:
                    break
                dfs(left - val, path + [val])

    dfs(target, [])

    return res


def _all_rhythmic_ratios(allowed_note_values, time_signature):
    common_denom = np.lcm(np.lcm.reduce(allowed_note_values), time_signature[1])

    allowed_numerators = common_denom // np.array(allowed_note_values)
    full_bar = time_signature[0] * (1/time_signature[1])
    target = full_bar * common_denom

    all_possibilities = _all_possibilities(allowed_numerators, target)

    out_list = [(np.array(result) / common_denom) for result in
                all_possibilities]

    return out_list


def random_rhythmic_sequence(n_bars, allowed_note_values, time_signature, quarternote_ms):
    """
    This function returns a randomly generated integer ratio Sequence on the basis of the provided params.
    """

    iois = np.empty(0)

    for bar in range(n_bars):
        all_rhythmic_ratios = _all_rhythmic_ratios(allowed_note_values, time_signature)
        ratios = random.choice(all_rhythmic_ratios)

        new_iois = ratios * 4 * quarternote_ms

        iois = np.concatenate((iois, new_iois), axis=None)

    return Sequence(iois, metrical=True, time_sig=time_signature, quarternote_ms=quarternote_ms, n_bars=n_bars)


