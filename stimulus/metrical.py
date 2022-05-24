import numpy as np
from stimulus import Sequence
import random


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


def _all_metrical_ratios(allowed_note_values, time_signature=(4, 4)):
    common_denom = np.lcm(np.lcm.reduce(allowed_note_values), time_signature[1])

    allowed_numerators = common_denom // np.array(allowed_note_values)
    target = int((time_signature[0] / time_signature[1]) * common_denom)

    out_list = [(np.array(result) / common_denom) * (time_signature[1] / 4) for result in
                _all_possibilities(allowed_numerators, target)]

    return out_list


def random_metrical_sequence(allowed_note_values, time_signature, quarternote_ms):
    """
    This function returns a StimulusSequence based on a randomly generated list
    of integer ratios, based on the given parameters.

    """

    ratios = random.choice(_all_metrical_ratios(allowed_note_values, time_signature))
    # todo Check whether this makes sense w.r. to the time signature
    iois = np.round(ratios * quarternote_ms * time_signature[1])

    return Sequence(iois, metrical=True)


def metrical_sequence(ratios, time_signature, quarternote_ms):
    """
    This function should return a Sequence object given a list of ratios etc.
    """
    iois = np.round(ratios * quarternote_ms * time_signature[1])

    return Sequence(iois, metrical=True)



