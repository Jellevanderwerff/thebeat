import numpy as np
from stimulus import Sequence
import random
from functools import lru_cache


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


def _all_rhythms(allowed_note_values_denoms, time_signature=(4, 4)):
    common_denom = np.lcm(np.lcm.reduce(allowed_note_values_denoms), time_signature[1])

    allowed_numerators = common_denom // np.array(allowed_note_values_denoms)
    target = int((time_signature[0] / time_signature[1]) * common_denom)

    out_list = [(np.array(result) / common_denom) * (time_signature[1] / 4) for result in
                _all_possibilities(allowed_numerators, target)]

    return out_list


def rrhythmic_sequence(allowed_note_values, time_signature, n_rests, ioi):
    """
    This function should return a SoundSequence, where:
        - A random rhythm is generated using _all_rhythms
        - The IOIs are calculated and then the onsets
        - The onsets minus the last one are used (because the final one - the pre-final one = note duration of final
          sound.
        - At random places rests are inserted (i.e. silences)
        - returns a one-bar SoundSequence

    """


sequence = rrhythmic_sequence([4], (4, 4), ioi=500)
print(sequence)
