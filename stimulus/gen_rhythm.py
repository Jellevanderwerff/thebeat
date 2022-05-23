import numpy as np
from stimulus import Stimulus, Sequence, StimulusSequence
from datetime import datetime
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


def _all_rhythms(allowed_note_values, time_signature=(4, 4)):
    common_denom = np.lcm(np.lcm.reduce(allowed_note_values), time_signature[1])

    allowed_numerators = common_denom // np.array(allowed_note_values)
    target = int((time_signature[0] / time_signature[1]) * common_denom)

    out_list = [(np.array(result) / common_denom) * (time_signature[1] / 4) for result in
                _all_possibilities(allowed_numerators, target)]

    return out_list


def rrhythmic_sequence(stim, allowed_note_values, time_signature, ioi):
    """
    This function should return a SoundSequence, where:
        - A random rhythm is generated using _all_rhythms
        - returns a one-bar SoundSequence

    """

    ratios = random.choice(_all_rhythms(allowed_note_values, time_signature))
    print(ratios)
    # todo Reasses whether this makes sense with the time_sign
    iois = ratios * ioi * time_signature[1]

    return StimulusSequence(stim, Sequence(iois, metrical=True))


if __name__ == "__main__":
    stim = Stimulus.generate(duration=50)
    rhythm = rrhythmic_sequence(stim, [4, 8], (4, 4), ioi=500)
    print(rhythm)
    rhythm.plot()
    rhythm.write_wav('random_rhythm.wav')








