import numpy as np
from stimulus import Stimulus, Sequence, StimulusSequence
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
    iois = ratios * quarternote_ms * time_signature[1]

    return Sequence(iois, metrical=True)


if __name__ == "__main__":
    # Example of a random metrical sequence in 4/4
    stim = Stimulus.generate()
    # We pass a list of allowed note values (in the example they are
    # 1/4th, 1/8th, and 1/16th notes) and a time signature (as a tuple)
    rand_met_sequence = random_metrical_sequence([4, 8, 16], (3, 4), quarternote_ms=500)
    print(rand_met_sequence)

    rand_met_stimsequence = StimulusSequence(stim, rand_met_sequence)
    rand_met_stimsequence.plot()
    rand_met_stimsequence.write_wav('random_met_seq.wav')

