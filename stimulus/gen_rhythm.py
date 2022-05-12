import numpy as np


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

    out_list = [(np.array(result) / common_denom) * (time_signature[1]/4) for result in
                _all_possibilities(allowed_numerators, target)]

    return out_list


def rrhythm_sequence(allowed_note_values_denoms, time_signature, ioi):
    pass