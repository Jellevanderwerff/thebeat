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


def _all_metrical_ratios(allowed_note_values, time_signature=(4, 4)):
    common_denom = np.lcm(np.lcm.reduce(allowed_note_values), time_signature[1])

    allowed_numerators = common_denom // np.array(allowed_note_values)
    target = int((time_signature[0] / time_signature[1]) * common_denom)

    out_list = [(np.array(result) / common_denom) * (time_signature[1] / 4) for result in
                _all_possibilities(allowed_numerators, target)]

    return out_list


def random_metrical_sequence(n_bars, allowed_note_values, time_signature, quarternote_ms):
    """
    This function returns a randomly generated integer ratio Sequence on the basis of the provided params.
    """

    iois = np.empty(0)

    for bar in range(n_bars):
        ratios = random.choice(_all_metrical_ratios(allowed_note_values, time_signature))
        iois = np.concatenate((iois, np.round(ratios * quarternote_ms * time_signature[1])), axis=0)

    return Sequence(iois, metrical=True)


def metrical_sequence(ratios, time_signature, quarternote_ms):
    """
    This function should return a Sequence object given a list of ratios etc.
    """

    ratios = np.array(ratios)

    if np.sum(ratios) % (time_signature[0]/time_signature[1]) != 0:
        warnings.warn("The provided ratios do not result in a sequence with only whole bars.")

    iois = np.round(ratios * quarternote_ms * time_signature[1])

    return Sequence(iois, metrical=True)


def iois_to_ratios(iois, time_signature, quarternote_ms):
    iois = np.array(iois)

    return iois / quarternote_ms / time_signature[1]


def iois_to_notevalues(iois, time_signature, quarternote_ms):
    iois = np.array(iois)
    ratios = iois / quarternote_ms / time_signature[1]

    note_values = np.array([1 // ratio for ratio in ratios])

    return note_values


def plot_note_values(filename, note_values, time_signature):

    b = Bar(meter=time_signature)

    for note_value in note_values:
        b.place_notes('G-4', note_value)

    lp = lilypond.from_Bar(
        b) + '\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""\noddFooterMarkup = ""\nevenFooterMarkup = ""\n}'

    lilypond.save_string_and_execute_LilyPond(lp, filename, '-dbackend=eps -dresolution=600 --png -s')

    filenames = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi']
    filenames = [filename[:-4] + x for x in filenames]

    for file in filenames:
        os.remove(file)

    img = mpimg.imread(filename)
    plt.imshow(img)
    plt.axis('off')
    plt.show()







