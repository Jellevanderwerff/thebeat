from mingus.containers import Bar, Track
from mingus.extra import lilypond
import numpy as np
from stimulus import Sequence
import random
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess


class Rhythm(Sequence):

    def __init__(self, iois, n_bars, time_sig, quarternote_ms):
        # Save attributes
        self.time_sig = time_sig  # Used for metrical sequences
        self.quarternote_ms = quarternote_ms  # Used for metrical sequences
        self.n_bars = n_bars

        # Call initializer of super class
        Sequence.__init__(self, iois, metrical=True)

    def __str__(self):
        return f"Object of type Rhythm. Time signature: {self.time_sig}\nNumber of bars: {self.n_bars}\nQuarternote (" \
               f"ms): {self.quarternote_ms}\nNumber of events: {len(self.onsets)}\nIOIs: {self.iois}\n" \
               f"Onsets:{self.onsets}\n "

    @property
    def note_values(self):
        """
        Get note values from the IOIs, based on quarternote_ms.
        """
        if not self.metrical or not self.time_sig or not self.quarternote_ms:
            raise ValueError("This is not a rhythmic sequence. Use class method Sequence.from_note_values or e.g."
                             "random_rhythmic_sequence(). Alternatively, you can set the following properties manually: "
                             "Sequence.metrical (boolean), Sequence.time_sig (tuple), Sequence.n_bars (int).")

        ratios = self.iois / self.quarternote_ms / 4

        note_values = np.array([1 // ratio for ratio in ratios])

        return note_values

    @classmethod
    def from_note_values(cls, note_values, time_signature, quarternote_ms):
        ratios = np.array([1 / note * time_signature[1] for note in note_values])

        n_bars = np.sum(ratios) / time_signature[0]

        if n_bars % 1 != 0:
            raise ValueError("The provided note values do not amount to whole bars.")
        else:
            n_bars = int(n_bars)

        iois = ratios * quarternote_ms

        return cls(iois,
                   time_sig=time_signature,
                   quarternote_ms=quarternote_ms,
                   n_bars=n_bars)

    def plot_rhythm(self, out_filepath=None):

        if out_filepath:
            location, filename = os.path.split(out_filepath)
            if location == '':
                location = '.'
        else:
            location = '.'
            filename = 'temp.png'

        # We want to split up the notes into bars first

        t = Track()

        # create initial bar
        b = Bar(meter=self.time_sig)

        # keep track of the index of the note_value
        note_i = 0
        # loop over the note values of the sequence

        for note_value in self.note_values:
            b.place_notes('G-4', self.note_values[note_i])
            # if bar is full, create new bar and add bar to track
            if b.current_beat == b.length:
                t.add_bar(b)
                b = Bar(meter=self.time_sig)

            note_i += 1

        # If final bar was not full yet, add a rest for the remaining duration
        if b.current_beat % 1 != 0:
            rest_value = 1/b.space_left()
            if round(rest_value) != rest_value:
                raise ValueError("The rhythm could not be plotted. Most likely because the IOIs cannot "
                                 "be (easily) captured in musical notation. This for instance happens when "
                                 "using one of the tempo manipulation methods.")

            b.place_rest(rest_value)
            t.add_bar(b)

        # make lilypond string
        lp = '\\version "2.10.33"\n' + lilypond.from_Track(t) + '\n\paper {\nindent = 0\mm\nline-width = ' \
                                                                '110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ' \
                                                                '""\noddFooterMarkup = ""\nevenFooterMarkup = ""\n} '

        # write lilypond string to file
        with open(os.path.join(location, filename[:-4]+'.ly'), 'w') as file:
            file.write(lp)

        # run subprocess
        if filename.endswith('.eps'):
            command = f'lilypond -dbackend=eps --silent -dresolution=600 --eps -o {filename[:-4]} {filename[:-4] + ".ly"}'
            to_be_removed = ['.ly']
        elif filename.endswith('.png'):
            command = f'lilypond -dbackend=eps --silent -dresolution=600 --png -o {filename[:-4]} {filename[:-4] + ".ly"}'
            to_be_removed = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi', '.ly']
        else:
            raise ValueError("Can only export .png or .eps files.")

        p = subprocess.Popen(command, shell=True, cwd=location).wait()

        # show plot
        if not out_filepath:
            img = mpimg.imread(os.path.join(location, filename))
            plt.imshow(img)
            plt.axis('off')
            plt.show()

        # remove files
        if out_filepath:
            filenames = [filename[:-4] + x for x in to_be_removed]
        else:
            to_be_removed = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi', '.ly', '.png']
            filenames = ['temp' + x for x in to_be_removed]

        for file in filenames:
            os.remove(os.path.join(location, file))


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
    full_bar = time_signature[0] * (1 / time_signature[1])
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

    return Rhythm(iois, time_sig=time_signature, quarternote_ms=quarternote_ms, n_bars=n_bars)

