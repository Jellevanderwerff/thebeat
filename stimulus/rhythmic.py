from mingus.containers import Bar, Track, Instrument
from mingus.extra import lilypond
import numpy as np
from stimulus import Sequence
from stimulus.base import _plot_lp
import random


class Rhythm(Sequence):

    def __init__(self, iois, n_bars, time_sig, quarternote_ms, played):
        # Save attributes
        self.time_sig = time_sig  # Used for metrical sequences
        self.quarternote_ms = quarternote_ms  # Used for metrical sequences
        self.n_bars = n_bars
        self.played = played

        # Call initializer of super class
        Sequence.__init__(self, iois, metrical=True, played=played)

    def __str__(self):
        return f"Object of type Rhythm.\nTime signature: {self.time_sig}\nNumber of bars: {self.n_bars}\nQuarternote (" \
               f"ms): {self.quarternote_ms}\nNumber of events: {len(self.onsets)}\nIOIs: {self.iois}\n" \
               f"Onsets:{self.onsets}\nOnsets played: {self.played}"

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
    def from_note_values(cls, note_values, time_signature, quarternote_ms, played=None):
        """
        Almost same as standard initialization, except that we don't provide the number of bars but calculate those.

        """
        if played is None:
            played = [True] * len(note_values)
        elif len(played) == len(note_values):
            played = played
        else:
            raise ValueError("The 'played' argument should contain an equal number of "
                             "booleans as the number of note_values.")

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
                   n_bars=n_bars,
                   played=played)

    @classmethod
    def from_iois(cls, iois, time_signature, quarternote_ms, played=None):
        if played is None:
            played = [True] * len(iois)
        elif len(played) == len(iois):
            played = played
        else:
            raise ValueError("The 'played' argument should contain an equal number of "
                             "booleans as the number of note_values.")

        note_values = np.array([ioi / quarternote_ms * time_signature[1] for ioi in iois])

        n_bars = np.sum(note_values) / time_signature[0]

        if n_bars % 1 != 0:
            raise ValueError("The provided note values do not amount to whole bars.")
        else:
            n_bars = int(n_bars)

        return cls(iois,
                   time_sig=time_signature,
                   quarternote_ms=quarternote_ms,
                   n_bars=n_bars,
                   played=played)

    def plot_rhythm(self, out_filepath=None):

        # create initial bar
        t = Track()
        b = Bar(meter=self.time_sig)

        # keep track of the index of the note_value
        note_i = 0
        # loop over the note values of the sequence

        values_and_played = list(zip(self.note_values, self.played))

        for note_value, played in values_and_played:
            if played:
                b.place_notes('G-4', self.note_values[note_i])
            elif not played:
                b.place_rest(self.note_values[note_i])

            # if bar is full, create new bar and add bar to track
            if b.current_beat == b.length:
                t.add_bar(b)
                b = Bar(meter=self.time_sig)

            note_i += 1

        # If final bar was not full yet, add a rest for the remaining duration
        if b.current_beat % 1 != 0:
            rest_value = 1 / b.space_left()
            if round(rest_value) != rest_value:
                raise ValueError("The rhythm could not be plotted. Most likely because the IOIs cannot "
                                 "be (easily) captured in musical notation. This for instance happens when "
                                 "using one of the tempo manipulation methods.")

            b.place_rest(rest_value)
            t.add_bar(b)

        _plot_lp(t, out_filepath)


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


def random_rhythmic_sequence(n_bars, allowed_note_values, time_signature, quarternote_ms, random_rests=False):
    """
    This function returns a randomly generated integer ratio Sequence on the basis of the provided params.
    """

    iois = np.empty(0)

    for bar in range(n_bars):
        all_rhythmic_ratios = _all_rhythmic_ratios(allowed_note_values, time_signature)
        ratios = random.choice(all_rhythmic_ratios)

        new_iois = ratios * 4 * quarternote_ms

        iois = np.concatenate((iois, new_iois), axis=None)

    if random_rests:
        played = random.choices([True, False], k=len(iois))
    else:
        played = [True] * len(iois)

    return Rhythm(iois, time_sig=time_signature, quarternote_ms=quarternote_ms, n_bars=n_bars, played=played)


def join_rhythms(iterator):
    """
    This function can join multiple Rhythm objects.
    """

    # Check whether iterable was passed
    if not hasattr(iterator, '__iter__'):
        raise ValueError("Please pass this function a list or other iterable object.")

    # Check whether all the objects are of the same type
    if not all(isinstance(rhythm, Rhythm) for rhythm in iterator):
        raise ValueError("This function can only join multiple Rhythm objects.")

    if not all(rhythm.time_sig == iterator[0].time_sig for rhythm in iterator):
        raise ValueError("Provided rhythms should have the same time signatures.")

    if not all(rhythm.quarternote_ms == iterator[0].quarternote_ms for rhythm in iterator):
        raise ValueError("Provided rhythms should have same tempo (quarternote_ms).")

    iois = [rhythm.iois for rhythm in iterator]
    iois = np.concatenate(iois)
    n_bars = np.sum([rhythm.n_bars for rhythm in iterator])
    played = [rhythm.played for rhythm in iterator]
    played = list(np.concatenate(played))
    played = [bool(x) for x in played]  # Otherwise we get Numpy booleans

    return Rhythm(iois,
                  n_bars=n_bars,
                  time_sig=iterator[0].time_sig,
                  quarternote_ms=iterator[0].quarternote_ms,
                  played=played)
