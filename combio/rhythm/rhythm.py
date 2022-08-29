from mingus.containers import Bar, Track
import numpy as np
import combio.rhythm.helpers
from combio.core import BaseSequence
import random


class Rhythm(BaseSequence):

    def __init__(self, iois, n_bars: int, time_signature, beat_ms):
        # Save attributes
        self.time_signature = time_signature  # Used for metrical sequences
        self.beat_ms = beat_ms  # Used for metrical sequences
        self.n_bars = n_bars

        if not np.sum(iois) % (n_bars * time_signature[0] * beat_ms) == 0:
            raise ValueError("The provided inter-onset intervals do not amount to whole bars.")

        # Call initializer of super class
        BaseSequence.__init__(self, iois, metrical=True)

    def __str__(self):
        return f"Object of type Rhythm.\nTime signature: {self.time_signature}\nNumber of bars: {self.n_bars}\nQuarternote (" \
               f"ms): {self.beat_ms}\nNumber of events: {len(self.onsets)}\nIOIs: {self.iois}\n" \
               f"Onsets:{self.onsets}\n"

    def __add__(self, other):
        return join_rhythms([self, other])

    def __len__(self):
        return len(self.onsets)

    @property
    def note_values(self):
        """
        Get note values from the IOIs, based on beat_ms.
        """

        ratios = self.iois / self.beat_ms / 4

        note_values = np.array([int(1 // ratio) for ratio in ratios])

        return note_values

    @classmethod
    def from_note_values(cls, note_values, time_signature=(4, 4), beat_ms=500):
        """
        Almost same as standard initialization, except that we don't provide the number of bars but calculate those.

        """

        ratios = np.array([1 / note * time_signature[1] for note in note_values])

        n_bars = np.sum(ratios) / time_signature[0]

        if n_bars % 1 != 0:
            raise ValueError("The provided note values do not amount to whole bars.")
        else:
            n_bars = int(n_bars)

        iois = ratios * beat_ms

        return cls(iois,
                   time_signature=time_signature,
                   beat_ms=beat_ms,
                   n_bars=n_bars)

    @classmethod
    def from_iois(cls, iois, time_signature, beat_ms):
        n_bars = np.sum(iois) / time_signature[0] / beat_ms

        if n_bars % 1 != 0:
            raise ValueError("The provided note values do not amount to whole bars.")
        else:
            n_bars = int(n_bars)

        return cls(iois, n_bars, time_signature, beat_ms)

    @classmethod
    def generate_random_rhythm(cls,
                               allowed_note_values,
                               n_bars=1,
                               time_signature=(4, 4),
                               beat_ms=500,
                               events_per_bar=None):
        """
        This function returns a randomly generated integer ratio Sequence on the basis of the provided params.
        """

        iois = np.empty(0)

        for bar in range(n_bars):
            all_rhythmic_ratios = combio.rhythm.helpers.all_rhythmic_ratios(allowed_note_values,
                                                                            time_signature,
                                                                            target_n=events_per_bar)
            ratios = random.choice(all_rhythmic_ratios)

            new_iois = ratios * 4 * beat_ms

            iois = np.concatenate((iois, new_iois), axis=None)

        return cls(iois, time_signature=time_signature, beat_ms=beat_ms, n_bars=n_bars)

    @classmethod
    def generate_isochronous(cls, n_bars, time_signature, beat_ms):

        n_iois = time_signature[0] * n_bars

        iois = n_iois * [beat_ms]

        return cls(iois=iois,
                   n_bars=n_bars,
                   time_signature=time_signature,
                   beat_ms=beat_ms)

    def plot_rhythm(self, filepath=None, print_staff=False):
        # create initial bar
        t = Track()
        b = Bar(meter=self.time_signature)

        # keep track of the index of the note_value
        note_i = 0
        # loop over the note values of the sequence

        for note_value in self.note_values:
            b.place_notes('G-4', note_value)

            # if bar is full, create new bar and add bar to track
            if b.current_beat == b.length:
                t.add_bar(b)
                b = Bar(meter=self.time_signature)

            note_i += 1

        # If final bar was not full yet, add a rest for the remaining duration
        if b.current_beat % 1 != 0:
            rest_value = 1 / b.space_left()
            if round(rest_value) != rest_value:
                raise ValueError("The rhythm could not be plotted. Most likely because the IOIs cannot "
                                 "be (easily) captured in musical notation. This for instance happens after "
                                 "using one of the tempo manipulation methods.")

            b.place_rest(rest_value)
            t.add_bar(b)

        lp = combio.rhythm.helpers.get_lp_from_track(t, print_staff=print_staff)

        combio.rhythm.helpers.plot_lp(lp, filepath)


def join_rhythms(iterator):
    """
    This function can join multiple Rhythm objects.
    """

    # Check whether iterable was passed
    if not hasattr(iterator, '__iter__'):
        raise ValueError("Please pass this function a list or other iterable object.")

    # Check whether all the objects are of the same type
    if not all(isinstance(rhythm, Rhythm) for rhythm in iterator):
        raise ValueError("You can only join multiple Rhythm objects.")

    if not all(rhythm.time_signature == iterator[0].time_signature for rhythm in iterator):
        raise ValueError("Provided rhythms should have the same time signatures.")

    if not all(rhythm.beat_ms == iterator[0].beat_ms for rhythm in iterator):
        raise ValueError("Provided rhythms should have same tempo (beat_ms).")

    iois = [rhythm.iois for rhythm in iterator]
    iois = np.concatenate(iois)
    n_bars = int(np.sum([rhythm.n_bars for rhythm in iterator]))

    return Rhythm(iois, n_bars=n_bars, time_signature=iterator[0].time_signature, beat_ms=iterator[0].beat_ms)
