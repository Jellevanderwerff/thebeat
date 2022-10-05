# Required imports
from __future__ import annotations

import os
from typing import Union, Optional
import numpy as np
import numpy.typing as npt

# Optional imports
from matplotlib import pyplot as plt

try:
    import abjad
except ImportError:
    abjad = None

# Local imports
import combio.core
from combio._decorators import requires_lilypond


class Rhythm(combio.core.sequence.BaseSequence):

    def __init__(self,
                 iois: Union[np.ndarray, list],
                 time_signature: tuple,
                 beat_ms: Union[int, float],
                 is_played: Optional[Union[npt.NDArray[bool], list[bool]]] = None,
                 name: Optional[str] = None):

        # Save attributes
        self.time_signature = time_signature  # Used for metrical sequences
        self.beat_ms = beat_ms  # Used for metrical sequences
        self.is_played = [True] * len(iois) if not is_played else list(is_played)

        # Calculate n_bars and check whether that makes whole bars
        n_bars = np.sum(iois) / time_signature[0] / beat_ms
        if not n_bars.is_integer():
            raise ValueError("The provided inter-onset intervals do not amount to whole bars.")
        self.n_bars = n_bars

        # Call initializer of super class
        combio.core.sequence.BaseSequence.__init__(self, iois=iois, metrical=True, name=name)

    def __str__(self):
        return (f"Object of type Rhythm.\n"
                f"Name: {self.name}"
                f"Time signature: {self.time_signature}\n"
                f"Number of bars: {self.n_bars}\n"
                f"Beat (ms): {self.beat_ms}\n"
                f"Number of events: {len(self.onsets)}\n"
                f"IOIs: {self.iois}\n"
                f"Onsets:{self.onsets}\n")

    def __add__(self, other):
        return combio._helpers.join_rhythms([self, other])

    def __len__(self):
        return len(self.onsets)

    @property
    def get_note_values(self):
        """
        Get note values from the IOIs, based on beat_ms.
        """

        # todo check this, I don't understand what the '4' means.

        ratios = self.iois / self.beat_ms / 4

        note_values = np.array([int(1 // ratio) for ratio in ratios])

        return note_values

    @classmethod
    def from_iois(cls,
                  iois: Union[npt.NDArray[Union[int, float]], list[Union[int, float]]],
                  time_signature: tuple[int],
                  beat_ms: int,
                  is_played: Optional[Union[list[bool], npt.NDArray[bool]]] = None,
                  name: Optional[str] = None) -> Rhythm:
        """
        Conveniance function; exactly the same as class constructor.

        Parameters
        ----------
        iois
        time_signature
        beat_ms
        is_played
        name

        Returns
        -------

        """
        n_bars = np.sum(iois) / time_signature[0] / beat_ms

        if not n_bars.is_integer():
            raise ValueError("The provided note values do not amount to whole bars.")

        return cls(iois=iois, time_signature=time_signature, beat_ms=beat_ms, is_played=is_played, name=name)

    @classmethod
    def from_note_values(cls,
                         note_values: Union[npt.NDArray[int], list[int]],
                         time_signature: tuple[int] = (4, 4),
                         beat_ms: int = 500,
                         is_played: Optional[Union[list[bool], npt.NDArray[bool]]] = None,
                         name: Optional[str] = None) -> Rhythm:

        ratios = np.array([1 / note * time_signature[1] for note in note_values])
        iois = ratios * beat_ms

        return cls(iois=iois, time_signature=time_signature, beat_ms=beat_ms, is_played=is_played, name=name)

    @classmethod
    def generate_random_rhythm(cls,
                               n_bars: int = 1,
                               time_signature: tuple[int] = (4, 4),
                               beat_ms: int = 500,
                               allowed_note_values: Optional[Union[list[int], npt.NDArray[int]]] = None,
                               n_rests: int = 0,
                               rng: Optional[np.random.Generator] = None,
                               name: Optional[str] = None) -> Rhythm:
        """
        This function returns a randomly generated integer ratio Sequence on the basis of the provided params.
        """

        if rng is None:
            rng = np.random.default_rng()

        if allowed_note_values is None:
            allowed_note_values = [4, 8, 16]

        iois = np.empty(0)

        all_ratios = combio._helpers.all_rhythmic_ratios(allowed_note_values, time_signature)

        for bar in range(n_bars):
            ratios = rng.choice(all_ratios, 1)[0]
            new_iois = ratios * 4 * beat_ms  # todo check what this 4 means
            iois = np.append(iois, new_iois)

        # Make rests
        if n_rests > len(iois):
            raise ValueError("The provided number of rests is higher than the number of onsets.")
        elif n_rests > 0:
            is_played = n_rests * [False] + (len(iois) - n_rests) * [True]
            rng.shuffle(is_played)
        else:
            is_played = None

        return cls(iois=iois, time_signature=time_signature, beat_ms=beat_ms, is_played=is_played, name=name)

    @classmethod
    def generate_isochronous(cls,
                             n_bars: int = 1,
                             time_signature: tuple[int] = (4, 4),
                             beat_ms: int = 500,
                             is_played: Optional[Union[list[bool], npt.NDArray[bool]]] = None,
                             name: Optional[str] = None) -> Rhythm:

        n_iois = time_signature[0] * n_bars

        iois = n_iois * [beat_ms]

        return cls(iois=iois, time_signature=time_signature, beat_ms=beat_ms, is_played=is_played, name=name)

    @requires_lilypond
    def plot_rhythm(self,
                    filepath: Union[os.PathLike, str] = None,
                    print_staff: bool = False,
                    suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:
        # Check whether abjad is installed
        if abjad is None:
            raise ImportError("This method requires the 'abjad' Python package."
                              "Install it, for instance by typing 'pip install abjad' into your terminal.")

        # Preliminaries
        time_signature = abjad.TimeSignature(self.time_signature)
        remove_footers = """\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""
                oddFooterMarkup = ""\nevenFooterMarkup = ""\n} """

        # Make the notes
        pitches = [abjad.NamedPitch('A3')] * len(self.onsets)
        durations = [abjad.Duration((1, int(note_value))) for note_value in self.get_note_values]
        note_maker = abjad.makers.NoteMaker()

        notes = []

        for pitch, duration, is_played in zip(pitches, durations, self.is_played):
            if is_played is True:
                note = note_maker(pitch, duration)
            else:
                note = abjad.Rest(duration)
            notes.append(note)

        # plot the notes
        staff = abjad.Staff(notes)
        abjad.attach(abjad.Clef('percussion'), staff[0])
        abjad.attach(time_signature, staff[0])

        # Make cleff transparent if necessary
        if print_staff is False:
            abjad.override(staff).clef.transparent = '##t'

        # Make the score and convert to lilypond object
        score = abjad.Score([staff])
        score_lp = abjad.lilypond(score)

        # Make lilypond string, adding the remove footers string (removes all unnecessary stuff, changes page size etc.)
        lpf = abjad.LilyPondFile([remove_footers, score_lp])
        lpf_str = abjad.lilypond(lpf)

        # Stop the staff if necessary (i.e. the horizontal lines behind the notes)
        if print_staff is False:
            lpf_str = lpf_str.replace(r'\clef "percussion"', r'\clef "percussion" \stopStaff')

        # Plot!
        fig, ax = combio._helpers.plot_lp(lp=lpf_str, filepath=filepath, suppress_display=suppress_display)

        return fig, ax
