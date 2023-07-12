# Copyright (C) 2023  Jelle van der Werff
#
# This file is part of thebeat.
#
# thebeat is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# thebeat is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with thebeat.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import copy
import re
import textwrap
from collections import namedtuple
import warnings

# Optional imports
try:
    import abjad
except ImportError:
    abjad = None

import sounddevice
import os
from fractions import Fraction
from typing import Union, Optional
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# Local imports
import thebeat.core
from thebeat._decorators import requires_lilypond
import thebeat._warnings
import thebeat.helpers
import thebeat.utils


class Rhythm(thebeat.core.sequence.BaseSequence):
    """
    The :py:class:`Rhythm` class can be used for working sequences that are rhythmical in the musical sense.
    This means that in addition to having inter-onset intervals (IOIs) that represent the timing of the events in the
    sequence, :py:class:`Rhythm` objects have musical properties such as a time signature, a beat duration,
    and they may contain rests.

    The :py:class:`Rhythm` class is also used as the basis for a :py:class:`~thebeat.music.Melody`.

    For more info on these properties, please refer to the class's :py:meth:`~thebeat.music.Rhythm.__init__`.
    """

    def __init__(self,
                 iois: Union[np.ndarray, list],
                 time_signature: tuple[int, int] = (4, 4),
                 beat_ms: float = 500,
                 is_played: Optional[np.typing.ArrayLike[bool]] = None,
                 name: Optional[str] = None):
        r"""
        Constructs a :py:class:`Rhythm` object.

        Parameters
        ----------
        iois
            An iterable of inter-onset intervals (IOIs). For instance: ``[500, 500, 400, 200]``.
        time_signature
            A musical time signature, for instance: ``(4, 4)``. As a reminder: the upper number indicates
            *how many beats* there are in a bar. The lower number indicates the denominator of the value that
            indicates *one beat*. So, in ``(4, 8)`` time, a bar would be filled if we have four
            :math:`\frac{1}{8}` th notes.
        beat_ms
            The value (in milliseconds) for the beat, i.e. the duration of a :math:`\frac{1}{4}` th note if the lower
            number in the time signature is 4.
        is_played
            A list or array containing booleans indicating whether a note should be played or not.
            Defaults to ``[True, True, True, ...]``.
        name
            Optionally, you can give the Sequence object a name. This is used when printing, plotting, or writing
            the Sequence object. It can always be retrieved and changed via :py:attr:`Rhythm.name`.

        Examples
        --------
        >>> iois = [500, 250, 250, 500, 500]
        >>> r = Rhythm(iois)
        >>> print(r.onsets)
        [   0.  500.  750. 1000. 1500.]

        >>> iois = [500, 250, 250, 500]
        >>> r = Rhythm(iois=iois, time_signature=(3, 4), beat_ms=250)
        >>> print(r.note_values)
        [2 4 4 2]

        """

        # Save attributes
        self.time_signature = time_signature
        self.beat_ms = beat_ms
        self.is_played = [True] * len(iois) if not is_played else list(is_played)

        # Calculate n_bars and check whether that makes whole bars
        n_bars = np.sum(iois) / time_signature[0] / beat_ms
        if not n_bars.is_integer():
            raise ValueError("The provided inter-onset intervals do not amount to whole bars.")
        self.n_bars = n_bars

        # Call initializer of super class
        super().__init__(iois=iois, end_with_interval=True, name=name)

    def __str__(self):
        return (f"Object of type Rhythm.\n"
                f"Time signature: {self.time_signature}\n"
                f"Number of bars: {self.n_bars}\n"
                f"Beat (ms): {self.beat_ms}\n"
                f"Number of events: {len(self.onsets)}\n"
                f"IOIs: {self.iois}\n"
                f"Onsets:{self.onsets}\n"
                f"Name: {self.name}\n")

    def __repr__(self):
        if self.name:
            return f"Rhythm(name={self.name}, n_bars={self.n_bars}, time_signature={self.time_signature})"

        return f"Rhythm(n_bars={self.n_bars}, time_signature={self.time_signature})"

    def __add__(self, other):
        return thebeat.helpers.join_rhythms([self, other])

    def __mul__(self, other):
        return self._repeat(times=other)

    @property
    def fractions(self) -> np.ndarray:
        r"""Calculate how to describe the rhythm in fractions from the total duration of the sequence.

        Example
        -------

        A sequence of IOIs ``[250, 500, 1000, 250]`` has a total duration of 2000 ms.
        This can be described using the fractions :math:`\frac{1}{8}, \frac{2}{8}, \frac{4}{8}, \frac{1}{8}`,
        so this method returns the fractions ``[1/8, 2/8, 4/8, 1/8]``.

        Examples
        --------
        >>> r = Rhythm([250, 500, 1000, 250])
        >>> print(r.fractions)
        [Fraction(1, 8) Fraction(1, 4) Fraction(1, 2) Fraction(1, 8)]

        """

        floats = self.iois / np.sum(self.iois)

        fractions = [Fraction.from_float(f) for f in floats]
        return np.array(fractions)

    @property
    def integer_ratios(self) -> np.ndarray:
        r"""Calculate how to describe the rhythm in integer ratio numerators from
        the total duration of the sequence by finding the least common multiplier.

        Example
        -------
        A sequence of IOIs ``[250, 500, 1000, 250]`` has a total duration of 2000 ms.
        This can be described using the least common multiplier as
        :math:`\frac{1}{8}, \frac{2}{8}, \frac{4}{8}, \frac{1}{8}`,
        so this method returns the numerators ``[1, 2, 4, 1]``.

        Notes
        -----
        The method for calculating the integer ratios is based on :cite:t:`jacobyIntegerRatioPriors2017`.

        Caution
        -------
        This function uses rounding to find the nearest integer ratio.

        Examples
        --------
        >>> r = Rhythm([250, 500, 1000, 250])
        >>> print(r.integer_ratios)
        [1 2 4 1]

        """

        fractions = [Fraction(int(ioi), int(self.duration)) for ioi in self.iois]
        lcm = np.lcm.reduce([fr.denominator for fr in fractions])

        vals = [int(fr.numerator * lcm / fr.denominator) for fr in fractions]

        return np.array(vals)

    @property
    def note_values(self):
        """
        This property returns the denominators of the note values in this sequence, calculated from the
        inter-onset intervals (IOIs). A note value of ``2`` means a half note. A note value of ``4`` means a
        quarternote, etc. One triplet of three notes would be ``[12, 12, 12]``.

        Caution
        -------
        Please note that this function is basic (e.g. there is no support for dotted notes etc.). That's beyond
        the scope of this package.

        Examples
        --------
        >>> r = Rhythm([500, 1000, 250, 250], time_signature=(4, 4), beat_ms=500)  # doctest: +SKIP
        >>> print(r.note_values)  # doctest: +SKIP
        [4 2 8 8]

        >>> r = Rhythm([166.66666667, 166.66666667, 166.66666667, 500, 500, 500], beat_ms=500]  # doctest: +SKIP
        >>> print(r.note_values)  # doctest: +SKIP
        [12 12 12  4  4  4]

        """
        # todo possibly just return tuples (so we can e.g. have a duration of 3 quarter notes)
        #  Simply use the abjad note durations function!

        ratios = self.iois / self.beat_ms / 4

        note_values = np.array([int(1 // ratio) for ratio in ratios])

        return note_values

    @classmethod
    def from_fractions(cls,
                       fractions: Union[list, np.ndarray],
                       time_signature: tuple[int, int] = (4, 4),
                       beat_ms: int = 500,
                       is_played: np.typing.ArrayLike[bool] = None,
                       name: Optional[str] = None) -> Rhythm:
        r"""

        This class method can be used for creating a Rhythm on the basis of fractions. The fractions
        can be input either as floats (e.g. 0.25) or as :class:`fractions.Fraction` objects.

        Parameters
        ----------
        fractions
            Contains the fractions of the rhythm. For instance: ``[1, 2, 4]``.
        time_signature
            The time signature of the rhythm. For instance: ``(4, 4)``.
        beat_ms
            The duration of a beat in milliseconds. This refers to the duration of the denominator of the time
            signature.
        is_played
            A list of booleans indicating which notes are played. If None, all notes are played.
        name
            A name for the rhythm.

        Example
        -------
        The fractions represent for each note the proportion of the bar that the note takes up,
        taking into account the time signature. A few examples:

        In 4/4 time, a quarter note would be 1/4, an eighth note 1/8, etc.
        In 4/8 time, a quarter note would be 1/2, an eighth note 1/4, etc.

        In 5/4 time, we would input a quarter note as 1/5, an eighth note as 1/10, etc.
        In 5/8 time, we would input a dotted quarter note as 3/5, an eighth  note as 1/5, etc.

        Examples
        --------
        >>> r = Rhythm.from_fractions([1/4, 1/4, 1/4, 1/4], time_signature=(4, 4), beat_ms=500)

        >>> import fractions
        >>> dotted_halfnote = fractions.Fraction(3, 5)
        >>> halfnote = fractions.Fraction(2, 5)
        >>> r = Rhythm.from_fractions([dotted_halfnote, halfnote], time_signature=(5, 4), beat_ms=500)

        >>> r = Rhythm.from_fractions([1/4, 1/4, 1/4, 1/4], time_signature=(4, 8), beat_ms=500)

        """

        fractions = np.array(fractions)

        iois_as_fractions = fractions * beat_ms * time_signature[0]

        iois = np.array([float(frac) for frac in iois_as_fractions])

        return cls(iois=iois, time_signature=time_signature, beat_ms=beat_ms, is_played=is_played, name=name)

    @classmethod
    def from_integer_ratios(cls,
                            numerators: npt.ArrayLike[float],
                            time_signature: tuple[int, int] = (4, 4),
                            beat_ms: int = 500,
                            is_played: np.typing.ArrayLike[bool] = None,
                            name: Optional[str] = None) -> Rhythm:
        r"""

        Very simple conveniance class method that constructs a Rhythm object by calculating the inter-onset intervals
        (IOIs) as ``numerators * beat_ms``.

        Parameters
        ----------
        numerators
            Contains the numerators of the integer ratios. For instance: ``[1, 2, 4]``.
        time_signature
            A musical time signature, for instance: ``(4, 4)``. As a reminder: the upper number indicates
            *how many beats* there are in a bar. The lower number indicates the denominator of the value that
            indicates *one beat*. So, in ``(4, 8)`` time, a bar would be filled if we have four
            :math:`\frac{1}{8}` th notes.
        beat_ms
            The value (in milliseconds) for the beat, i.e. the duration of a :math:`\frac{1}{4}` th note if the lower
            number in the time signature is 4.
        is_played
            A list or array containing booleans indicating whether a note should be played or not.
            Defaults to ``[True, True, True, ...]``.
        name
            Optionally, you can give the Sequence object a name. This is used when printing, plotting, or writing
            the Sequence object. It can always be retrieved and changed via :py:attr:`Rhythm.name`.

        """

        numerators = np.array(numerators)

        return cls(iois=numerators * beat_ms,
                   beat_ms=beat_ms,
                   time_signature=time_signature,
                   is_played=is_played,
                   name=name)

    @classmethod
    def from_note_values(cls,
                         note_values: np.typing.ArrayLike[int],
                         time_signature: tuple[int, int] = (4, 4),
                         beat_ms: int = 500,
                         is_played: Optional[np.typing.ArrayLike[bool]] = None,
                         name: Optional[str] = None) -> Rhythm:
        r"""Create a Rhythm object on the basis of note values (i.e. note durations).

        Note values are input as the denominators of the fraction. A note value of ``2`` means a half note,
        a note value of ``4`` means a quarternote etc. A triplet would be ``[12, 12, 12]``.

        Parameters
        ----------
        note_values
            A list or array containing the denominators of the note values. A note value of ``2`` means a half note,
            a note value of ``4`` means a quarternote etc. A triplet would be ``[12, 12, 12]``.
        time_signature
            A musical time signature, for instance: ``(4, 4)``. As a reminder: the upper number indicates
            *how many beats* there are in a bar. The lower number indicates the denominator of the value that
            indicates *one beat*. So, in ``(4, 8)`` time, a bar would be filled if we have four
            :math:`\frac{1}{8}` th notes.
        beat_ms
            The value (in milliseconds) for the beat, i.e. the duration of a :math:`\frac{1}{4}` th note if the lower
            number in the time signature is 4.
        is_played
            A list or array containing booleans indicating whether a note should be played or not.
            Defaults to ``[True, True, True, ...]``.
        name
            Optionally, you can give the Sequence object a name. This is used when printing, plotting, or writing
            the Sequence object. It can always be retrieved and changed via :py:attr:`Rhythm.name`.

        Examples
        --------
        >>> r = Rhythm.from_note_values([16, 16, 16, 16, 4, 4, 4], beat_ms=500)
        >>> print(r.iois)
        [125. 125. 125. 125. 500. 500. 500.]
        """

        ratios = np.array([1 / note * time_signature[1] for note in note_values])
        iois = ratios * beat_ms

        return cls(iois=iois, time_signature=time_signature, beat_ms=beat_ms, is_played=is_played, name=name)

    @classmethod
    def generate_random_rhythm(cls,
                               n_bars: int = 1,
                               beat_ms: int = 500,
                               time_signature: tuple[int, int] = (4, 4),
                               allowed_note_values: Optional[np.typing.ArrayLike[int]] = None,
                               n_rests: int = 0,
                               rng: Optional[np.random.Generator] = None,
                               name: Optional[str] = None) -> Rhythm:
        r"""
        This function generates a random rhythmic sequence on the basis of the provided parameters.

        It does so by first generating all possible combinations of note values that amount to one bar based on the
        ``allowed_note_values`` parameter, and then selecting (with replacement) ``n_bars`` combinations out of
        the possibilities.

        Parameters
        ----------
        n_bars
            The desired number of musical bars.
        beat_ms
            The value (in milliseconds) for the beat, i.e. the duration of a :math:`\frac{1}{4}` th note if the lower
            number in the time signature is 4.
        time_signature
            A musical time signature, for instance: ``(4, 4)``. As a reminder: the upper number indicates
            *how many beats* there are in a bar. The lower number indicates the denominator of the value that
            indicates *one beat*. So, in ``(4, 8)`` time, a bar would be filled if we have four
            :math:`\frac{1}{8}` th notes.
        allowed_note_values
            A list or array containing the denominators of the allowed note values. A note value of ``2`` means a half
            note, a note value of ``4`` means a quarternote etc. Defaults to ``[4, 8, 16]``.
        n_rests
            If desired, one can provide a number of rests to be inserted at random locations. These are placed after
            the random selection of note values.
        rng
            A :class:`numpy.random.Generator` object. If not supplied :func:`numpy.random.default_rng` is
            used.
        name
            If desired, one can give a rhythm a name. This is for instance used when printing the rhythm,
            or when plotting the rhythm. It can always be retrieved and changed via :py:attr:`Rhythm.name`.

        Examples
        --------
        >>> import numpy as np  # not required, here for reproducability
        >>> generator = np.random.default_rng(seed=321)  # not required, for reproducability
        >>> r = Rhythm.generate_random_rhythm(rng=generator)
        >>> print(r.iois)
        [125. 250. 125. 125. 500. 125. 125. 125. 500.]

        >>> import numpy as np  # not required, here for reproducability
        >>> generator = np.random.default_rng(seed=321)  # not required, here for reproducability
        >>> r = Rhythm.generate_random_rhythm(beat_ms=1000,allowed_note_values=[2, 4, 8],rng=generator)
        >>> print(r.iois)
        [ 500. 1000.  500.  500. 1000.  500.]
        """

        if rng is None:
            rng = np.random.default_rng()

        if allowed_note_values is None:
            allowed_note_values = [4, 8, 16]

        iois = np.empty(0)

        all_ratios = thebeat.helpers.all_rhythmic_ratios(allowed_note_values, time_signature)

        for bar in range(n_bars):
            ratios = rng.choice(all_ratios, 1)[0]
            new_iois = ratios * 4 * beat_ms
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
                             time_signature: tuple[int, int] = (4, 4),
                             beat_ms: int = 500,
                             is_played: Optional[np.typing.ArrayLike[bool]] = None,
                             name: Optional[str] = None) -> Rhythm:
        r"""

        Simply generates an isochronous (i.e. with equidistant inter-onset intervals) rhythm. Will have
        the bars filled with intervals of ``beat_ms``.

        Parameters
        ----------
        n_bars
            The desired number of musical bars.
        time_signature
            A musical time signature, for instance: ``(4, 4)``. As a reminder: the upper number indicates
            *how many beats* there are in a bar. The lower number indicates the denominator of the value that
            indicates *one beat*. So, in ``(4, 8)`` time, a bar would be filled if we have four
            :math:`\frac{1}{8}` th notes.
        beat_ms
            The value (in milliseconds) for the beat, i.e. the duration of a :math:`\frac{1}{4}` th note if the lower
            number in the time signature is 4.
        is_played
            A list or array containing booleans indicating whether a note should be played or not.
            Defaults to ``[True, True, True, ...]``.
        name
            If desired, one can give a rhythm a name. This is for instance used when printing the rhythm,
            or when plotting the rhythm. It can always be retrieved and changed via :py:attr:`Rhythm.name`.

        """

        n_iois = time_signature[0] * n_bars

        iois = n_iois * [beat_ms]

        return cls(iois=iois, time_signature=time_signature, beat_ms=beat_ms, is_played=is_played, name=name)

    @requires_lilypond
    def plot_rhythm(self,
                    filepath: Union[os.PathLike, str] = None,
                    staff_type: str = "rhythm",
                    print_staff: bool = True,
                    title: Optional[str] = None,
                    suppress_display: bool = False,
                    dpi: int = 600,
                    ax: Optional[plt.Axes] = None) -> tuple[plt.Figure, plt.Axes]:
        """
        Make a plot containing the musical notation of the rhythm. This function requires you to install:

        * `abjad <https://abjad.github.io/>`_ (install via ``pip install abjad``).
        * `lilypond <https://lilypond.org/download.en.html>`_

        For lilypond on Windows, make sure to follow the website's instructions on how to make lilypond available to
        be run through the command line. Linux and Mac OS users are advised to either ``apt install lilypond`` or
        ``brew install lilypond``. On Mac that requires you to install `Homebrew <https://brew.sh/>`_.

        The plot is returned as a :class:`matplotlib.figure.Figure` and :class:`matplotlib.axes.Axes` object,
        which you can manipulate.

        .. figure:: images/plot_rhythm_rhythmstaff.png
            :scale: 100 %

            A plot with the default ``print_staff=True`` and the default ``staff_type="rhythm"``.



        .. figure:: images/plot_rhythm_withstaff.png
            :scale: 50 %

            A plot with the default ``print_staff=True`` and ``staff_type="percussion"``.


        .. figure:: images/plot_rhythm_nostaff.png
            :scale: 50 %

            A plot with ``print_staff=False``.


        Parameters
        ----------
        filepath
            Optionally, you can save the plot to a file. Supported file formats are only '.png' and '.pdf'.
            The desired file format will be selected based on what the filepath ends with.
        staff_type
            Either 'percussion' or 'rhythm'. 'Rhythm' is a single line (like a woodblock score). Percussion
            is drum notation.
        print_staff
            If desired, you can choose to print a musical staff (the default is not to do this). The staff will be a
            `percussion staff <https://en.wikipedia.org/wiki/Percussion_notation>`_.
        title
            A title for the plot. Note that this is not considered when saving the plot as an .eps file.
        suppress_display
            If desired,you can choose to suppress displaying the plot in your IDE. This means that
            :func:`matplotlib.pyplot.show` is not called. This is useful when you just want to save the plot or
            use the returned :class:`matplotlib.figure.Figure` and :class:`matplotlib.axes.Axes` objects.
        dpi
            The resolution of the plot in dots per inch.
        ax
            Optionally, you can provide an existing :class:`matplotlib.axes.Axes` object to plot the rhythm on.


        Examples
        --------
        >>> r = Rhythm([500, 250, 1000, 250], beat_ms=500)
        >>> r.plot_rhythm()  # doctest: +SKIP

        >>> r = Rhythm([250, 250, 500, 500, 1500], time_signature=(3, 4))
        >>> fig, ax = r.plot_rhythm(print_staff=True, suppress_display=True)  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP

        >>> r = Rhythm.from_note_values([4, 4, 4, 4])
        >>> r.plot_rhythm(filepath='isochronous_rhythm.pdf')  # doctest: +SKIP


        """
        # Check whether abjad is installed
        if abjad is None:
            raise ImportError("This method requires the 'abjad' Python package."
                              "Install it, for instance by typing 'pip install abjad' into your terminal.")

        # Abjad 3.12 and lower use the NoteMaker class, which is deprecated in 3.13. This is a workaround for compatability with all versions.
        # This because abjad 3.13 etc require Python 3.10.
        try:
            make_notes = abjad.makers.make_notes
        except AttributeError:
            note_maker = abjad.makers.NoteMaker()
            def make_notes(*args):
                return list(note_maker(*args))

        # Preliminaries
        time_signature = abjad.TimeSignature(self.time_signature)
        remove_footers = """\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""
                oddFooterMarkup = ""\nevenFooterMarkup = ""\n} """

        # Make the notes
        durations = self._get_abjad_note_durations()
        durations, ties_at = self._get_abjad_ties(durations)
        pitches = [abjad.NamedPitch('A3')] * len(durations)

        notes = []

        # Here we insert another of the same type of is_played
        # at the place where it was split
        count = 0
        is_played = self.is_played

        # if we split a note at the end of a bar and tie it to the first note in the subsequent bar,
        # we now suddenly have two notes instead of one. so we need to add another of the same
        # boolean value to is_played for that note.
        for tie_at in ties_at:
            is_played.insert(tie_at + count, is_played[tie_at])

        # loop over the pitch duration and whether it is a note or rest, and add to notes
        for pitch, duration, is_plyd in zip(pitches, durations, is_played):
            note = make_notes(pitch, duration)[0] if is_plyd else abjad.Rest(duration)
            notes.append(note)

        # plot the notes
        staff = abjad.Staff(notes)
        # add ties at the places where _get_abjad_ties thinks they should be (most of the time this is skipped)
        for tie_at in ties_at:
            # but only for notes, not for rests
            if is_played[tie_at]:
                tie = abjad.Tie()
                abjad.attach(tie, staff[tie_at])

        # Change clef and staff type
        if staff_type == "percussion":
            abjad.attach(abjad.Clef('percussion'), staff[0])
        elif staff_type == "rhythm":
            staff.lilypond_type = "RhythmicStaff"
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
            lpf_str = lpf_str.replace(r'\time', r'\stopStaff \time')

        # Plot!
        fig, ax = thebeat.helpers.plot_lp(lp=lpf_str, filepath=filepath, suppress_display=suppress_display, title=title,
                                          dpi=dpi, ax=ax)

        return fig, ax

    def copy(self, deep: bool = True):
        """Returns a copy of itself. See :py:func:`copy.copy` for more information.

        Parameters
        ----------
        deep
            If ``True``, a deep copy is returned. If ``False``, a shallow copy is returned.

        """
        if deep is True:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    def _repeat(self, times: int) -> Rhythm:
        """
        Repeat the Rhythm ``times`` times. Returns a new Rhythm object. The old one remains unchanged.

        Parameters
        ----------
        times
            How many times the Rhythm should be repeated.

        """

        if not isinstance(times, int):
            raise TypeError("You can only multiply Sequence objects by integers.")

        new_iois = np.tile(self.iois, reps=times)
        is_played = self.is_played * times

        return Rhythm(new_iois, beat_ms=self.beat_ms, time_signature=self.time_signature, is_played=is_played,
                      name=self.name)

    def to_sequence(self) -> thebeat.core.Sequence:
        """
        Convert the rhythm to a :class:`~thebeat.core.Sequence` object.
        """
        return thebeat.core.Sequence(iois=self.iois, first_onset=0.0, end_with_interval=True, name=self.name)

    def _get_abjad_note_durations(self):
        """Get abjad note durations from the integer_ratios
        #todo This needs to be done with lcm to avoid rounding problems,
          though seems to work for now.
        """
        total_duration = np.sum(self.integer_ratios)
        duration_of_bar = total_duration / self.n_bars
        ratios = np.array([ratio / duration_of_bar for ratio in self.integer_ratios])
        numerators = ratios * self.time_signature[0]

        durations = [abjad.Duration(Fraction(numerator) / self.time_signature[1]) for numerator in numerators]

        return durations

    def _get_abjad_ties(self, durations):
        full_bar = self.time_signature[0] / self.time_signature[1]
        # will be output
        notes = []
        ties_at = []

        # Keep track of how full the current bar is
        bar_fullness = 0

        for i, note in enumerate(durations):
            # if the note fits in the bar
            if (note + bar_fullness) <= full_bar:
                bar_fullness += note
                notes.append(note)
            # if note doesn't fit the bar
            else:
                # try to divide the note up into smaller bits
                for division in (2, 4, 8):
                    # if now it fits in the bar
                    if (note / division) + bar_fullness <= 1:
                        # we split up the original note into a small bit, and the rest (e.g. 1/4 and 3/4)
                        split_notes = [note / division, note - (note / division)]
                        # We need to remember which notes to tie later
                        notes += split_notes
                        ties_at.append(i)
                        bar_fullness += sum(split_notes)
                        bar_fullness -= full_bar
                        break

            # if bar is full set bar_fullness to zero
            if bar_fullness % full_bar == 0:
                bar_fullness = 0

        # If at the end of all this the bars are not full yet, raise an error
        if not bar_fullness % full_bar == 0:
            raise ValueError("There was an error while trying to tie the final note of a bar to the first note"
                             "of the subsequent bar. Try a different rhythm.")

        return notes, ties_at


class Melody(thebeat.core.sequence.BaseSequence):
    """
    A :py:class:`Melody` object contains a both a **rhythm** and **pitch information**.
    It does not contain sound. However, the :py:class:`Melody` can be synthesized and played or written to
    disk, for instance using the :py:meth:`~Melody.synthesize_and_play()` method.

    See the :py:meth:`~Melody.__init__` to learn how a :py:class:`Melody` object is constructed, or use one
    of the different class methods, such as the
    :py:meth:`~Melody.generate_random_melody` method.

    Most of the functions require you to install `abjad <https://abjad.github.io/>`_. Please note that the
    current version of `abjad` requires Python 3.10. The last version that supported Python 3.6-3.9 is
    `this one <https://pypi.org/project/abjad/3.4/>`_.

    """

    def __init__(self,
                 rhythm: thebeat.music.Rhythm,
                 pitch_names: Union[npt.NDArray[str], list[str], str],
                 octave: Optional[int] = None,
                 key: Optional[str] = None,
                 is_played: Optional[list] = None,
                 name: Optional[str] = None):
        """

        Parameters
        ----------
        rhythm
            A :py:class:`~thebeat.music.Rhythm` object.
        pitch_names
            An array or list containing note names. They can be in a variety of formats, such as
            ``"G4"`` for a G note in the fourth octave, or ``"g'"``, or simply ``G``. The names are
            processed by :class:`abjad.pitch.NamedPitch`. Follow the link to find examples of the different
            formats. Alternatively it can be a string, but only in the formats: ``'CCGGC'`` or ``'C4C4G4G4C4'``.
        key
            Optionally, you can provide a key. This is for instance used when plotting a :py:class:`Melody` object.
        is_played
            Optionally, you can indicate if you want rests in the :py:class:`Melody`. Provide an array or list of
            booleans, for instance: ``[True, True, False, True]`` would mean a rest in place of the third event.
            The default is True for each event.
        name
            Optionally, the :py:class:`Melody` object can have a name. This is saved to the :py:attr:`Melody.name`
            attribute.

        Examples
        --------
        >>> r = thebeat.music.Rhythm.from_note_values([4, 4, 4, 4, 4, 4, 2])
        >>> mel = Melody(r, 'CCGGAAG')

        """

        # Initialize namedtuple. The namedtuple template is saved as an attribute.
        self.Event = namedtuple('event', 'onset_ms duration_ms note_value pitch_name is_played')

        # Make is_played if None supplied
        if is_played is None:
            is_played = [True] * len(rhythm.onsets)

        # Process pitch names
        if isinstance(pitch_names, str):
            pitch_names_list = re.split(r"([A-Z])([0-9]?)", pitch_names)
            pitch_names_list = list(filter(None, pitch_names_list))
            search = re.search(r"[0-9]", pitch_names)
            if search is None:
                if octave is None:
                    pitch_names_list = [pitch + str(4) for pitch in pitch_names_list]
                elif octave is not None:
                    pitch_names_list = [pitch + str(octave) for pitch in pitch_names_list]
        else:
            pitch_names_list = pitch_names

        self.pitch_names = pitch_names_list

        # Add initial events
        self.events = self._make_namedtuples(rhythm=rhythm,
                                             iois=rhythm.iois,
                                             note_values=rhythm.note_values,
                                             pitch_names=self.pitch_names,
                                             is_played=is_played)

        # Save rhythmic/musical attributes
        self.time_signature = rhythm.time_signature
        self.beat_ms = rhythm.beat_ms
        self.key = key

        # Check whether the provided IOIs result in a sequence only containing whole bars
        n_bars = np.sum(rhythm.iois) / self.time_signature[0] / self.beat_ms
        if not n_bars.is_integer():
            raise ValueError("The provided inter-onset intervals do not amount to whole bars.")
        # Save number of bars as an attribute
        self.n_bars = n_bars

        # Call BaseSequence constructor
        super().__init__(iois=rhythm.iois, end_with_interval=True, name=name)

    # todo add __str__ __add__ _mul__ etc.

    def __repr__(self):
        if self.name:
            return f"Melody(name={self.name}, n_bars={self.n_bars}, key={self.key})"

        return f"Melody(n_bars={self.n_bars}, key={self.key})"

    @classmethod
    def generate_random_melody(cls,
                               n_bars: int = 1,
                               beat_ms: int = 500,
                               time_signature: tuple = (4, 4),
                               key: str = 'C',
                               octave: int = 4,
                               n_rests: int = 0,
                               allowed_note_values: list = None,
                               rng: np.random.Generator = None,
                               name: Optional[str] = None) -> Melody:
        r"""

        Generate a random rhythm as well as a melody, based on the given parameters. Internally, for the
        rhythm, the :py:meth:`Rhythm.generate_random_rhythm` method is used. The melody is a random selection
        of pitch values based on the provided key and octave.

        Parameters
        ----------
        n_bars
            The desired number of musical bars.
        beat_ms
            The value (in milliseconds) for the beat, i.e. the duration of a :math:`\frac{1}{4}` th note if the lower
            number in the time signature is 4.
        time_signature
            A musical time signature, for instance: ``(4, 4)``. As a reminder: the upper number indicates
            *how many beats* there are in a bar. The lower number indicates the denominator of the value that
            indicates *one beat*. So, in ``(4, 8)`` time, a bar would be filled if we have four
            :math:`\frac{1}{8}` th notes.
        key
            The musical key used for randomly selecting the notes. Only major keys are supported for now.
        octave
            The musical octave. The default is concert pitch, i.e. ``4``.
        n_rests
            If desired, one can provide a number of rests to be inserted at random locations. These are placed after
            the random selection of note values.
        allowed_note_values
            A list or array containing the denominators of the allowed note values. A note value of ``2`` means a half
            note, a note value of ``4`` means a quarternote etc. Defaults to ``[4, 8, 16]``.
        rng
            A :class:`numpy.random.Generator` object. If not supplied :func:`numpy.random.default_rng` is
            used.
        name
            If desired, one can give the melody a name. This is for instance used when printing the rhythm,
            or when plotting the rhythm. It can always be retrieved and changed via :py:attr:`Rhythm.name`.

        Examples
        --------
        >>> generator = np.random.default_rng(seed=123)
        >>> m = Melody.generate_random_melody(rng=generator)
        >>> print(m.note_values)
        [16 16 16 16 16 16 16  8  8 16  8 16 16]
        >>> print(m.pitch_names)
        ["a'", "g'", "c'", "c''", "d'", "e'", "d'", "e'", "d'", "e'", "b'", "f'", "c''"]


        """
        if abjad is None:
            raise ImportError("This method requires the 'abjad' Python package."
                              "Install it, for instance by typing 'pip install abjad' into your terminal.")

        if rng is None:
            rng = np.random.default_rng()

        if allowed_note_values is None:
            allowed_note_values = [4, 8, 16]

        # Generate random rhythm and random tone_heights
        rhythm = thebeat.music.Rhythm.generate_random_rhythm(n_bars=n_bars, beat_ms=beat_ms,
                                                             time_signature=time_signature,
                                                             allowed_note_values=allowed_note_values, rng=rng)
        pitch_names_possible = [pitch.name for pitch in thebeat.utils.get_major_scale(tonic=key, octave=octave)]

        pitch_names_chosen = list(rng.choice(pitch_names_possible, size=len(rhythm.onsets)))

        if n_rests > len(rhythm.onsets):
            raise ValueError("The provided number of rests is higher than the number of sounds.")

        # Make the rests and shuffle
        is_played = n_rests * [False] + (len(rhythm.onsets) - n_rests) * [True]
        rng.shuffle(is_played)

        return cls(rhythm=rhythm, pitch_names=pitch_names_chosen, is_played=is_played, name=name, key=key)

    @property
    def note_values(self):
        """
        This property returns the denominators of the note values in this sequence, calculated from the
        inter-onset intervals (IOIs). A note value of ``2`` means a half note. A note value of ``4`` means a
        quarternote, etc. One triplet of three notes would be ``[12, 12, 12]``.

        Caution
        -------
        Please note that this function is basic (e.g. there is no support for dotted notes etc.). That's beyond
        the scope of this package.

        Examples
        --------
        >>> r = thebeat.music.Rhythm([500, 1000, 250, 250], time_signature=(4, 4), beat_ms=500)
        >>> m = Melody(r, pitch_names='CCGC')
        >>> print(r.note_values)  # doctest: +SKIP
        [4 2 8 8]

        >>> r = thebeat.music.Rhythm([166.66666667, 166.66666667, 166.66666667, 500, 500, 500], beat_ms=500]  # doctest: +SKIP
        >>> print(r.note_values)  # doctest: +SKIP
        [12 12 12  4  4  4]

        """

        ratios = self.iois / self.beat_ms / 4

        note_values = np.array([int(1 // ratio) for ratio in ratios])

        return note_values

    def copy(self, deep: bool = True):
        """Returns a copy of itself. See :py:func:`copy.copy` for more information.

        Parameters
        ----------
        deep
            If ``True``, a deep copy is returned. If ``False``, a shallow copy is returned.

        """
        if deep is True:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    @requires_lilypond
    def plot_melody(self,
                    filepath: Optional[Union[os.PathLike, str]] = None,
                    key: Optional[str] = None,
                    suppress_display: bool = False,
                    dpi: int = 600) -> tuple[plt.Figure, plt.Axes]:
        """
        Use this function to plot the melody in musical notes. It requires lilypond to be installed. See
        :py:meth:`Rhythm.plot_rhythm` for installation instructions.



        .. figure:: images/plot_melody.png
            :scale: 50 %

            An example of a melody plotted with this method.


        Parameters
        ----------
        filepath
            Optionally, you can save the plot to a file. Supported file formats are only '.png' and '.eps'.
            The desired file format will be selected based on what the filepath ends with.
        key
            The musical key to plot in. Can differ from the key used to construct the :class:`Melody` object.
            Say you want to emphasize the accidentals (sharp or flat note), you can choose to plot the melody
            in 'C'. The default is to plot in the key that was used to construct the object.
        suppress_display
            If desired,you can choose to suppress displaying the plot in your IDE. This means that
            :func:`matplotlib.pyplot.show` is not called. This is useful when you just want to save the plot or
            use the returned :class:`matplotlib.figure.Figure` and :class:`matplotlib.axes.Axes` objects.
        dpi
            The resolution of the plot in dots per inch.



        Examples
        --------
        >>> r = thebeat.music.Rhythm(iois=[250, 500, 250, 500], time_signature=(3, 4))
        >>> m = Melody(r, 'CCGC')
        >>> m.plot_melody()  # doctest: +SKIP

        >>> m.plot_melody(filepath='mymelody.png', suppress_display=True)  # doctest: +SKIP

        >>> fig, ax = m.plot_melody(key='C', suppress_display=True)  # doctest: +SKIP

        """
        if abjad is None:
            raise ImportError("This method requires the installation of abjad. Please install, for instance "
                              "using 'pip install abjad'.")

        key = self.key if key is None else key

        lp = self._get_lp_from_events(time_signature=self.time_signature, key=key)

        fig, ax = thebeat.helpers.plot_lp(lp, filepath=filepath, suppress_display=suppress_display, dpi=dpi)

        return fig, ax

    def synthesize_and_return(self,
                              event_durations_ms: Optional[Union[list[int], npt.NDArray[int], int]] = None,
                              fs: int = 48000,
                              n_channels: int = 1,
                              amplitude: float = 1.0,
                              oscillator: str = 'sine',
                              onramp_ms: int = 0,
                              offramp_ms: int = 0,
                              ramp_type: str = 'linear',
                              metronome: bool = False,
                              metronome_amplitude: float = 1.0) -> tuple[np.ndarray, int]:
        """Since :py:class:`Melody` objects do not contain any sound information, you can use this method to
        synthesize the sound. It returnes a tuple containing the sound samples as a NumPy 1-D array,
        and the sampling frequency.

        Note
        ----
        Theoretically, four quarternotes played after each other constitute one long sound. This
        behaviour is the default here. However, in many cases it will probably be best to supply
        ``event_durations``, which means the events are played in the rhythm of the melody (i.e. according
        to the inter-onset intervals of the rhythm), but using a supplied duration.

        Parameters
        ----------
        event_durations_ms
            Can be supplied as a single integer, which means that duration will be used for all events
            in the melody, or as an array of list containing individual durations for each event. That of course
            requires an array or list with a size equal to the number of notes in the melody.
        fs
            The desired sampling frequency in hertz.
        n_channels
            The desired number of channels. Can be 1 (mono) or 2 (stereo).
        amplitude
            Factor with which sound is amplified. Values between 0 and 1 result in sounds that are less loud,
            values higher than 1 in louder sounds. Defaults to 1.0.
        oscillator
            The oscillator used for generating the sound. Either 'sine' (the default), 'square' or 'sawtooth'.
        onramp_ms
            The sound's 'attack' in milliseconds.
        offramp_ms
            The sound's 'decay' in milliseconds.
        ramp_type
            The type of on- and offramp_ms used. Either 'linear' (the default) or 'raised-cosine'.
        metronome
            If ``True``, a metronome sound is added to the samples. It uses :py:attr:`Melody.beat_ms` as the inter-onset
            interval.
        metronome_amplitude
            If desired, when synthesizing the object with a metronome sound you can adjust the
            metronome amplitude. A value between 0 and 1 means a less loud metronome, a value larger than 1 means
            a louder metronome sound.


        Examples
        --------
        >>> mel = Melody.generate_random_melody()
        >>> samples, fs = mel.synthesize_and_return()

        """

        samples = self._make_melody_sound(fs=fs, oscillator=oscillator, amplitude=amplitude, onramp_ms=onramp_ms,
                                          n_channels=n_channels, offramp_ms=offramp_ms, ramp_type=ramp_type,
                                          event_durations_ms=event_durations_ms)

        if metronome is True:
            samples = thebeat.helpers.get_sound_with_metronome(samples=samples, fs=fs, metronome_ioi=self.beat_ms,
                                                               metronome_amplitude=metronome_amplitude)

        return samples, fs

    def synthesize_and_play(self,
                            event_durations_ms: Optional[Union[list[int], npt.NDArray[int], int]] = None,
                            fs: int = 48000,
                            n_channels: int = 1,
                            amplitude: float = 1.0,
                            oscillator: str = 'sine',
                            onramp_ms: int = 0,
                            offramp_ms: int = 0,
                            ramp_type: str = 'linear',
                            metronome: bool = False,
                            metronome_amplitude: float = 1.0):
        """
        Since :py:class:`Melody` objects do not contain any sound information, you can use this method to
        first synthesize the sound, and subsequently have it played via the internally used :func:`sounddevice.play`.

        Note
        ----
        Theoretically, four quarternotes played after each other constitute one long sound. This
        behaviour is the default here. However, in many cases it will probably be best to supply
        ``event_durations``, which means the events are played in the rhythm of the melody (i.e. according
        to the inter-onset intervals of the rhythm), but using a supplied duration.

        Parameters
        ----------
        event_durations_ms
            Can be supplied as a single integer, which means that duration will be used for all events
            in the melody, or as an array of list containing individual durations for each event. That of course
            requires an array or list with a size equal to the number of notes in the melody.
        fs
            The desired sampling frequency in hertz.
        n_channels
            The desired number of channels. Can be 1 (mono) or 2 (stereo).
        amplitude
            Factor with which sound is amplified. Values between 0 and 1 result in sounds that are less loud,
            values higher than 1 in louder sounds. Defaults to 1.0.
        oscillator
            The oscillator used for generating the sound. Either 'sine' (the default), 'square' or 'sawtooth'.
        onramp_ms
            The sound's 'attack' in milliseconds.
        offramp_ms
            The sound's 'decay' in milliseconds.
        ramp_type
            The type of on- and offramp used. Either 'linear' (the default) or 'raised-cosine'.
        metronome
            If ``True``, a metronome sound is added for playback. It uses :py:attr:`Melody.beat_ms` as the inter-onset
            interval.
        metronome_amplitude
            If desired, when writing the object with a metronome sound you can adjust the
            metronome amplitude. A value between 0 and 1 means a less loud metronome, a value larger than 1 means
            a louder metronome sound.


        Examples
        --------
        >>> mel = Melody.generate_random_melody()
        >>> mel.synthesize_and_play()  # doctest: +SKIP

        >>> mel.synthesize_and_play(event_durations_ms=50)

        """

        samples, _ = self.synthesize_and_return(event_durations_ms=event_durations_ms, fs=fs, n_channels=n_channels,
                                                amplitude=amplitude, oscillator=oscillator, onramp_ms=onramp_ms,
                                                offramp_ms=offramp_ms, ramp_type=ramp_type, metronome=metronome,
                                                metronome_amplitude=metronome_amplitude)

        sounddevice.play(samples, samplerate=fs)
        sounddevice.wait()

    def synthesize_and_write(self,
                             filepath: Union[str, os.PathLike],
                             event_durations_ms: Optional[Union[list[int], npt.NDArray[int], int]] = None,
                             fs: int = 48000,
                             n_channels: int = 1,
                             amplitude: float = 1.0,
                             dtype: Union[str, np.dtype] = np.int16,
                             oscillator: str = 'sine',
                             onramp_ms: int = 0,
                             offramp_ms: int = 0,
                             ramp_type: str = 'linear',
                             metronome: bool = False,
                             metronome_amplitude: float = 1.0):
        """Since :py:class:`Melody` objects do not contain any sound information, you can use this method to
        first synthesize the sound, and subsequently write it to disk as a wave file.

        Note
        ----
        Theoretically, four quarternotes played after each other constitute one long sound. This
        behaviour is the default here. However, in many cases it will probably be best to supply
        ``event_durations``, which means the events are played in the rhythm of the melody (i.e. according
        to the inter-onset intervals of the rhythm), but using a supplied duration.

        Parameters
        ----------
        filepath
            The output destination for the .wav file. Either pass e.g. a ``Path`` object, or a string.
            Of course be aware of OS-specific filepath conventions.
        event_durations_ms
            Can be supplied as a single integer, which means that duration will be used for all events
            in the melody, or as an array of list containing individual durations for each event. That of course
            requires an array or list with a size equal to the number of notes in the melody.
        fs
            The desired sampling frequency in hertz.
        n_channels
            The desired number of channels. Can be 1 (mono) or 2 (stereo).
        amplitude
            Factor with which sound is amplified. Values between 0 and 1 result in sounds that are less loud,
            values higher than 1 in louder sounds. Defaults to 1.0.
        dtype
            The desired data type for the output file. Defaults to ``np.int16``.
            This means that the output file will be 16-bit PCM.
        oscillator
            The oscillator used for generating the sound. Either 'sine' (the default), 'square' or 'sawtooth'.
        onramp_ms
            The sound's 'attack' in milliseconds.
        offramp_ms
            The sound's 'decay' in milliseconds.
        ramp_type
            The type of on- and offramp used. Either 'linear' (the default) or 'raised-cosine'.
        metronome
            If ``True``, a metronome sound is added to the output file. It uses :py:attr:`Melody.beat_ms` as the inter-onset
            interval.
        metronome_amplitude
            If desired, when playing the object with a metronome sound you can adjust the
            metronome amplitude. A value between 0 and 1 means a less loud metronome, a value larger than 1 means
            a louder metronome sound.

        Examples
        --------
        >>> mel = Melody.generate_random_melody()
        >>> mel.synthesize_and_write(filepath='random_melody.wav')  # doctest: +SKIP

        """

        samples, _ = self.synthesize_and_return(event_durations_ms=event_durations_ms, fs=fs, n_channels=n_channels,
                                                amplitude=amplitude, oscillator=oscillator, onramp_ms=onramp_ms,
                                                offramp_ms=offramp_ms, ramp_type=ramp_type, metronome=metronome,
                                                metronome_amplitude=metronome_amplitude)

        thebeat.helpers.write_wav(samples=samples, fs=fs, filepath=filepath, dtype=dtype, metronome=metronome,
                                  metronome_ioi=self.beat_ms, metronome_amplitude=metronome_amplitude)

    def _make_namedtuples(self,
                          rhythm,
                          iois,
                          note_values,
                          pitch_names,
                          is_played) -> list:
        events = []

        for event in zip(rhythm.onsets, iois, note_values, pitch_names, is_played):
            entry = self.Event(event[0], event[1], event[2], event[3], event[4])
            events.append(entry)

        return events

    def _make_melody_sound(self,
                           fs: int,
                           n_channels: int,
                           oscillator: str,
                           amplitude: float,
                           onramp_ms: int,
                           offramp_ms: int,
                           ramp_type: str,
                           event_durations_ms: Optional[Union[list[int], npt.NDArray[int], int]] = None):

        # Calculate required number of frames
        total_duration_ms = np.sum(self.iois)
        n_frames = total_duration_ms / 1000 * fs

        # Avoid rounding issues
        if not n_frames.is_integer():
            warnings.warn(thebeat._warnings.framerounding)
        n_frames = int(np.ceil(n_frames))

        # Create empty array with length n_frames
        if n_channels == 1:
            samples = np.zeros(n_frames, dtype=np.float64)
        else:
            samples = np.zeros((n_frames, 2), dtype=np.float64)

        # Set event durations to the IOIs if no event durations were supplied (i.e. use full length notes)
        if event_durations_ms is None:
            event_durations = self.iois
        # If a single integer is passed, use that value for all the events
        elif isinstance(event_durations_ms, (int, float)):
            event_durations = np.tile(event_durations_ms, len(self.events))
        else:
            event_durations = event_durations_ms

        # Loop over the events, synthesize event sound, and add all of them to the samples array at the appropriate
        # times.
        for event, duration_ms in zip(self.events, event_durations):
            if event.is_played is True:
                event_samples = thebeat.helpers.synthesize_sound(duration_ms=duration_ms, fs=fs,
                                                                 freq=abjad.NamedPitch(event.pitch_name).hertz,
                                                                 n_channels=n_channels, oscillator=oscillator,
                                                                 amplitude=amplitude)
                if onramp_ms or offramp_ms:
                    event_samples = thebeat.helpers.make_ramps(samples=event_samples, fs=fs, onramp_ms=onramp_ms,
                                                               offramp_ms=offramp_ms, ramp_type=ramp_type)

                # Calculate start- and end locations for inserting the event into the output array
                # and warn if the location in terms of frames was rounded off.
                start_pos = event.onset_ms / 1000 * fs
                end_pos = start_pos + event_samples.shape[0]
                if not start_pos.is_integer() or not end_pos.is_integer():
                    warnings.warn(thebeat._warnings.framerounding)
                start_pos = int(np.ceil(start_pos))
                end_pos = int(np.ceil(end_pos))

                # Add event samples to output array
                samples[start_pos:end_pos] = samples[start_pos:end_pos] + event_samples

            else:
                pass

        if np.max(samples) > 1:
            warnings.warn(thebeat._warnings.normalization)
            samples = thebeat.helpers.normalize_audio(samples)

        return samples

    def _get_lp_from_events(self,
                            key: str,
                            time_signature: tuple):

        # Abjad 3.12 and lower use the NoteMaker class, which is deprecated in 3.13. This is a workaround for compatability with all versions.
        # This because abjad 3.13 etc require Python 3.10.
        try:
            make_notes = abjad.makers.make_notes
        except AttributeError:
            note_maker = abjad.makers.NoteMaker()
            def make_notes(*args):
                return list(note_maker(*args))
        time_signature = abjad.TimeSignature(time_signature)
        pitch = abjad.NamedPitchClass(key)
        key = abjad.KeySignature(pitch)
        preamble = textwrap.dedent(r"""
             \version "2.22.1"
             \language "english"
             \paper {
             indent = 0\mm
             line-width = 110\mm
             oddHeaderMarkup = ""
             evenHeaderMarkup = ""
             oddFooterMarkup = ""
             evenFooterMarkup = ""
             }
             """)

        pitch_names = [event.pitch_name for event in self.events]
        is_played = [event.is_played for event in self.events]

        # Get note durations
        fractions = [Fraction(int(ioi), int(self.duration)) for ioi in self.iois]
        lcm = np.lcm.reduce([fr.denominator for fr in fractions])
        integer_ratios = [int(fr.numerator * lcm / fr.denominator) for fr in fractions]
        total_duration = np.sum(integer_ratios)
        duration_of_bar = total_duration / self.n_bars
        ratios = np.array([ratio / duration_of_bar for ratio in integer_ratios])
        numerators = ratios * self.time_signature[0]

        note_durations = [abjad.Duration(Fraction(numerator) / self.time_signature[1]) for numerator in numerators]

        notes = []

        for pitch_name, note_duration, is_played in zip(pitch_names, note_durations, is_played):
            if is_played is True:
                pitch = abjad.NamedPitch(pitch_name)
                note = make_notes(pitch, note_duration)[0]
            else:
                note = abjad.Rest(note_duration)
            notes.append(note)

        voice = abjad.Voice(notes)
        abjad.attach(time_signature, voice[0])
        abjad.attach(key, voice[0])

        staff = abjad.Staff([voice])
        score = abjad.Score([staff])
        score_lp = abjad.lilypond(score)

        lpf_str = preamble + score_lp

        return lpf_str
