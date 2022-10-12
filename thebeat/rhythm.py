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
import thebeat.core
from thebeat._decorators import requires_lilypond


class Rhythm(thebeat.core.sequence.BaseSequence):
    """
    The :py:class:`Rhythm` class can be used for working sequences that are rhythmical in the musical sense.
    This means that in addition to having inter-onset intervals (IOIs) that represent the timing of the events in the
    sequence, :py:class:`Rhythm` objects have musical properties such as a time signature, a beat duration,
    and they may contain rests.

    The :py:class:`Rhythm` class is also used as the basis for a :py:class:`~thebeat.melody.Melody`.

    For more info on these properties, please refer to the class's :py:meth:`~thebeat.rhythm.Rhythm.__init__`.
    """

    def __init__(self,
                 iois: Union[np.ndarray, list],
                 time_signature: tuple = (4, 4),
                 beat_ms: float = 500,
                 is_played: Optional[Union[npt.NDArray[bool], list[bool]]] = None,
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
        self.time_signature = time_signature  # Used for metrical sequences
        self.beat_ms = beat_ms  # Used for metrical sequences
        self.is_played = [True] * len(iois) if not is_played else list(is_played)

        # Calculate n_bars and check whether that makes whole bars
        n_bars = np.sum(iois) / time_signature[0] / beat_ms
        if not n_bars.is_integer():
            raise ValueError("The provided inter-onset intervals do not amount to whole bars.")
        self.n_bars = n_bars

        # Call initializer of super class
        super().__init__(iois=iois, metrical=True, name=name)

    def __str__(self):
        return (f"Object of type Rhythm.\n"
                f"Name: {self.name}"
                f"Time signature: {self.time_signature}\n"
                f"Number of bars: {self.n_bars}\n"
                f"Beat (ms): {self.beat_ms}\n"
                f"Number of events: {len(self.onsets)}\n"
                f"IOIs: {self.iois}\n"
                f"Onsets:{self.onsets}\n")

    def __repr__(self):
        if self.name:
            return f"Rhythm(name={self.name}, n_bars={self.n_bars}"

        return f"Rhythm(n_bars={self.n_bars}"

    def __add__(self, other):
        return thebeat._helpers.join_rhythms([self, other])

    def __len__(self):
        return len(self.onsets)

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

        ratios = self.iois / self.beat_ms / 4

        note_values = np.array([int(1 // ratio) for ratio in ratios])

        return note_values

    @classmethod
    def from_note_values(cls,
                         note_values: Union[npt.NDArray[int], list[int]],
                         time_signature: tuple[int, int] = (4, 4),
                         beat_ms: int = 500,
                         is_played: Optional[Union[list[bool], npt.NDArray[bool]]] = None,
                         name: Optional[str] = None) -> Rhythm:
        r"""

        This class method may be used for creating a Rhythm object on the basis of note values (i.e. durations).

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
                               allowed_note_values: Optional[Union[list[int], npt.NDArray[int]]] = None,
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

        all_ratios = thebeat._helpers.all_rhythmic_ratios(allowed_note_values, time_signature)

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
                             is_played: Optional[Union[list[bool], npt.NDArray[bool]]] = None,
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
                    print_staff: bool = False,
                    suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:
        """
        Make a plot containing the musical notation of the rhythm. This function requires you to install:

        * `abjad <https://abjad.github.io/>`_ (install via ``pip install abjad``).
        * `lilypond <https://lilypond.org/download.en.html>`_

        For lilypond on Windows, make sure to follow the website's instructions on how to make lilypond available to
        be run through the command line. Linux and Mac OS users are advised to either ``apt install lilypond`` or
        ``brew install lilypond``. On Mac that requires you to install `Homebrew <https://brew.sh/>`_.

        The plot is returned as a :class:`matplotlib.figure.Figure` and :class:`matplotlib.axes.Axes` object,
        which you can manipulate.


        .. figure:: images/plot_rhythm_nostaff.png
            :scale: 50 %
            :class: with-border

            A plot with the default ``print_staff=False``.


        .. figure:: images/plot_rhythm_withstaff.png
            :scale: 50 %
            :class: with-border

            A plot with ``print_staff=True``.

        Caution
        -------
        This method does not check whether the plot makes musical sense. It simply converts
        inter-onset intervals (IOIs) to note values and plots those. Always manually check the plot.


        Parameters
        ----------
        filepath
            Optionally, you can save the plot to a file. Supported file formats are only '.png' and '.eps'.
            The desired file format will be selected based on what the filepath ends with.
        print_staff
            If desired, you can choose to print a musical staff (the default is not to do this). The staff will be a
            `percussion staff <https://en.wikipedia.org/wiki/Percussion_notation>`_.
        suppress_display
            If desired,you can choose to suppress displaying the plot in your IDE. This means that
            :func:`matplotlib.pyplot.show` is not called. This is useful when you just want to save the plot or
            use the returned :class:`matplotlib.figure.Figure` and :class:`matplotlib.axes.Axes` objects.


        Examples
        --------
        >>> r = Rhythm([500, 250, 1000, 250], beat_ms=500)
        >>> r.plot_rhythm()  # doctest: +SKIP

        >>> r = Rhythm([250, 250, 500, 500, 1500], time_signature=(3, 4))
        >>> fig, ax = r.plot_rhythm(print_staff=True, suppress_display=True)  # doctest: +SKIP
        >>> plt.show()  # doctest: +SKIP

        >>> r = Rhythm.from_note_values([4, 4, 4, 4])
        >>> r.plot_rhythm(filepath='isochronous_rhythm.eps')  # doctest: +SKIP


        """
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
        durations = [abjad.Duration((1, int(note_value))) for note_value in self.note_values]
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
        fig, ax = thebeat._helpers.plot_lp(lp=lpf_str, filepath=filepath, suppress_display=suppress_display)

        return fig, ax
