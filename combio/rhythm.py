# Required imports
from __future__ import annotations
from typing import Union, Optional
import numpy as np


# Optional imports
# Mingus
try:
    import mingus
    from mingus.extra import lilypond as mingus_lilypond
except ImportError:
    mingus = None

# Local imports
import combio.core


class Rhythm(combio.core.sequence.BaseSequence):

    def __init__(self, iois, n_bars: int, time_signature, beat_ms):
        # Save attributes
        self.time_signature = time_signature  # Used for metrical sequences
        self.beat_ms = beat_ms  # Used for metrical sequences
        self.n_bars = n_bars

        if not np.sum(iois) % (n_bars * time_signature[0] * beat_ms) == 0:
            raise ValueError("The provided inter-onset intervals do not amount to whole bars.")

        # Call initializer of super class
        combio.core.sequence.BaseSequence.__init__(self, iois, metrical=True)

    def __str__(self):
        return f"Object of type Rhythm.\nTime signature: {self.time_signature}\nNumber of bars: {self.n_bars}\nQuarternote (" \
               f"ms): {self.beat_ms}\nNumber of events: {len(self.onsets)}\nIOIs: {self.iois}\n" \
               f"Onsets:{self.onsets}\n"

    def __add__(self, other):
        return combio._helpers.join_rhythms([self, other])

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
        print(type(n_bars))

        if not n_bars % 1 != 0:
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
                               events_per_bar=None,
                               rng=None):
        """
        This function returns a randomly generated integer ratio Sequence on the basis of the provided params.
        """

        if rng is None:
            rng = np.random.default_rng()

        iois = np.empty(0)

        all_ratios = combio._helpers.all_rhythmic_ratios(allowed_note_values,
                                                         time_signature,
                                                         target_n=events_per_bar)

        for bar in range(n_bars):
            ratios = rng.choice(all_ratios, 1)[0]
            new_iois = ratios * 4 * beat_ms
            iois = np.append(iois, new_iois)

        return cls(iois, time_signature=time_signature, beat_ms=beat_ms, n_bars=n_bars)

    @classmethod
    def generate_isochronous(cls, n_bars, time_signature, beat_ms):

        n_iois = time_signature[0] * n_bars

        iois = n_iois * [beat_ms]

        return cls(iois=iois,
                   n_bars=n_bars,
                   time_signature=time_signature,
                   beat_ms=beat_ms)

    def plot_rhythm(self, filepath=None, print_staff=False, suppress_display=False):
        if mingus is None:
            raise ValueError("This method requires the 'mingus' Python package."
                             "Install it, for instance by typing 'pip install mingus' into your terminal.")

        # create initial bar
        t = mingus.containers.Track()
        b = mingus.containers.Bar(meter=self.time_signature)

        # keep track of the index of the note_value
        note_i = 0
        # loop over the note values of the sequence

        for note_value in self.note_values:
            b.place_notes('G-4', note_value)

            # if bar is full, create new bar and add bar to track
            if b.current_beat == b.length:
                t.add_bar(b)
                b = mingus.containers.Bar(meter=self.time_signature)

            note_i += 1

        # If final bar was not full yet, add a rest for the remaining duration
        # Todo think what the reasoning was here
        if b.current_beat % 1 != 0:
            rest_value = 1 / b.space_left()
            if not rest_value.is_integer():
                raise ValueError("The rhythm could not be plotted. Most likely because the IOIs cannot "
                                 "be (easily) captured in musical notation. This for instance happens after "
                                 "using one of the tempo manipulation methods.")

            b.place_rest(rest_value)
            t.add_bar(b)

        lp = combio._helpers.get_lp_from_track(t, print_staff=print_staff)

        combio._helpers.plot_lp(lp, filepath, suppress_display=suppress_display)

    def plot_sequence(self,
                      style: str = 'seaborn',
                      title: Optional[str] = None,
                      linewidth: int = 50,
                      figsize: Optional[tuple] = None,
                      suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the :py:class:`Rhythm` object as an event plot on the basis of the event onsets.

        In principle, the `x` axis shows milliseconds. However, if the total sequence duration is over 10 seconds,
        this changes to seconds.

        Parameters
        ----------
        style
            Matplotlib style to use for the plot. Defaults to 'seaborn'.
            See `matplotlib style sheets reference
            <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
        title
            If desired, one can provide a title for the plot. This takes precedence over using the
            StimSequence name as the title of the plot (if the object has one).
        linewidth
            The desired width of the bars (events) in milliseconds.
        figsize
            A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
            This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.figure`.
        suppress_display
            If ``True``, the plot is only returned, and not displayed via :func:`matplotlib.pyplot.show`.

        Examples
        --------
        >>> rhythm = Rhythm.generate_random_rhythm(allowed_note_values=[4, 8, 16], n_bars = 1, beat_ms=500)
        >>> rhythm.plot_sequence()  # doctest: +SKIP
        """

        # If a title was provided that has preference. If none is provided,
        # use the Sequence object's name. Otherwise use ``None``.
        if title:
            title = title
        elif self.name:
            title = self.name

        # Linewidths
        linewidths = np.repeat(linewidth, len(self.onsets))

        fig, ax = combio._helpers.plot_sequence_single(onsets=self.onsets, style=style, title=title,
                                                       linewidths=linewidths, figsize=figsize,
                                                       suppress_display=suppress_display)

        return fig, ax
