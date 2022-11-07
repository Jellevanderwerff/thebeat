from __future__ import annotations
import os
import re
import textwrap
from typing import Union, Optional
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import warnings
try:
    import abjad
except ImportError:
    abjad = None
from thebeat._decorators import requires_lilypond
import thebeat._warnings
import thebeat.rhythm
import thebeat.helpers
import thebeat.utils
import sounddevice
import numpy.typing as npt


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
                 rhythm: thebeat.rhythm.Rhythm,
                 pitch_names: Union[npt.NDArray[str], list[str], str],
                 octave: Optional[int] = None,
                 key: Optional[str] = None,
                 is_played: Optional[list] = None,
                 name: Optional[str] = None):
        """

        Parameters
        ----------
        rhythm
            A :py:class:`~thebeat.rhythm.Rhythm` object.
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
        >>> r = thebeat.rhythm.Rhythm.from_note_values([4, 4, 4, 4, 4, 4, 2])
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
        super().__init__(iois=rhythm.iois, metrical=True, name=name)

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
        rhythm = thebeat.rhythm.Rhythm.generate_random_rhythm(n_bars=n_bars, beat_ms=beat_ms,
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
        >>> r = thebeat.rhythm.Rhythm([500, 1000, 250, 250], time_signature=(4, 4), beat_ms=500)
        >>> m = Melody(r, pitch_names='CCGC')
        >>> print(r.note_values)  # doctest: +SKIP
        [4 2 8 8]

        >>> r = thebeat.rhythm.Rhythm([166.66666667, 166.66666667, 166.66666667, 500, 500, 500], beat_ms=500]  # doctest: +SKIP
        >>> print(r.note_values)  # doctest: +SKIP
        [12 12 12  4  4  4]

        """

        ratios = self.iois / self.beat_ms / 4

        note_values = np.array([int(1 // ratio) for ratio in ratios])

        return note_values

    @requires_lilypond
    def plot_melody(self,
                    filepath: Optional[Union[os.PathLike, str]] = None,
                    key: Optional[str] = None,
                    suppress_display: bool = False,
                    dpi: int = 300) -> tuple[plt.Figure, plt.Axes]:
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
        >>> r = thebeat.rhythm.Rhythm(iois=[250, 500, 250, 500], time_signature=(3, 4))
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

        if metronome is True:
            samples = thebeat.helpers.get_sound_with_metronome(samples=samples, fs=fs, metronome_ioi=self.beat_ms,
                                                               metronome_amplitude=metronome_amplitude)

        thebeat.helpers.write_wav(samples=samples, fs=fs, filepath=filepath, metronome=metronome,
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

        # Set up what we need
        note_maker = abjad.makers.NoteMaker()
        time_signature = abjad.TimeSignature(time_signature)
        key = abjad.KeySignature(key)
        preamble = textwrap.dedent(r"""
             \version "2.22.2"
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
        note_values = [event.note_value for event in self.events]
        is_played = [event.is_played for event in self.events]

        notes = []

        for pitch_name, note_value, is_played in zip(pitch_names, note_values, is_played):
            duration = abjad.Duration((1, int(note_value)))
            if is_played is True:
                pitch = abjad.NamedPitch(pitch_name)
                note = note_maker(pitch, duration)
            else:
                note = abjad.Rest(duration)
            notes.append(note)

        voice = abjad.Voice(notes)
        abjad.attach(time_signature, voice[0])
        abjad.attach(key, voice[0])

        staff = abjad.Staff([voice])
        score = abjad.Score([staff])
        score_lp = abjad.lilypond(score)

        lpf_str = preamble + score_lp

        return lpf_str
