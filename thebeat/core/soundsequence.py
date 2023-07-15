# Copyright (C) 2022-2023  Jelle van der Werff
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
import os
import warnings

import numpy as np
import sounddevice as sd

import thebeat._warnings
import thebeat.helpers
from thebeat.core.sequence import BaseSequence, Sequence
from thebeat.core.soundstimulus import SoundStimulus


class SoundSequence(BaseSequence):
    """
    The :py:class:`SoundSequence` class can be thought of as a combination of the
    :py:class:`SoundStimulus` class and the :py:class:`Sequence` class; hence *SoundSequence*.
    It combines the timing information of a :py:class:`Sequence` object with the auditory
    information (sound) of a :py:class:`SoundStimulus` object. In most research one would refer to
    a :py:class:`SoundSequence` as a trial (which is also the variable name used in some of the
    examples here). Remember that a :py:class:`Sequence` object is agnostic about the used time
    unit, so when constructing a :py:class:`SoundSequence` object, you can specify the time unit of
    the :py:class:`Sequence` object using the ``sequence_time_unit`` parameter (see
    under :py:meth:`Sequence.__init__`.

    One can construct a :py:class:`SoundSequence` object either by passing it a single
    :py:class:`SoundStimulus` object (and a :py:class:`Sequence` object), or by passing it an array
    or list of :py:class:`SoundStimulus` objects (and a :py:class:`Sequence` object).

    If a single :py:class:`SoundStimulus` object is passed, this SoundStimulus sound is used for
    each event onset. Otherwise, each :py:class:`SoundStimulus` sound is used for its respective
    event onsets. Of course, then the number of :py:class:`SoundStimulus` objects in the iterable
    must be the same as the number of event onsets.

    :py:class:`SoundSequence` objects can be plotted, played, written to disk, statistically
    analyzed, and more... During construction, checks are done to ensure you dit not accidentally
    use sounds that are longer than the IOIs (impossible), that the sampling frequencies of all
    the :py:class:`SoundStimulus` objects are the same (undesirable), and that the
    :py:class:`SoundStimulus` objects' number of channels are the same (probable).

    """

    def __init__(
        self,
        sound: SoundStimulus | list[SoundStimulus] | np.typing.NDArray[SoundStimulus],
        sequence: Sequence,
        sequence_time_unit: str = "ms",
        name: str | None = None,
    ):
        """
        Initialize a :py:class:`SoundSequence` object using a :py:class:`SoundStimulus` object, or
        list or array of :py:class:`SoundStimulus` objects, and a :py:class:`Sequence` object.

        During the construction of a :py:class:`SoundSequence` object, sound is generated on the
        basis of the passed :py:class:`SoundStimulus` objects and the passed :py:class:`Sequence`
        object. A warning is issued if the frame number, where one of the sounds would be placed,
        had to be rounded off. To get rid of this warning, you can use the
        :py:meth:`Sequence.round_onsets` method before passing it to the :py:class`SoundSequence`
        constructor, or try a different sampling frequency for the :py:class`SoundStimulus` sound.

        Parameters
        ----------
        sound
            Either a single :py:class:`SoundStimulus` object (in which case the same sound is used
            for each event onset), or a list or array of :py:class:`SoundStimulus` objects (in
            which case different sounds are used for each event onset).
        sequence
            A :py:class:`Sequence` object. This contains the timing information for the played
            events.
        sequence_time_unit
            If the :py:class:`Sequence` object was created using seconds, use "s". The default is
            milliseconds ("ms").
        name
            You can provide a name for the :py:class:`SoundSequence` which is sometimes used
            (e.g. when printing the object, or when plotting one). You can always retrieve this
            attribute from :py:attr:`SoundSequence.name`.

        Examples
        --------
        >>> sound = SoundStimulus.generate(freq=440)
        >>> seq = Sequence.generate_isochronous(n_events=5, ioi=500)
        >>> trial = SoundSequence(sound, seq)

        >>> from random import randint
        >>> sounds = [SoundStimulus.generate(freq=randint(100, 1000)) for x in range(5)]
        >>> seq = Sequence.generate_isochronous(n_events=5, ioi=500)
        >>> trial = SoundSequence(sounds, seq)
        """

        # If a single SoundStimulus object is passed, repeat that sound for each onset
        # Otherwise use the array/list of Stimlus objects.
        if isinstance(sound, SoundStimulus):
            sounds = [sound] * len(sequence.onsets)
        elif isinstance(sound, list) or isinstance(sound, np.ndarray):
            if len(sound) != len(sequence.onsets):
                raise ValueError("Please provide an equal number of sounds as onsets.")
            sounds = sound
        else:
            raise TypeError(
                "Please pass a SoundStimulus object, or a list or array of SoundStimulus objects."
            )

        # Type checking for sequence
        if not isinstance(sequence, Sequence):
            raise TypeError("Please provide a Sequence object as the second argument.")

        # Get IOIs from sequence
        iois = sequence.iois
        if sequence._first_onset < 0:
            raise ValueError(
                "The first onset of the sequence is negative. This is not allowed "
                "when creating SoundSequence objects."
            )

        # If we're dealing with seconds, internally change to milliseconds
        if sequence_time_unit == "s":
            iois *= 1000

        # Check whether dtype, number of channels etc. is the same. This function raises errors if
        # that's not the case
        thebeat.helpers.check_sound_properties_sameness(sounds)

        # Save attributes
        self.fs = sounds[0].fs
        self.n_channels = sounds[0].n_channels
        self.name = name
        self.sound_names = [sound.name for sound in sounds]
        self.end_with_interval = sequence.end_with_interval
        self.sound_objects = sounds

        # Initialize BaseSequence class
        super().__init__(
            iois,
            end_with_interval=sequence.end_with_interval,
            name=name,
            first_onset=sequence._first_onset,
        )

        # Check whether there's overlap between the sounds with these IOIs
        sounds_durations = [sound.duration_ms for sound in sounds]
        thebeat.helpers.check_for_overlap(sounds_durations=sounds_durations, onsets=self.onsets)

        # Make sound which saves the samples to self.samples
        self.samples = self._make_soundseq_sound(sounds=sounds, onsets=self.onsets)

    def __add__(self, other):
        return thebeat.utils.concatenate_soundsequences([self, other])

    def __mul__(self, other: int):
        return self._repeat(times=other)

    def __str__(self):
        # Name of the SoundSequence
        name = self.name if self.name else "Not provided"

        # Names of the sounds
        if all(sound_name is None for sound_name in self.sound_names):
            sound_names = "None provided"
        else:
            sound_names = []
            for sound_name in self.sound_names:
                sound_names.append(sound_name if sound_name else "Unknown")

        end_with_interval = (
            "(ends with interval)" if self.end_with_interval else "(ends with event)"
        )

        return (
            f"Object of type SoundSequence {end_with_interval}:\n"
            f"{len(self.onsets)} events\n"
            f"IOIs: {self.iois}\n"
            f"Onsets: {self.onsets}\n"
            f"SoundStimulus names: {sound_names}\n"
            f"SoundSequence name: {name}"
        )

    def __repr__(self):
        if self.name:
            return f"SoundSequence(name={self.name}, n_events={len(self.onsets)})"

        return f"SoundSequence(n_events={len(self.onsets)})"

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

    def merge(self, other: thebeat.core.SoundSequence | list[thebeat.core.SoundSequence]):
        """
        Merge this :py:class:`SoundSequence` object with one or multiple other
        :py:class:`SoundSequence` objects.

        Returns a new :py:class:`SoundSequence` object.


        Parameters
        ----------
        other
            A :py:class:`SoundSequence` object, or a list of :py:class:`SoundSequence` objects.

        Returns
        -------
        object
            A :py:class:`SoundSequence` object.

        """
        if isinstance(other, thebeat.SoundSequence):
            return thebeat.utils.merge_soundsequences([self, other])

        return thebeat.utils.merge_soundsequences([self, *other])

    def play(self, loop: bool = False, metronome: bool = False, metronome_amplitude: float = 1.0):
        """
        This method uses the :func:`sounddevice.play` to play the object's audio.

        Parameters
        ----------
        loop
            If ``True``, the :py:class:`SoundSequence` will continue playing until the
            :py:meth:`SoundSequence.stop` method is called.
        metronome
            If ``True``, a metronome sound is added for playback.
        metronome_amplitude
            If desired, when playing the object with a metronome sound you can adjust the
            metronome amplitude. A value between 0 and 1 means a less loud metronome, a value
            larger than 1 means a louder metronome sound.

        Examples
        --------
        >>> sound = SoundStimulus.generate(offramp_ms=10)
        >>> seq = Sequence.generate_random_normal(n_events=10, mu=500, sigma=50)
        >>> seq.round_onsets()
        >>> soundseq = SoundSequence(sound, seq)
        >>> soundseq.play(metronome=True)  # doctest: +SKIP

        """
        thebeat.helpers.play_samples(
            samples=self.samples,
            fs=self.fs,
            mean_ioi=self.mean_ioi,
            loop=loop,
            metronome=metronome,
            metronome_amplitude=metronome_amplitude,
        )

    def stop(self) -> None:
        """
        Stop playing the :py:class:`SoundSequence` sound. Calls :func:`sounddevice.stop`.

        Examples
        --------
        >>> import time  # doctest: +SKIP
        >>> sound = SoundStimulus.generate()  # doctest: +SKIP
        >>> seq = Sequence([500, 300, 800])  # doctest: +SKIP
        >>> soundseq = SoundSequence(sound, seq)  # doctest: +SKIP
        >>> soundseq.play()  # doctest: +SKIP
        >>> time.sleep(secs=1)  # doctest: +SKIP
        >>> soundseq.stop()  # doctest: +SKIP
        """

        sd.stop()

    def plot_sequence(
        self, linewidth: float | list[float] | np.typing.NDArray[float] | None = None, **kwargs
    ):
        """
        Plot the :py:class:`SoundSequence` object as an event plot on the basis of the event onsets
        and their durations. See :py:func:`thebeat.visualization.plot_single_sequence`.

        Parameters
        ----------
        linewidth
            The desired width of the bars (events). Defaults to the event durations.
            Can be a single value that will be used for each onset, or a list or array of values
            (i.e with a value for each respective onsets).
        **kwargs
            Additional parameters (e.g. 'title', 'dpi' etc.) are passed to
            :py:func:`thebeat._helpers.plot_single_sequence`.

        Examples
        --------
        >>> sound = SoundStimulus.generate()
        >>> seq = Sequence.generate_isochronous(n_events=5,ioi=500)
        >>> trial = SoundSequence(sound, seq)
        >>> trial.plot_sequence()  # doctest: +SKIP

        """

        # Make title
        if self.name and kwargs.get("title") is None:
            kwargs.get("title", self.name)

        # The linewidths are the event durations for a SoundSequence unless otherwise specified
        if linewidth:
            linewidths = (
                [linewidth] * len(self.onsets) if isinstance(linewidth, (int, float)) else linewidth
            )
        else:
            linewidths = self.event_durations

        # If we're dealing with a long SoundSequence, plot seconds instead of milliseconds
        if np.max(self.onsets) > 10000:
            linewidths /= 1000
            onsets = self.onsets / 1000
            x_axis_label = "Time (s)"
            final_ioi = self.iois[-1] / 1000
        else:
            onsets = self.onsets
            x_axis_label = "Time (ms)"
            final_ioi = self.iois[-1]

        fig, ax = thebeat.helpers.plot_single_sequence(
            onsets=onsets,
            end_with_interval=self.end_with_interval,
            final_ioi=final_ioi,
            x_axis_label=x_axis_label,
            linewidths=linewidths,
            **kwargs,
        )

        return fig, ax

    def plot_waveform(self, **kwargs):
        """
        Plot the :py:class:`SoundSequence` object as a waveform.

        Parameters
        ----------
        **kwargs
            Additional parameters (e.g. 'title', 'dpi' are passed to
            :py:meth:`thebeat.helpers.plot_waveform`).

        Examples
        --------
        >>> sound = SoundStimulus.generate()
        >>> seq = Sequence.generate_isochronous(n_events=10,ioi=500)
        >>> trial = SoundSequence(sound, seq)
        >>> trial.plot_waveform()  # doctest: +SKIP

        """
        if self.name and kwargs.get("title") is None:
            kwargs.get("title", self.name)

        fig, ax = thebeat.helpers.plot_waveform(
            samples=self.samples, fs=self.fs, n_channels=self.n_channels, **kwargs
        )

        return fig, ax

    def write_wav(
        self,
        filepath: str | os.PathLike,
        dtype: str | np.dtype = np.int16,
        metronome: bool = False,
        metronome_ioi: float | None = None,
        metronome_amplitude: float | None = None,
    ):
        """
        Parameters
        ----------
        filepath
            The output destination for the .wav file. Either pass e.g. a Path object, or a pass a
            string. Of course be aware of OS-specific filepath conventions.
        dtype
            The data type of the samples. Defaults to ``np.int16``, meaning that the
            samples are saved as 16-bit integers.
        metronome
            If ``True``, a metronome sound is added for playback.
        metronome_ioi
            If desired, when playing the StimSequence with a metronome sound you can adjust the
            metronome inter-onset interval (IOI). Defaults to the mean IOI of the sequence.
        metronome_amplitude
            If desired, when playing the StimSequence with a metronome sound you can adjust the
            metronome amplitude. A value between 0 and 1 means a less loud metronme, a value larger
            than 1 means a louder metronome sound.

        Examples
        --------
        >>> sound = SoundStimulus.generate()
        >>> seq = Sequence.generate_isochronous(n_events=5,ioi=500)
        >>> soundseq = SoundSequence(sound, seq)
        >>> soundseq.write_wav('my_soundseq.wav')  # doctest: +SKIP
        """

        if metronome is True and metronome_ioi is None:
            metronome_ioi = self.mean_ioi

        thebeat.helpers.write_wav(
            samples=self.samples,
            fs=self.fs,
            filepath=filepath,
            dtype=dtype,
            metronome=metronome,
            metronome_ioi=metronome_ioi,
            metronome_amplitude=metronome_amplitude,
        )

    def _make_soundseq_sound(self, sounds, onsets):
        """Internal function used for combining different SoundStimulus samples and a passed
        Sequence object into one array of samples containing the sound of a SoundSequence."""

        # Generate an array of silence that has the length of all the onsets + one final sound.
        # In the case of a sequence that ends with an interval, we add the final ioi.
        # The dtype is important, because that determines the values that the magnitudes can take.
        if self.end_with_interval:
            array_length = (onsets[-1] + self.iois[-1]) / 1000 * self.fs
        elif not self.end_with_interval:
            array_length = (onsets[-1] / 1000 * self.fs) + sounds[-1].samples.shape[0]
        else:
            raise ValueError("Error during calculation of array_length")

        # Round off array length to ceiling if necessary
        array_length = int(np.ceil(array_length))

        if self.n_channels == 1:
            samples = np.zeros(array_length, dtype=np.float64)
        else:
            samples = np.zeros((array_length, 2), dtype=np.float64)

        samples_with_onsets = zip([sound.samples for sound in sounds], onsets)

        for sound, onset in samples_with_onsets:
            # Calculate start and end point in frames
            start_pos = onset * self.fs / 1000
            end_pos = start_pos + sound.shape[0]

            # Check whether there was frame rounding
            if not start_pos.is_integer() or not end_pos.is_integer():
                warnings.warn(thebeat._warnings.framerounding_soundseq)

            # Now we can safely round
            start_pos = int(start_pos)
            end_pos = int(end_pos)

            # Add the sound to the samples array. For stereo, do this for both channels.
            if self.n_channels == 1:
                samples[start_pos:end_pos] = sound
            elif self.n_channels == 2:
                samples[start_pos:end_pos, :2] = sound

        # set self.event_durations
        self.event_durations = np.array([sound.duration_ms for sound in sounds])

        # return sound
        if np.max(np.abs(samples)) > 1:
            warnings.warn(thebeat._warnings.normalization)
            return thebeat.helpers.normalize_audio(samples)
        else:
            return samples

    def _repeat(self, times: int):
        if not isinstance(times, int):
            raise TypeError("Can only multiply the StimSequenec by an integer value")

        if not self.end_with_interval or not self.onsets[0] == 0:
            raise ValueError(
                "You can only repeat sequences that end with an interval."
                "Try adding the end_with_interval=True flag when creating the Sequence object."
            )

        new_iois = np.tile(self.iois, reps=times)
        new_seq = Sequence(iois=new_iois, first_onset=0, end_with_interval=True)
        new_sounds = np.tile(self.sound_objects, reps=times)

        return SoundSequence(sound=new_sounds, sequence=new_seq, name=self.name)
