from __future__ import annotations
from scipy.io import wavfile
from thebeat.core.sequence import BaseSequence, Sequence
from thebeat.core.stimulus import Stimulus
import numpy as np
import thebeat.helpers
import thebeat._warnings
import warnings
import os
from typing import Union, Optional
import sounddevice as sd


class StimSequence(BaseSequence):
    """
    The :py:class:`StimSequence` class can be thought of as a combination of the :py:class:`Stimulus` class and the
    :py:class:`Sequence` class; hence *StimSequence*. It combines the timing information of a :py:class:`Sequence`
    object with the auditory information (sound) of a :py:class:`Stimulus` object.
    In most research one would refer to a :py:class:`StimSequence` as a trial (which is also the
    variable name used in some of the examples here).

    One can construct a :py:class:`StimSequence` object either by passing it a single :py:class:`Stimulus` object (and
    a :py:class:`Sequence` object), or by passing it an array or list of :py:class:`Stimulus` objects
    (and a :py:class:`Sequence` object).

    If a single :py:class:`Stimulus` object is passed, this Stimulus sound is used for each event onset. Otherwise,
    each :py:class:`Stimulus` sound is used for its respective event onsets. Of course, then the number of
    :py:class:`Stimulus` objects in the iterable must be the same as the number of event onsets.

    :py:class:`StimSequence` objects can be plotted, played, written to disk, statistically analyzed, and more...
    During construction, checks are done to ensure you dit not accidentally use stimuli that are longer
    than the IOIs (impossible), that the sampling frequencies of all the :py:class:`Stimulus` objects are the same
    (undesirable), and that the :py:class:`Stimulus` objects' number of channels are the same (probable).

    """

    def __init__(self,
                 stimulus: Union[Stimulus, list[Stimulus], np.typing.NDArray[Stimulus]],
                 sequence: Sequence,
                 sequence_time_unit: str = "ms",
                 name: Optional[str] = None):
        """
        Initialize a `StimSequence` object using a `Stimulus` object, or list or array of
        :py:class:`Stimulus` objects, and a :py:class:`Sequence` object.

        During the construction of a `StimSequence` object, sound is generated on the basis of the passed
        `Stimulus` objects and the passed `Sequence` object. A warning is issued if the frame number, where
        one of the sounds would be placed, had to be rounded off. To get rid of this warning, you can use
        the `Sequence.round_onsets` method before passing it to the `StimSequence` constructor,
        or try a different sampling frequency for the `Stimulus` sound.

        Parameters
        ----------
        stimulus
            Either a single Stimulus object (in which case the same sound is used for each event onset), or a
            list or array of Stimulus objects (in which case different sounds are used for each event onset).
        sequence
            A Sequence object. This contains the timing information for the played events.
        sequence_time_unit
            If the Sequence object was created using seconds, use "s". The default is milliseconds ("ms").
        name
            You can provide a name for the :py:class:`StimSequence` which is sometimes used
            (e.g. when printing the object, or when plotting one). You can always retrieve this attribute from
            :py:attr:`StimSequence.name`.

        Examples
        --------
        >>> stim = Stimulus.generate(freq=440)
        >>> seq = Sequence.generate_isochronous(n_events=5,ioi=500)
        >>> trial = StimSequence(stim, seq)

        >>> from random import randint
        >>> stims = [Stimulus.generate(freq=randint(100, 1000)) for x in range(5)]
        >>> seq = Sequence.generate_isochronous(n_events=5,ioi=500)
        >>> trial = StimSequence(stims, seq)
        """

        # If a single Stimulus object is passed, repeat that stimulus for each onset
        # Otherwise use the array/list of Stimlus objects.
        if isinstance(stimulus, Stimulus):
            stimuli = [stimulus] * len(sequence.onsets)
        elif isinstance(stimulus, list) or isinstance(stimulus, np.ndarray):
            if len(stimulus) != len(sequence.onsets):
                raise ValueError("Please provide an equal number of stimuli as onsets.")
            stimuli = stimulus
        else:
            raise ValueError("Please pass a Stimulus object, or a list or array of Stimulus objects.")

        # Type checking for sequence
        if not isinstance(sequence, Sequence):
            raise ValueError("Please provide a Sequence object as the second argument.")

        # Get IOIs from sequence
        iois = sequence.iois

        # If we're dealing with seconds, internally change to milliseconds
        if sequence_time_unit == "s":
            iois *= 1000

        # Check whether dtype, number of channels etc. is the same. This function raises errors if that's
        # not the case
        thebeat.helpers.check_sound_properties_sameness(stimuli)

        # Save attributes
        self.fs = stimuli[0].fs
        self.n_channels = stimuli[0].n_channels
        self.name = name
        self.stim_names = [stimulus.name for stimulus in stimuli]
        self.end_with_interval = sequence.end_with_interval
        self.stim_objects = stimuli

        # Initialize BaseSequence class
        super().__init__(iois, end_with_interval=sequence.end_with_interval, name=name)

        # Check whether there's overlap between the stimuli with these IOIs
        stimulus_durations = [stim.duration_ms for stim in stimuli]
        thebeat.helpers.check_for_overlap(stimulus_durations=stimulus_durations, onsets=self.onsets)

        # Make sound which saves the samples to self.samples
        self.samples = self._make_stimseq_sound(stimuli=stimuli, onsets=self.onsets)

    def __add__(self, other):
        return thebeat.utils.join([self, other])

    def __mul__(self, other: int):
        return self._repeat(times=other)

    def __str__(self):
        # Name of the StimSequence
        name = self.name if self.name else "Not provided"

        # Names of the stimuli
        if all(stim_name is None for stim_name in self.stim_names):
            stim_names = "None provided"
        else:
            stim_names = []
            for stim_name in self.stim_names:
                stim_names.append(stim_name if stim_name else "Unknown")

        end_with_interval = "(ends with interval)" if self.end_with_interval else "(ends with event)"

        return (f"Object of type StimSequence {end_with_interval}:\n"
                f"{len(self.onsets)} events\n"
                f"IOIs: {self.iois}\n"
                f"Onsets: {self.onsets}\n"
                f"Stimulus names: {stim_names}\n"
                f"StimSequence name: {name}")

    def __repr__(self):
        if self.name:
            return f"StimSequence(name={self.name}, n_events={len(self.onsets)})"

        return f"StimSequence(n_events={len(self.onsets)})"

    def play(self,
             loop: bool = False,
             metronome: bool = False,
             metronome_amplitude: float = 1.0):
        """
        This method uses the :func:`sounddevice.play` to play the object's audio.

        Parameters
        ----------
        loop
            If ``True``, the :py:class:`StimSequence` will continue playing until the :py:meth:`StimSequence.stop`
            method is called.
        metronome
            If ``True``, a metronome sound is added for playback.
        metronome_amplitude
            If desired, when playing the object with a metronome sound you can adjust the
            metronome amplitude. A value between 0 and 1 means a less loud metronome, a value larger than 1 means
            a louder metronome sound.

        Examples
        --------
        >>> stim = Stimulus.generate(offramp_ms=10)
        >>> seq = Sequence.generate_random_normal(n_events=10,mu=500,sigma=50)
        >>> stimseq = StimSequence(stim, seq)
        >>> stimseq.play(metronome=True)  # doctest: +SKIP

        """
        thebeat.helpers.play_samples(samples=self.samples, fs=self.fs, mean_ioi=self.mean_ioi, loop=loop,
                                     metronome=metronome, metronome_amplitude=metronome_amplitude)

    def stop(self) -> None:
        """
        Stop playing the :py:class:`StimSequence` sound. Calls :func:`sounddevice.stop`.

        Examples
        --------
        >>> import time  # doctest: +SKIP
        >>> stim = Stimulus.generate()  # doctest: +SKIP
        >>> seq = Sequence([500, 300, 800])  # doctest: +SKIP
        >>> stimseq = StimSequence(stim, seq)  # doctest: +SKIP
        >>> stimseq.play()  # doctest: +SKIP
        >>> time.sleep(secs=1)  # doctest: +SKIP
        >>> stimseq.stop()  # doctest: +SKIP
        """

        sd.stop()

    def plot_sequence(self,
                      linewidth: Optional[Union[float, list[float], np.typing.NDArray[float]]] = None,
                      **kwargs):
        """
        Plot the StimSequence object as an event plot on the basis of the event onsets and their durations.
        See :py:func:`thebeat.visualization.plot_single_sequence`.

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
        >>> stim = Stimulus.generate()
        >>> seq = Sequence.generate_isochronous(n_events=5,ioi=500)
        >>> trial = StimSequence(stim, seq)
        >>> trial.plot_sequence()  # doctest: +SKIP

        """

        # Make title
        if self.name and kwargs.get('title') is None:
            kwargs.get('title', self.name)

        # The linewidths are the event durations for a StimSequence unless otherwise specified
        if linewidth:
            linewidths = [linewidth] * len(self.onsets) if isinstance(linewidth, (int, float)) else linewidth
        else:
            linewidths = self.event_durations

        # If we're dealing with a long StimSequence, plot seconds instead of milliseconds
        if np.max(self.onsets) > 10000:
            linewidths /= 1000
            onsets = self.onsets / 1000
            x_axis_label = "Time (s)"
            final_ioi = self.iois[-1] / 1000
        else:
            onsets = self.onsets
            x_axis_label = "Time (ms)"
            final_ioi = self.iois[-1]

        fig, ax = thebeat.helpers.plot_single_sequence(onsets=onsets,
                                                       end_with_interval=self.end_with_interval,
                                                       final_ioi=final_ioi,
                                                       x_axis_label=x_axis_label,
                                                       linewidths=linewidths, **kwargs)

        return fig, ax

    def plot_waveform(self, **kwargs):
        """
        Plot the StimSequence as a waveform. Equivalent to :py:meth:`Stimulus.plot`.

        Parameters
        ----------
        **kwargs
            Additional parameters (e.g. 'title', 'dpi' are passed to :py:meth:`thebeat.helpers.plot_waveform`).

        Examples
        --------
        >>> trial = StimSequence(Stimulus.generate(), Sequence.generate_isochronous(n_events=10,ioi=500))
        >>> trial.plot_waveform()  # doctest: +SKIP

        """
        if self.name and kwargs.get('title') is None:
            kwargs.get('title', self.name)

        fig, ax = thebeat.helpers.plot_waveform(samples=self.samples, fs=self.fs, n_channels=self.n_channels,
                                                **kwargs)

        return fig, ax

    def write_wav(self,
                  filepath: Union[str, os.PathLike],
                  metronome: bool = False,
                  metronome_amplitude: float = 1.0):
        """
        Parameters
        ----------
        filepath
            The output destination for the .wav file. Either pass e.g. a Path object, or a pass a string. Of course be
            aware of OS-specific filepath conventions.
        metronome
            If ``True``, a metronome sound is added for playback.
        metronome_amplitude
            If desired, when playing the StimSequence with a metronome sound you can adjust the
            metronome amplitude. A value between 0 and 1 means a less loud metronme, a value larger than 1 means
            a louder metronome sound.

        Examples
        --------
        >>> stimseq = StimSequence(Stimulus.generate(), Sequence.generate_isochronous(n_events=5,ioi=500))
        >>> stimseq.write_wav('my_stimseq.wav')  # doctest: +SKIP
        """

        if metronome is True:
            samples = thebeat.helpers.get_sound_with_metronome(samples=self.samples, fs=self.fs,
                                                               metronome_ioi=self.mean_ioi,
                                                               metronome_amplitude=metronome_amplitude)
        else:
            samples = self.samples

        # Make filepath string if it is a Path object
        filepath = str(filepath)

        # Process filepath
        if filepath.endswith('.wav'):
            path, filename = os.path.split(filepath)
        elif os.path.isdir(filepath):
            path = filepath
            filename = f"{self.name}.wav" if self.name else "out.wav"
        else:
            raise ValueError("Wrong filepath specified. Please provide a directory or a complete filepath.")

        # Write the wav
        wavfile.write(filename=os.path.join(path, filename), rate=self.fs, data=samples)

    def _make_stimseq_sound(self, stimuli, onsets):
        """Internal function used for combining different Stimulus samples and a passed Sequence object
        into one array of samples containing the sound of a StimSequence."""

        # Generate an array of silence that has the length of all the onsets + one final stimulus.
        # In the case of a sequence that ends with an interval, we add the final ioi.
        # The dtype is important, because that determines the values that the magnitudes can take.
        if self.end_with_interval:
            array_length = (onsets[-1] + self.iois[-1]) / 1000 * self.fs
        elif not self.end_with_interval:
            array_length = (onsets[-1] / 1000 * self.fs) + stimuli[-1].samples.shape[0]
        else:
            raise ValueError("Error during calculation of array_length")

        if not array_length.is_integer():  # let's avoid rounding issues
            warnings.warn(thebeat._warnings.framerounding_stimseq)

        # Round off array length to ceiling if necessary
        array_length = int(np.ceil(array_length))

        if self.n_channels == 1:
            samples = np.zeros(array_length, dtype=np.float64)
        else:
            samples = np.zeros((array_length, 2), dtype=np.float64)

        samples_with_onsets = zip([stimulus.samples for stimulus in stimuli], onsets)

        for stimulus, onset in samples_with_onsets:
            # Calculate start and end point in frames
            start_pos = onset * self.fs / 1000
            end_pos = start_pos + stimulus.shape[0]

            # Check whether there was frame rounding
            if not start_pos.is_integer() or not end_pos.is_integer():
                warnings.warn(thebeat._warnings.framerounding_stimseq)

            # Now we can safely round
            start_pos = int(start_pos)
            end_pos = int(end_pos)

            # Add the stimulus to the samples array. For stereo, do this for both channels.
            if self.n_channels == 1:
                samples[start_pos:end_pos] = stimulus
            elif self.n_channels == 2:
                samples[start_pos:end_pos, :2] = stimulus

        # set self.event_durations
        self.event_durations = np.array([stim.duration_ms for stim in stimuli])

        # return sound
        if np.max(np.abs(samples)) > 1:
            warnings.warn(thebeat._warnings.normalization)
            return thebeat.helpers.normalize_audio(samples)
        else:
            return samples

    def _repeat(self, times: int):
        if not isinstance(times, int):
            raise ValueError("Can only multiply the StimSequenec by an integer value")

        if not self.end_with_interval or not self.onsets[0] == 0:
            raise ValueError("You can only repeat sequences that end with an interval."
                             "Try adding the end_with_interval=True flag when creating the Sequence object.")

        new_iois = np.tile(self.iois, reps=times)
        new_seq = Sequence(iois=new_iois, first_onset=0, end_with_interval=True)
        new_stims = np.tile(self.stim_objects, reps=times)

        return StimSequence(stimulus=new_stims, sequence=new_seq, name=self.name)
