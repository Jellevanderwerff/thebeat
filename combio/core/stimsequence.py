from scipy.io import wavfile
from .sequence import BaseSequence, Sequence
from .stimulus import Stimulus
import numpy as np
from combio.core.helpers import play_samples, plot_waveform, normalize_audio, get_sound_with_metronome
import warnings
import os
from typing import Iterable, Union


class StimSequence(BaseSequence):
    """
    The StimSequence class can be thought of as a combination of the Stimulus class and the Sequence class;
    hence StimSequence. It combines the timing information of a Sequence object with the auditory information (sound)
    of a Stimulus object. In most research one would refer to a StimSequence as a trial (which is also the
    variable name used in all the examples here).

    One can construct a StimSequence object either by passing it a single Stimulus object (and
    a Sequence object), or by passing it an iterable (e.g. list) of Stimulus objects (and a Sequence object).

    If a single Stimulus object is passed, this Stimulus sound is used for each event onset. Otherwise,
    each Stimulus sound is used for its respective event onsets. Of course, then the number of Stimulus
    objects in the iterable must be the same as the number of event onsets.

    StimSequence objects can be plotted, played, written to disk, statistically analyzed, and more...
    During construction, checks are done to ensure one dit not accidentally use stimuli that are longer
    than the IOIs (impossible), that the sampling frequencies of all the Stimulus objects are the same (undesirable),
    and that the Stimulus objects' number of channels are the same (probable).

    Attributes
    ----------
    dtype : numpy.dtype
        blabla
    fs : int
        blabla
    iois : numpy.ndarray
        blabla
    metrical : bool
        blabla
    n_channels : int
        blabla
    name : str
        blabla
    samples : np.ndarray
        blabla
    stim_names : list
        blabla

    Examples
    --------
    >>> stim = Stimulus.generate(freq=440, duration=50)
    >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
    >>> trial = StimSequence(stim, seq)

    >>> from random import randint
    >>> stims = [Stimulus.generate(freq=randint(100, 1000) for x in range(5))]
    >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
    >>> trial = StimSequence(stim, seq)
    """

    def __init__(self,
                 stimulus: Union[Stimulus, Iterable],
                 sequence: Sequence,
                 name: str = None):
        """
        Initialize a StimSequence object using a Stimulus object, or an iterable of Stimulus objects, and a Sequence
        object.

        Parameters
        ----------
        stimulus : Stimulus, or iterable
            Either a single Stimulus object (in which case the same sound is used for each event onset), or an
            iterable of Stimulus objects (in which case different sounds are used for each event onset).
        sequence : Sequence
            A Sequence object. This contains the timing information for the played events.
        name : str, optional
            You can provide a name for the StimSequence which is sometimes used (e.g. when printing a StimSequence
            object, or when plotting one). One can always retrieve this attribute from StimSequence.name
        """

        # If a single Stimulus object is passed, repeat that stimulus for each onset
        if isinstance(stimulus, Stimulus):
            stimuli = [stimulus] * len(sequence.onsets)
        else:
            stimuli = stimulus

        # Type checking for sequence
        if not isinstance(sequence, Sequence):
            raise ValueError("Please provide a Sequence object as the second argument.")

        # Check whether fs, dtype, and n_channels are the same for all stimuli
        all_fs = [snd.fs for snd in stimuli]
        all_dtypes = [snd.dtype for snd in stimuli]
        all_n_channels = [snd.n_channels for snd in stimuli]

        if not all(fs == all_fs[0] for fs in all_fs):
            raise ValueError("The Stimulus objects in the passed list have different sampling frequencies!")
        elif not all(dtype == all_dtypes[0] for dtype in all_dtypes):
            raise ValueError("The Stimulus objects in the passed list have different dtypes!")
        elif not all(n_channels == all_n_channels[0] for n_channels in all_n_channels):
            raise ValueError("The Stimulus objects in the passed list have differing number of channels!")

        # Save attributes
        self.fs = stimuli[0].fs
        self.dtype = stimuli[0].dtype
        self.n_channels = stimuli[0].n_channels
        self.name = name
        self.stim_names = [stimulus.name for stimulus in stimuli]
        self.metrical = sequence.metrical

        # Initialize Sequence class
        BaseSequence.__init__(self, sequence.iois, metrical=sequence.metrical)

        # Make sound which saves the samples to self.samples
        self.samples = self._make_sound(stimuli, self.onsets)

    def __str__(self):

        if self.name:
            name = self.name
        else:
            name = "Not provided"

        if np.all(self.stim_names is None):
            stim_names = "None provided"
        else:
            stim_names = []
            for stim_name in self.stim_names:
                if stim_name is None:
                    stim_names.append("Unknown")
                else:
                    stim_names.append(stim_name)

        if self.metrical:
            return f"""
Object of type StimSequence (metrical version):
StimSequence name: {name}
{len(self.onsets)} events
IOIs: {self.iois}
Onsets: {self.onsets}
Stimulus names: {stim_names}
            """
        else:
            return f"""
Object of type StimSequence (non-metrical version):
StimSequence name: {name}
{len(self.onsets)} events
IOIs: {self.iois}
Onsets: {self.onsets}
Stimulus names: {stim_names}
            """

    @property
    def mean_ioi(self) -> np.float32:
        return np.float32(np.mean(self.iois))

    @property
    def duration_ms(self) -> np.float32:
        """Get the total duration of the StimSequence object in milliseconds.
        """
        return np.float32(np.sum(self.iois))

    @property
    def duration_s(self) -> np.float32:
        """Get the total duration of the StimSequence object in seconds.
        """
        return np.float32(np.sum(self.iois) / 1000)

    def play(self, loop=False, metronome=False, metronome_amplitude=1):
        play_samples(samples=self.samples,
                     fs=self.fs,
                     mean_ioi=self.mean_ioi,
                     loop=loop,
                     metronome=metronome,
                     metronome_amplitude=metronome_amplitude)

    def plot_waveform(self, title=None):
        if title:
            title = title
        else:
            if self.name:
                title = f"Waveform of {self.name}"
            else:
                title = "Waveform of StimSequence"

        plot_waveform(self.samples, self.fs, self.n_channels, title)

    def write_wav(self, out_path: Union[str, os.PathLike] = '.',
                  metronome=False,
                  metronome_amplitude=1):
        """
        Writes audio to disk.
        """

        _write_wav(self.samples, self.fs, out_path, self.name, metronome, self.mean_ioi, metronome_amplitude)

    def _make_sound(self, stimuli, onsets):
        # Check for overlap
        for i in range(len(onsets)):
            stim_duration = stimuli[i].samples.shape[0] / self.fs * 1000
            try:
                ioi_after_onset = onsets[i + 1] - onsets[i]
                if ioi_after_onset < stim_duration:
                    raise ValueError(
                        "The duration of one or more stimuli is longer than its respective IOI. "
                        "The events will overlap: either use different IOIs, or use a shorter stimulus sound.")
            except IndexError:
                pass

        # Generate an array of silence that has the length of all the onsets + one final stimulus.
        # In the case of a metrical sequence, we add the final ioi
        # The dtype is important, because that determines the values that the magnitudes can take.
        if self.metrical:
            length = (onsets[-1] + self.iois[-1]) / 1000 * self.fs
        elif not self.metrical:
            length = (onsets[-1] / 1000 * self.fs) + stimuli[-1].samples.shape[0]
        else:
            raise ValueError("Error during calculation of array_length")

        if int(length) != length:  # let's avoid rounding issues
            warnings.warn("Number of frames was rounded off to nearest integer ceiling. "
                          "This shouldn't be much of a problem.")

        # Round off array length to ceiling if necessary
        array_length = int(np.ceil(length))

        if self.n_channels == 1:
            samples = np.zeros(array_length, dtype=self.dtype)
        else:
            samples = np.zeros((array_length, 2), dtype=self.dtype)

        samples_with_onsets = list(zip([stimulus.samples for stimulus in stimuli], onsets))

        for stimulus, onset in samples_with_onsets:

            start_pos = int(onset * self.fs / 1000)
            end_pos = int(start_pos + stimulus.shape[0])

            if self.n_channels == 1:
                samples[start_pos:end_pos] = stimulus
            elif self.n_channels == 2:
                samples[start_pos:end_pos, :2] = stimulus

        # set self.event_durations
        self.event_durations = np.array([stim.duration_ms for stim in stimuli])

        # return sound
        if np.max(samples) > 1:
            warnings.warn("Sound was normalized")
            return normalize_audio(samples)
        else:
            return samples


def _write_wav(samples, fs, out_path, name, metronome, metronome_ioi, metronome_amplitude):
    if metronome is True:
        samples = get_sound_with_metronome(samples, fs, metronome_ioi, metronome_amplitude)
    else:
        samples = samples

    out_path = str(out_path)

    if out_path.endswith('.wav'):
        path, filename = os.path.split(out_path)
    elif os.path.isdir(out_path):
        path = out_path
        if name:
            filename = f"{name}.wav"
        else:
            filename = f"out.wav"

    else:
        raise ValueError("Wrong out_path specified. Please provide a directory or a complete filepath.")

    wavfile.write(filename=os.path.join(path, filename), rate=fs, data=samples)
