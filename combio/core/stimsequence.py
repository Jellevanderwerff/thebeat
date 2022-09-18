from scipy.io import wavfile
from combio.core.sequence import BaseSequence, Sequence
from combio.core.stimulus import Stimulus
import numpy as np
import combio.core.helpers
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
        Contains the NumPy data type object. Hard-coded as np.float32.
    fs : int
        Sampling frequency of the sound. 48000 is used as the standard in this package.
    iois : NumPy 1-D array
        Contains the inter-onset intervals (IOIs). This is the bread and butter of the BaseSequence class.
        Non-metrical sequences have n IOIs and n+1 onsets. Metrical sequences have an equal number of IOIs
        and onsets.
    metrical : bool
            If False, sequence has an n-1 inter-onset intervals (IOIs) for n event onsets. If True,
            sequence has an equal number of IOIs and event onsets.
    n_channels : int
        The StimSequence's number of channels. 1 for mono, 2 for stereo.
    name : str
        Defaults to None. If name is provided during object construction it is saved here.
    samples : NumPy 1-D array (float32)
        Contains the samples of the sound.
    stim_names : list
        A list containing the names of the passed Stimulus object(s).

    Examples
    --------
    >>> stim = Stimulus.generate(freq=440,duration=50)
    >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
    >>> trial = StimSequence(stim, seq)

    >>> from random import randint
    >>> stims = [Stimulus.generate(freq=randint(100, 1000)) for x in range(5)]
    >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
    >>> trial = StimSequence(stims, seq)
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

        Examples
        --------
        >>> stim = Stimulus.generate(freq=440,duration=50)
        >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
        >>> trial = StimSequence(stim, seq)

        >>> from random import randint
        >>> stims = [Stimulus.generate(freq=randint(100, 1000)) for x in range(5)]
        >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
        >>> trial = StimSequence(stims, seq)
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
        """The average inter-onset interval (IOI) in milliseconds."""
        return np.float32(np.mean(self.iois))

    @property
    def duration_ms(self) -> np.float32:
        """The total duration of the StimSequence object in milliseconds.
        """
        return np.float32(np.sum(self.iois))

    @property
    def duration_s(self) -> np.float32:
        """The total duration of the StimSequence object in seconds.
        """
        return np.float32(np.sum(self.iois) / 1000)

    def play(self, loop=False, metronome=False, metronome_amplitude=1) -> None:
        """
        This method uses the sounddevice library to play the StimSequence object.

        Parameters
        ----------
        loop: bool, optional
            If 'True', the StimSequence will continue playing until the StimSequence.stop() method is called.
            Defaults to 'False'.
        metronome: bool, optional
            If 'True', a metronome sound is added for playback. Defaluts to 'False'.
        metronome_amplitude: bool, optional
            If desired, when playing the StimSequence with a metronome sound you can adjust the
            metronome amplitude. A value between 0 and 1 means a less loud metronme, a value larger than 1 means
            a louder metronome sound.

        """
        combio.core.helpers.play_samples(samples=self.samples, fs=self.fs, mean_ioi=self.mean_ioi, loop=loop,
                                         metronome=metronome,
                                         metronome_amplitude=metronome_amplitude)

    def plot_sequence(self,
                      style: str = 'seaborn',
                      title: str = None,
                      figsize: tuple = None,
                      suppress_display: bool = False):
        """
        Plot the StimSequence object as an event plot on the basis of the event onsets and their durations.

        Parameters
        ----------
        style : str, optional
            Matplotlib style to use for the plot. Defaults to 'seaborn'. Please refer to the matplotlib
            docs for other styles.
        title : str, optional
            If desired, one can provide a title for the plot. This takes precedence over using the
            StimSequence name as the title of the plot (if the object has one).
        figsize : tuple, optional
            The desired figure size in inches as a tuple: (width, height).
        suppress_display : bool, optional
            If True, the plot is only returned, and not displayed via plt.show()

        Returns
        -------
        fig : Figure
            A matplotlib Figure object
        ax : Axes
            A matplotlib Axes object

        Examples
        --------
        >>> trial = StimSequence(Stimulus.generate(), Sequence.generate_isochronous(n=5, ioi=500))
        >>> trial.plot_sequence()

        """

        linewidths = self.event_durations

        fig, ax = combio.core.helpers.plot_sequence_single(onsets=self.onsets, style=style, title=title,
                                                           linewidths=linewidths, figsize=figsize,
                                                           suppress_display=suppress_display)

        return fig, ax

    def plot_waveform(self,
                      style: str = 'seaborn',
                      title: str = None,
                      figsize: tuple = None,
                      suppress_display: bool = False):
        """
        Parameters
        ----------
        style : str, optional
            Style used by matplotlib, defaults to 'seaborn'. See matplotlib docs for other styles.
        title : str, optional
            If desired, one can provide a title for the plot.
        figsize : tuple, optional
            A tuple containing the desired output size of the plot in inches, e.g. (4, 1).
        suppress_display : bool, optional
            If 'True', plt.show() is not run. Defaults to 'False'.

        Returns
        -------
        fig : Figure
            A matplotlib Figure object
        ax : Axes
            A matplotlib Axes object

        Examples
        --------
        >>> trial = StimSequence(Stimulus.generate(), Sequence.generate_isochronous(n=10, ioi=500))
        >>> trial.plot_waveform(title="My StimSeq plot")  # doctest: +SKIP

        """
        if self.name and title is None:
            title = self.name

        fig, ax = combio.core.helpers.plot_waveform(samples=self.samples, fs=self.fs, n_channels=self.n_channels,
                                                    style=style,
                                                    title=title, figsize=figsize, suppress_display=suppress_display)

        return fig, ax

    def write_wav(self,
                  out_path: Union[str, os.PathLike] = '.',
                  metronome: bool = False,
                  metronome_amplitude: float = 1.0) -> None:
        """

        Parameters
        ----------
        out_path: str or PathLike object
            The output destination for the .wav file. Either pass e.g. a Path object, or a pass a string. Of course be aware of OS-specific filepath conventions.
        metronome: bool, optional
            If 'True', a metronome sound is added for playback. Defaluts to 'False'.
        metronome_amplitude: bool, optional
            If desired, when playing the StimSequence with a metronome sound you can adjust the
            metronome amplitude. A value between 0 and 1 means a less loud metronme, a value larger than 1 means
            a louder metronome sound.

        Examples
        --------
        >>> stimseq = StimSequence(Stimulus.generate(), Sequence.generate_isochronous(n=5, ioi=500))
        >>> stimseq.write_wav('my_stimseq.wav')  # doctest: +SKIP
        """

        _write_wav(self.samples, self.fs, out_path, self.name, metronome, self.mean_ioi, metronome_amplitude)

    def _make_sound(self, stimuli, onsets):
        """Internal function used for combining different Stimulus samples and a passed Sequence object
        into one array of samples containing the sound of a StimSequence."""
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
            return combio.core.helpers.normalize_audio(samples)
        else:
            return samples


def _write_wav(samples, fs, out_path, name, metronome, metronome_ioi, metronome_amplitude):
    """Internal function used for writing a .wav to disk."""
    if metronome is True:
        samples = combio.core.helpers.get_sound_with_metronome(samples, fs, metronome_ioi, metronome_amplitude)
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
