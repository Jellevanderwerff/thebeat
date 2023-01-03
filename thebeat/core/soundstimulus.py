from __future__ import annotations
import os
import re
from typing import Optional, Union
import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt
import scipy.io
import scipy.signal
import thebeat.helpers
import copy

try:
    import abjad
except ImportError:
    abjad = None


class SoundStimulus:
    """
    A SoundStimulus object holds a sound. You can use the different class methods to generate a sound,
    to get a sound from a .wav file, or to import a :class:`parselmouth.Sound` object.
    This :py:class:`SoundStimulus` sound is used when generating a trial with the :py:class:`SoundSequence` class.
    Additionally, one can plot the object, change the amplitude, etc.

    """

    def __init__(self,
                 samples: np.ndarray,
                 fs: int,
                 name: Optional[str] = None):
        """
        The constructor for the :py:class:`SoundStimulus` class. Except for special cases, this is only used internally.
        You'll most likely want to use one of the different class methods to construct a :py:class:`SoundStimulus` object,
        such as :py:meth:`SoundStimulus.generate` or :py:meth:`SoundStimulus.from_wav`.

        Both mono and stereo sounds are supported.

        Notes
        -----
        For more information on how these samples are created, see the `SciPy documentation
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html#scipy-io-wavfile-read>`_.

        Parameters
        ----------
        samples
            An array containing the audio samples with frequency ``fs``.
        fs
            The sampling frequency.
        name
            Optionally, the :py:class:`SoundStimulus` object can have a name. This is saved to the :py:attr:`SoundStimulus.name`
            attribute.
        """

        # Check number of channels from array's dimensions:
        if samples.ndim == 1:
            n_channels = 1
        elif samples.ndim == 2:
            n_channels = 2
        else:
            raise ValueError("Wrong number of dimensions in given samples. Can only be 1 (mono) or 2 (stereo).")

        # Save attributes
        self.samples = samples
        self.fs = fs
        self.dtype = samples.dtype
        self.n_channels = n_channels
        self.name = name

    def __add__(self, other: SoundStimulus):
        if not isinstance(other, SoundStimulus):
            raise ValueError("Can only overlay another SoundStimulus object on this SoundStimulus object.")

        # Check sameness of number of channels etc.
        thebeat.helpers.check_sound_properties_sameness([self, other])

        # Overlay sounds
        samples = thebeat.helpers.overlay_samples([self.samples, other.samples])

        return SoundStimulus(samples=samples, fs=self.fs, name=self.name)

    def __mul__(self, other: int):
        if not isinstance(other, int):
            raise ValueError("Can only multiply by an integer.")

        return SoundStimulus(samples=np.tile(self.samples, other), fs=self.fs, name=self.name)

    def __repr__(self):
        if self.name:
            return f"SoundStimulus(name={self.name}, duration_ms={self.duration_ms})"
        return f"SoundStimulus(duration_ms={self.duration_ms})"

    def __str__(self):

        return (f"Object of type SoundStimulus\n"
                f"SoundStimulus name: {self.name if self.name else 'Not provided'}\n"
                f"SoundStimulus duration: {self.duration_ms} ms\n"
                f"Number of channels: {self.n_channels}\n"
                f"Sampling frequency {self.fs}")

    def copy(self):
        """Returns a shallow copy of itself"""
        return copy.copy(self)

    @classmethod
    def from_wav(cls,
                 filepath: Union[os.PathLike, str],
                 new_fs: Optional[int] = None,
                 name: Optional[str] = None) -> SoundStimulus:
        """
        This method loads a stimulus from a PCM ``.wav`` file, and reads in the samples.
        If necessary, it additionally converts the ``dtype`` to :class:`numpy.float64`.

        The standard behaviour is that the sampling frequency (``fs``) of the input file is used.
        If desired, the input file can be resampled using the ``new_fs`` parameter.

        Parameters
        ----------
        filepath
            The location of the .wav file. Either pass it e.g. a Path object, or a string.
            Of course be aware of OS-specific filepath conventions.
        name
            If desired, one can give a SoundStimulus object a name. This is used, for instance,
            when plotting or printing. It can always be retrieved from the SoundStimulus.name atrribute.
        new_fs
            If resampling is required, you can provide the target sampling frequency here, for instance ``48000``.

        """

        # Read in the sampling frequency and all the samples from the wav file
        samples, fs = _read_wavfile(filepath=filepath, new_fs=new_fs)

        # Return SoundStimulus object
        return cls(samples, fs, name)

    @classmethod
    def generate(cls,
                 freq: int = 440,
                 fs: int = 48000,
                 duration_ms: float = 50,
                 n_channels: int = 1,
                 amplitude: float = 1.0,
                 oscillator: str = 'sine',
                 onramp_ms: float = 0,
                 offramp_ms: float = 0,
                 ramp_type: str = 'linear',
                 name: Optional[str] = None) -> SoundStimulus:
        """
        This class method returns a SoundStimulus object with a generated sound, based on the given parameters.
        It is recommended to use the on- and offramp parameters for the best results.

        Parameters
        ----------
        freq
            The pitch frequency in hertz.
        fs
            The sampling frequency in hertz.
        duration_ms
            The duration in milliseconds.
        n_channels
            The number of channels. 1 for mono, 2 for stereo.
        amplitude
            Factor with which sound is amplified. Values between 0 and 1 result in sounds that are less loud,
            values higher than 1 in louder sounds.
        oscillator
            Either 'sine' (the default) 'square' or 'sawtooth'.
        onramp_ms
            The sound's 'attack' in milliseconds.
        offramp_ms
            The sound's 'decay' in milliseconds.
        ramp_type
            The type of on- and offramp used. Either 'linear' (the default) or 'raised-cosine'.
        name
            Optionally, one can provide a name for the SoundStimulus. This is for instance useful when distinguishing
            A and B stimuli. It is used when the SoundStimulus sound is printed, written to a file, or when it is plotted.

        Examples
        --------
        >>> stim = SoundStimulus.generate(freq=1000, onramp_ms=10, offramp_ms=10)
        >>> stim.plot_waveform()  # doctest: +SKIP

        """

        # Generate signal
        samples = thebeat.helpers.synthesize_sound(duration_ms=duration_ms, fs=fs, freq=freq, n_channels=n_channels,
                                                   amplitude=amplitude, oscillator=oscillator)

        # Make ramps
        samples = thebeat.helpers.make_ramps(samples, fs, onramp_ms, offramp_ms, ramp_type)

        # Return class
        return cls(samples, fs, name)

    @classmethod
    def generate_white_noise(cls,
                             duration_ms: int = 50,
                             sigma: float = 1,
                             fs: int = 48000,
                             n_channels: int = 1,
                             amplitude: float = 1.0,
                             rng: Optional[np.random.Generator] = None,
                             name: Optional[str] = None) -> SoundStimulus:
        """
        This class method returns a SoundStimulus object with white noise. They are simply random samples from a normal
        distribution with mean 0 and standard deviation ``sd``.

        Parameters
        ----------
        duration_ms
            The desired duration in milliseconds.
        sigma
            The standard deviation of the normal distribution from which the samples are drawn.
        fs
            The sampling frequency in hertz.
        n_channels
            The number of channels. 1 for mono, 2 for stereo.
        amplitude
            Factor with which sound is amplified. Values between 0 and 1 result in sounds that are less loud,
            values higher than 1 in louder sounds.
        rng
            A :class:`numpy.random.Generator` object. If not supplied :func:`numpy.random.default_rng` is
            used.
        name
            Optionally, one can provide a name for the SoundStimulus. This is for instance useful when distinguishing
            A and B stimuli. It is used when the SoundStimulus sound is printed, written to a file, or when it is plotted.

        Examples
        --------
        >>> stim = SoundStimulus.generate_white_noise()
        >>> stim.plot_waveform()  # doctest: +SKIP

        """
        if rng is None:
            rng = np.random.default_rng()

        # Generate signal
        n_frames = fs * duration_ms // 1000
        output_shape = n_frames if n_channels == 1 else (n_frames, 2)
        samples = amplitude * rng.normal(loc=0,
                                         scale=sigma,
                                         size=output_shape)

        samples = thebeat.helpers.normalize_audio(samples)

        # Return class
        return cls(samples=samples, fs=fs, name=name)

    @classmethod
    def from_note(cls,
                  note_str: str,
                  duration: int = 50,
                  fs: int = 48000,
                  amplitude: float = 1.0,
                  oscillator: str = 'sine',
                  onramp_ms: int = 0,
                  offramp_ms: int = 0,
                  ramp: str = 'linear',
                  name: Optional[str] = None) -> SoundStimulus:
        """
        Generate a :py:class:`SoundStimulus` object on the basis of a note as a string.
        For instance, a ``note_str`` of ``'G4'`` results in a generated SoundStimulus with a pitch frequency of 392 hertz.

        Parameters
        ----------
        note_str
            A note as a string. Can either be provided as ``'G'`` or ``'G4'``.
        duration
            The duration in milliseconds.
        fs
            The sampling frequency in hertz.
        amplitude
            Factor with which sound is amplified. Values between 0 and 1 result in sounds that are less loud,
            values higher than 1 in louder sounds.
        oscillator
            The oscillator used for generating the sound. Either 'sine' (the default), 'square' or 'sawtooth'.
        onramp_ms
            The sound's 'attack' in milliseconds.
        offramp_ms
            The sound's 'decay' in milliseconds.
        ramp
            The type of on- and offramp_ms used. Either 'linear' (the default) or 'raised-cosine'.
        name
            Optionally, one can provide a name for the SoundStimulus. This is for instance useful when distinguishing
            A and B stimuli. It is used when the SoundStimulus sound is printed, written to a file, or when it is plotted.

        Examples
        --------
        >>> stim = SoundStimulus.from_note('G',duration=20)

        >>> stim = SoundStimulus.from_note('G4',onramp_ms=10, offramp_ms=10, ramp='raised-cosine')

        """

        if abjad is None:
            raise ImportError("This method requires the abjad package. Please install, for instance by typing "
                              "'pip install abjad' in your terminal.")

        note_strings = re.split(r"([A-Z])([0-9]?)", note_str)
        note_strings = [string for string in note_strings if string != '']

        if len(note_strings) == 1:
            freq = abjad.NamedPitch(note_strings[0], octave=4).hertz
        elif len(note_strings) == 2:
            freq = abjad.NamedPitch(note_strings[0], octave=note_strings[1]).hertz
        else:
            raise ValueError("Please provide a string of format 'G' or 'G4'.")

        return SoundStimulus.generate(freq=freq, fs=fs, duration_ms=duration, amplitude=amplitude, oscillator=oscillator,
                                      onramp_ms=onramp_ms, offramp_ms=offramp_ms, ramp_type=ramp, name=name)

    @classmethod
    def from_parselmouth(cls,
                         sound_object,
                         name: Optional[str] = None) -> SoundStimulus:

        """
        This class method generates a :py:class:`SoundStimulus` object from a :class:`parselmouth.Sound` object.

        Parameters
        ----------
        sound_object : :class:`parselmouth.Sound` object
            The to-be imported Parselmouth Sound object.
        name : str, optional
            Optionally, one can provide a name for the SoundStimulus. This is for instance useful when distinguishing
            A and B stimuli. It is used when the SoundStimulus sound is printed, written to a file, or when it is plotted.

        """
        if not sound_object.__class__.__name__ == "Sound":
            raise ValueError("Please provide a parselmouth.Sound object.")

        fs = sound_object.sampling_frequency

        if sound_object.samples.ndim == 1:
            samples = sound_object.values[0]
        elif sound_object.samples.ndim == 2:
            samples = sound_object.values
        else:
            raise ValueError("Incorrect number of dimensions in samples. Should be 1 (mono) or 2 (stereo).")

        return cls(samples, fs, name)

    def change_amplitude(self,
                         factor: float):
        """
        This method can be used to change the amplitude of the SoundStimulus sound.
        A factor between 0 and 1 decreases the amplitude; a factor larger than 1 increases the amplitude.

        Parameters
        ----------
        factor
            The factor by which the sound should be amplified.
        """

        if not factor > 0:
            raise ValueError("Please provide a 'factor' larger than zero.")

        # set samples to new amplitude
        self.samples = self.samples * factor

    # Visualization
    def play(self,
             loop: bool = False) -> None:
        """

        This method uses :func:`sounddevice.play` to play the :py:class:`SoundStimulus` sound.

        Parameters
        ----------
        loop
            If ``True``, the SoundStimulus object is played until the :py:meth:`SoundStimulus.stop` method is called.

        Examples
        --------
        >>> stim = SoundStimulus.generate()
        >>> stim.play()  # doctest: +SKIP
        """

        sd.play(self.samples, self.fs, loop=loop)
        sd.wait()

    def stop(self) -> None:
        """
        Stop playing the :py:class:`SoundStimulus` sound. Calls :func:`sounddevice.stop`.

        Examples
        --------
        >>> import time  # doctest: +SKIP
        >>> stim = SoundStimulus.generate()  # doctest: +SKIP
        >>> stim.play()  # doctest: +SKIP
        >>> time.sleep(secs=1)  # doctest: +SKIP
        >>> stim.stop()  # doctest: +SKIP
        """

        sd.stop()

    def plot_waveform(self,
                      **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """
        Parameters
        ----------
        Plot the SoundSequence as a waveform. Equivalent to :py:meth:`SoundStimulus.plot`.

        Parameters
        ----------
        **kwargs
            Additional parameters (e.g. 'title', 'dpi' are passed to :py:meth:`thebeat.helpers.plot_waveform`).

        Examples
        --------
        >>> stim = SoundStimulus.generate()
        >>> stim.plot_waveform(title="Waveform of stimulus")  # doctest: +SKIP

        >>> import matplotlib.pyplot as plt  # doctest: +SKIP
        >>> stim = SoundStimulus.generate()
        >>> fig, ax = stim.plot_waveform(suppress_display=True)
        >>> fig.set_facecolor('blue')
        >>> plt.show()  # doctest: +SKIP

        """

        if self.name and kwargs.get('title') is None:
            kwargs.get('title', self.name)

        fig, ax = thebeat.helpers.plot_waveform(samples=self.samples, fs=self.fs, n_channels=self.n_channels,
                                                **kwargs)

        return fig, ax

    # Stats
    @property
    def duration_s(self) -> np.float64:
        """
        The duration of the SoundStimulus sound in seconds.
        """
        return np.float64(self.samples.shape[0] / self.fs)

    @property
    def duration_ms(self) -> np.float64:
        """
        The duration of the SoundStimulus sound in milliseconds.
        """
        return np.float64(self.samples.shape[0] / self.fs * 1000)

    def write_wav(self,
                  filepath: Union[str, os.PathLike]) -> None:
        """
        Save the SoundStimulus sound to disk as a wave file.

        Parameters
        ----------
        filepath
            The output destination for the .wav file. Either pass e.g. a ``Path`` object, or a string.
            Of course be aware of OS-specific filepath conventions.

        Examples
        --------
        >>> stim = SoundStimulus.generate()
        >>> stim.write_wav('my_stimulus.wav')  # doctest: +SKIP

        """

        thebeat.helpers.write_wav(samples=self.samples, fs=self.fs, filepath=filepath, metronome=False)


def _read_wavfile(filepath: Union[str, os.PathLike],
                  new_fs: Optional[int]):
    """Internal function used to read a wave file. Returns the wave file's samples and the sampling frequency.
    If dtype is different than np.float64, it converts the samples to that."""
    file_fs, samples = scipy.io.wavfile.read(filepath)

    # Change dtype so we always have float64
    if samples.dtype == 'int16':
        samples = samples.astype(np.float64, order='C') / 32768.0
    elif samples.dtype == 'int32':
        samples = samples.astype(np.float64, order='C') / 2147483648.0
    elif samples.dtype == 'float32':
        samples = samples.astype(np.float64, order='C')
    elif samples.dtype == 'float64':
        pass
    else:
        raise ValueError("Unknown dtype for wav file. 'int16', 'int32', 'float32' and 'float64' are supported:'"
                         "https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html")

    # Resample if necessary
    if new_fs is None:
        fs = file_fs
    else:
        samples, fs = _resample(samples, file_fs, new_fs)

    return samples, fs


def _resample(samples, input_fs, output_fs):
    """Internal function used to resample sounds. Uses scipy.signal.resample"""
    if output_fs == input_fs:
        fs = input_fs
        samples = samples
    elif output_fs != input_fs:
        resample_factor = float(output_fs) / float(input_fs)
        resampled = scipy.signal.resample(samples, int(len(samples) * resample_factor))
        samples = resampled
        fs = output_fs
    else:
        raise ValueError("Error while comparing old and new sampling frequencies.")

    return samples, fs