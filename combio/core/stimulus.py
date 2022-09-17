from __future__ import annotations
import os
import re
from typing import Union
import numpy as np
import sounddevice as sd
from mingus.containers import Note
from scipy.io import wavfile
from scipy.signal import resample
from combio.core import helpers


class Stimulus:
    """
    Stimulus class that holds a NumPy 1-D array of sound that is generated, or read from a .wav file or Parselmouth
    object.

    Attributes
    ----------
    fs : int
        Sampling frequency of the sound. 48000 is used as the standard in this package.
    samples : NumPy 1-D array (float32)
        Contains the samples of the sound.
    dtype : numpy.dtype
        Contains the NumPy data type object. Hard-coded as np.float32. If a read .wav file has a different dtype,
        the samples will be converted to np.float32.
    n_channels : int
        The Stimulus' number of channels. 1 for mono, 2 for stereo.
    name : str
        Defaults to None. If name is provided during object construction it is saved here.

    """

    def __init__(self,
                 samples: np.ndarray,
                 fs: int,
                 name: str = None):
        """Initialization of Stimulus object. """

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

    def __str__(self):

        return f"Object of type Stimulus\nStimulus name: {self.name}\nStimulus duration: {self.duration_ms} " \
               f"milliseconds"

    @classmethod
    def from_wav(cls,
                 filepath: Union[os.PathLike, str],
                 new_fs: int = None,
                 name: str = None) -> Stimulus:
        """
        This method loads a stimulus from a PCM .wav file, and reads in the samples.
        It additionally converts .wav files with dtype int16 or int32 to float32.

        Parameters
        ----------
        filepath: str or PathLike object
            The location of the .wav file. Either pass it e.g. a Path object, or a string.
            Of course be aware of OS-specific filepath conventions.
        name: str, optional
            If desired, one can give a Stimulus object a name. This is used, for instance,
            when plotting or printing. It can always be retrieved from the Stimulus.name atrribute.
        new_fs : int, optional
            If resampling is required, you can provide the target sampling frequency here.
        """

        # Read in the sampling frequency and all the samples from the wav file
        samples, fs = _read_wavfile(filepath=filepath, new_fs=new_fs)

        # Return Stimulus object
        return cls(samples, fs, name)

    @classmethod
    def generate(cls,
                 freq: int = 440,
                 fs: int = 48000,
                 duration: int = 50,
                 amplitude: float = 1.0,
                 osc: str = 'sine',
                 onramp: int = 0,
                 offramp: int = 0,
                 ramp: str = 'linear',
                 name: str = None) -> Stimulus:
        """
        This class method returns a Stimulus object with a generated sound, based on the given parameters.
        It is recommended to use the on- and offramp parameters for the best results.

        Parameters
        ----------
        freq : int, optional
            The pitch frequency in hertz.
        fs : int, optional
            The sampling frequency in hertz.
        duration : int, optional
            The duration in milliseconds.
        amplitude : float, optional
            Factor with which sound is amplified. Values between 0 and 1 result in sounds that are less loud,
            values higher than 1 in louder sounds.
        osc : str, optional
            Either 'sine' (the default) or 'square'.
        onramp : int, optional
            The sound's 'attack' in milliseconds.
        offramp : int, optional
            The sound's 'decay' in milliseconds.
        ramp : str, optional
            The type of on- and offramp used. Either 'linear' (the default) or 'raised-cosine'.
        name : str, optional
            Optionally, one can provide a name for the Stimulus. This is for instance useful when distinguishing
            A and B stimuli. It is used when the Stimulus sound is printed, written to a file, or when it is plotted.

        Returns
        -------
        Stimulus
            A newly created Stimulus object.

        Examples
        --------
        >>> stim = Stimulus.generate(freq=1000, duration=100, onramp=10, offramp=10)
        >>> stim.plot()

        """
        # Duration in seconds
        t = duration / 1000

        # Generate signal
        samples = np.linspace(0, t, int(fs * t), endpoint=False, dtype=np.float32)
        if osc == 'sine':
            signal = amplitude * np.sin(2 * np.pi * freq * samples)
        elif osc == 'square':
            signal = amplitude * np.square(2 * np.pi * freq * samples)
        else:
            raise ValueError("Choose existing oscillator (for now only 'sine' or 'square')")

        # Make ramps
        signal, fs = _make_ramps(signal, fs, onramp, offramp, ramp)

        # Return class
        return cls(signal, fs, name)

    @classmethod
    def from_note(cls,
                  note_str: str,
                  duration: int = 50,
                  fs: int = 48000,
                  amplitude: float = 1.0,
                  osc: str = 'sine',
                  onramp: int = 0,
                  offramp: int = 0,
                  ramp: str = 'linear',
                  name: str = None) -> Stimulus:
        """
        Generate a Stimulus object on the basis of a note as a string.
        For instance, a note_str of 'G4' results in a generated Stimulus with a pitch frequency of 392 hertz.

        Parameters
        ----------
        note_str: str
            A note as a string. Can either be 'G' or 'G4'.
        duration : int, optional
            The duration in milliseconds.
        fs : int, optional
            The sampling frequency in hertz.
        amplitude : float, optional
            Factor with which sound is amplified. Values between 0 and 1 result in sounds that are less loud,
            values higher than 1 in louder sounds.
        osc : str, optional
            Either 'sine' (the default) or 'square'.
        onramp : int, optional
            The sound's 'attack' in milliseconds.
        offramp : int, optional
            The sound's 'decay' in milliseconds.
        ramp : str, optional
            The type of on- and offramp used. Either 'linear' (the default) or 'raised-cosine'.
        name : str, optional
            Optionally, one can provide a name for the Stimulus. This is for instance useful when distinguishing
            A and B stimuli. It is used when the Stimulus sound is printed, written to a file, or when it is plotted.

        Returns
        -------
        Stimulus
            A newly created Stimulus object.

        Examples
        --------
        >>> stim = Stimulus.from_note('G', duration=20)
        >>> stim.plot()

        >>> stim = Stimulus.from_note('G4', onramp=10, offramp=10, ramp='raised-cosine')
        >>> stim.plot()
        """

        note_strings = re.split(r"([A-Z])([0-9]?)", note_str)
        note_strings = [string for string in note_strings if string != '']

        if len(note_strings) == 1:
            freq = Note(note_strings[0]).to_hertz()

        elif len(note_strings) == 2:
            note, num = tuple(note_strings)
            freq = Note(note, int(num)).to_hertz()
        else:
            raise ValueError("Provide one note as either e.g. 'G' or 'G4' ")

        return Stimulus.generate(freq=freq,
                                 fs=fs,
                                 duration=duration,
                                 amplitude=amplitude,
                                 osc=osc,
                                 onramp=onramp,
                                 offramp=offramp,
                                 ramp=ramp,
                                 name=name)

    @classmethod
    def from_parselmouth(cls, snd_obj, stim_name=None):
        if not snd_obj.__class__.__name__ == "Sound":
            raise ValueError("Please provide a parselmouth.Sound object.")

        fs = snd_obj.sampling_frequency

        if snd_obj.samples.ndim == 1:
            samples = snd_obj.values[0]
        elif snd_obj.samples.ndim == 2:
            samples = snd_obj.values
        else:
            raise ValueError("Incorrect number of dimensions in samples. Should be 1 (mono) or 2 (stereo).")

        return cls(samples, fs, stim_name)

    # Manipulation
    def change_amplitude(self, factor):
        # get original frequencies
        self.samples *= factor

    # Visualization
    def play(self, loop=False):
        sd.play(self.samples, self.fs, loop=loop)
        sd.wait()

    def stop(self):
        sd.stop()

    def plot(self, title=None):
        if title:
            title = title
        else:
            if self.name:
                title = f"Waveform of {self.name}"
            else:
                title = "Waveform of Stimulus"

        helpers.plot_waveform(samples=self.samples, fs=self.fs, n_channels=self.n_channels, title=title)

    # Stats
    @property
    def duration_s(self) -> float:
        return self.samples.shape[0] / self.fs

    @property
    def duration_ms(self) -> float:
        return self.samples.shape[0] / self.fs * 1000

    # Out
    def write_wav(self, out_path: Union[str, os.PathLike]):
        """
        Writes audio to disk.
        """


def _make_ramps(signal, fs, onramp, offramp, ramp):
    # Create onramp
    if onramp > 0:
        onramp_samples_len = int(onramp / 1000 * fs)
        end_point = onramp_samples_len

        if ramp == 'linear':
            onramp_amps = np.linspace(0, 1, onramp_samples_len)

        elif ramp == 'raised-cosine':
            hanning_complete = np.hanning(onramp_samples_len * 2)
            onramp_amps = hanning_complete[:(hanning_complete.shape[0] // 2)]  # only first half of Hanning window

        else:
            raise ValueError("Unknown ramp type. Use 'linear' or 'raised-cosine'")

        signal[:end_point] *= onramp_amps

    elif onramp < 0:
        raise ValueError("Onramp cannot be negative")
    elif onramp == 0:
        pass
    else:
        raise ValueError("Wrong value supplied to onramp argument.")

    # Create offramp
    if offramp > 0:
        offramp_samples_len = int(offramp / 1000 * fs)
        start_point = signal.shape[0] - offramp_samples_len

        if ramp == 'linear':
            offramp_amps = np.linspace(1, 0, int(offramp / 1000 * fs))
        elif ramp == 'raised-cosine':
            hanning_complete = np.hanning(offramp_samples_len * 2)
            offramp_amps = hanning_complete[hanning_complete.shape[0] // 2:]
        else:
            raise ValueError("Unknown ramp type. Use 'linear' or 'raised-cosine'")

        signal[start_point:] *= offramp_amps

    elif offramp < 0:
        raise ValueError("Offramp cannot be negative")
    elif offramp == 0:
        pass
    else:
        raise ValueError("Wrong value supplied to offramp argument.")

    return signal, fs


def _read_wavfile(filepath: Union[str, os.PathLike],
                  new_fs: int):
    file_fs, samples = wavfile.read(filepath)

    # Change dtype so we always have float32
    if samples.dtype == 'int16':
        samples = samples.astype(np.float32, order='C') / 32768.0
    elif samples.dtype == 'int32':
        samples = samples.astype(np.float32, order='C') / 2147483648.0
    elif samples.dtype == 'float32':
        pass
    else:
        raise ValueError("Unknown dtype for wav file. 'int16', 'int32' and 'float32' are supported:'"
                         "https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html")

    # Resample if necessary
    if new_fs is None:
        fs = file_fs
    else:
        samples, fs = _resample(samples, file_fs, new_fs)

    return samples, fs


def _resample(samples, input_fs, output_fs):
    if output_fs == input_fs:
        fs = input_fs
        samples = samples
    elif output_fs != input_fs:
        resample_factor = float(output_fs) / float(input_fs)
        resampled = resample(samples, int(len(samples) * resample_factor))
        samples = resampled
        fs = output_fs
    else:
        raise ValueError("Error while comparing old and new sampling frequencies.")

    return samples, fs