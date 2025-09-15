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
import re
import warnings

import numpy as np
import sounddevice as sd
from matplotlib import pyplot as plt

import thebeat.helpers

try:
    import abjad
except ImportError:
    abjad = None


class SoundStimulus:
    """A SoundStimulus object holds a sound.

    You can use the different class methods to generate a sound,
    to get a sound from a .wav file, or to import a :class:`parselmouth.Sound` object.
    This :py:class:`SoundStimulus` sound is used when generating a trial with the
    :py:class:`SoundSequence` class. Additionally, one can plot the object, change the amplitude,
    etc.
    """

    def __init__(self, samples: np.ndarray, fs: int, name: str | None = None):
        """The constructor for the :py:class:`SoundStimulus` class. Except for special cases, this
        is only used internally. You'll most likely want to use one of the different class methods
        to construct a :py:class:`SoundStimulus` object, such as :py:meth:`SoundStimulus.generate`
        or :py:meth:`SoundStimulus.from_wav`.

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
            Optionally, the :py:class:`SoundStimulus` object can have a name. This is saved to the
            :py:attr:`SoundStimulus.name` attribute.
        """

        # Check number of channels from array's dimensions:
        if samples.ndim == 1:
            n_channels = 1
        elif samples.ndim == 2:
            n_channels = 2
        else:
            raise ValueError(
                "Wrong number of dimensions in given samples. Can only be 1 (mono) or 2 (stereo)."
            )

        # Save attributes
        self.samples = samples
        self.fs = fs
        self.dtype = samples.dtype
        self.n_channels = n_channels
        self.name = name

    def __add__(self, other: SoundStimulus):
        return thebeat.utils.concatenate_soundstimuli([self, other])

    def __mul__(self, other: int):
        if not isinstance(other, int):
            raise TypeError("Can only multiply by an integer.")

        return SoundStimulus(samples=np.tile(self.samples, other), fs=self.fs, name=self.name)

    def __repr__(self):
        if self.name:
            return f"SoundStimulus(name={self.name}, duration_ms={self.duration_ms})"
        return f"SoundStimulus(duration_ms={self.duration_ms})"

    def __str__(self):
        return (
            f"Object of type SoundStimulus\n"
            f"SoundStimulus name: {self.name if self.name else 'Not provided'}\n"
            f"SoundStimulus duration: {self.duration_ms} ms\n"
            f"Number of channels: {self.n_channels}\n"
            f"Sampling frequency {self.fs}"
        )

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

    @classmethod
    def from_wav(
        cls, filepath: os.PathLike | str, new_fs: int | None = None, name: str | None = None
    ) -> SoundStimulus:
        """This method loads a stimulus from a PCM ``.wav`` file, and reads in the samples. If
        necessary, it additionally converts the ``dtype`` to :class:`numpy.float64`.

        The standard behaviour is that the sampling frequency (``fs``) of the input file is used.
        If desired, the input file can be resampled using the ``new_fs`` parameter.

        Parameters
        ----------
        filepath
            The location of the .wav file. Either pass it e.g. a :class:`pathlib.Path` object, or a
            string. Of course be aware of OS-specific filepath conventions.
        name
            If desired, one can give a :py:class:`SoundStimulus` object a name. This is used, for
            instance, when plotting or printing. It can always be retrieved from the
            :attr:`SoundStimulus.name` atrribute.
        new_fs
            If resampling is required, you can provide the target sampling frequency here,
            for instance ``48000``.

        Examples
        --------

        >>> from thebeat import SoundStimulus
        >>> sound = SoundStimulus.from_wav(filepath="path/to/sound.wav")  # doctest: +SKIP

        >>> sound_newfs = SoundStimulus.from_wav(filepath="path/to/sound.wav", new_fs=48000)  # doctest: +SKIP
        """

        # Read in the sampling frequency and all the samples from the wav file
        samples, fs = thebeat.helpers.read_wav(filepath=filepath, new_fs=new_fs)

        # Return SoundStimulus object
        return cls(samples, fs, name)

    @classmethod
    def generate(
        cls,
        freq: int = 440,
        fs: int = 48000,
        duration_ms: float = 50,
        n_channels: int = 1,
        amplitude: float = 1.0,
        oscillator: str = "sine",
        onramp_ms: float = 0,
        offramp_ms: float = 0,
        ramp_type: str = "linear",
        name: str | None = None,
    ) -> SoundStimulus:
        """This class method returns a SoundStimulus object with a generated sound, based on the
        given parameters. It is recommended to use the on- and offramp parameters for the best
        results.

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
            Factor with which sound is amplified. Values between 0 and 1 result in sounds that are
            less loud, values higher than 1 in louder sounds.
        oscillator
            Either 'sine' (the default) 'square' or 'sawtooth'.
        onramp_ms
            The sound's 'attack' in milliseconds.
        offramp_ms
            The sound's 'decay' in milliseconds.
        ramp_type
            The type of on- and offramp used. Either 'linear' (the default) or 'raised-cosine'.
        name
            Optionally, one can provide a name for the SoundStimulus. This is for instance useful
            when distinguishing A and B stimuli. It is used when the SoundStimulus sound is printed,
            written to a file, or when it is plotted.

        Examples
        --------
        >>> stim = SoundStimulus.generate(freq=1000, onramp_ms=10, offramp_ms=10)
        >>> stim.plot_waveform()  # doctest: +SKIP
        """

        # Generate signal
        samples = thebeat.helpers.synthesize_sound(
            duration_ms=duration_ms,
            fs=fs,
            freq=freq,
            n_channels=n_channels,
            amplitude=amplitude,
            oscillator=oscillator,
        )

        # Make ramps
        samples = thebeat.helpers.make_ramps(samples, fs, onramp_ms, offramp_ms, ramp_type)

        # Return class
        return cls(samples, fs, name)

    @classmethod
    def generate_white_noise(
        cls,
        duration_ms: int = 50,
        sigma: float = 1,
        fs: int = 48000,
        n_channels: int = 1,
        amplitude: float = 1.0,
        rng: np.random.Generator | None = None,
        name: str | None = None,
    ) -> SoundStimulus:
        """This class method returns a SoundStimulus object with white noise. They are simply random
        samples from a normal distribution with mean 0 and standard deviation ``sd``.

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
            Optionally, one can provide a name for the SoundStimulus. This is for instance useful
            when distinguishing A and B stimuli. It is used when the SoundStimulus sound is printed,
            written to a file,
            or when it is plotted.

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
        samples = amplitude * rng.normal(loc=0, scale=sigma, size=output_shape)

        samples = thebeat.helpers.normalize_audio(samples)

        # Return class
        return cls(samples=samples, fs=fs, name=name)

    @classmethod
    def from_note(
        cls,
        note_str: str,
        duration: int = 50,
        fs: int = 48000,
        amplitude: float = 1.0,
        oscillator: str = "sine",
        onramp_ms: int = 0,
        offramp_ms: int = 0,
        ramp: str = "linear",
        name: str | None = None,
    ) -> SoundStimulus:
        """Generate a :py:class:`SoundStimulus` object on the basis of a note as a string. For
        instance, a ``note_str`` of ``'G4'`` results in a generated SoundStimulus with a pitch
        frequency of 392 hertz.

        Parameters
        ----------
        note_str
            A note as a string. Can either be provided as ``'G'`` or ``'G4'``.
        duration
            The duration in milliseconds.
        fs
            The sampling frequency in hertz.
        amplitude
            Factor with which sound is amplified. Values between 0 and 1 result in sounds that are
            less loud, values higher than 1 in louder sounds.
        oscillator
            The oscillator used for generating the sound. Either 'sine' (the default), 'square' or
            'sawtooth'.
        onramp_ms
            The sound's 'attack' in milliseconds.
        offramp_ms
            The sound's 'decay' in milliseconds.
        ramp
            The type of on- and offramp_ms used. Either 'linear' (the default) or 'raised-cosine'.
        name
            Optionally, one can provide a name for the SoundStimulus. This is for instance useful
            when distinguishing A and B stimuli. It is used when the SoundStimulus sound is printed,
            written to a file, or when it is plotted.

        Examples
        --------
        >>> stim = SoundStimulus.from_note('G',duration=20)

        >>> stim = SoundStimulus.from_note('G4',onramp_ms=10, offramp_ms=10, ramp='raised-cosine')
        """

        if abjad is None:
            raise ImportError(
                "This function requires the abjad package. Install, for instance by typing "
                "`pip install abjad` or `pip install thebeat[music-notation]` into your terminal.\n"
                "For more details, see https://thebeat.readthedocs.io/en/latest/installation.html."
            )

        note_strings = re.split(r"([A-Z])([0-9]?)", note_str)
        note_strings = [string for string in note_strings if string != ""]

        # In Abjad 3.26 and lower, these values were properties; now they are member functions.
        # Abjad 2.28 requires at least Python 3.12; so this is a patch for backwards compatibility.
        # Remove once support for 3.11 gets dropped.
        def get_hertz(named_pitch):
            try:
                return named_pitch.hertz()
            except TypeError:
                return named_pitch.hertz

        if len(note_strings) == 1:
            freq = get_hertz(abjad.NamedPitch(note_strings[0], octave=4))
        elif len(note_strings) == 2:
            freq = get_hertz(abjad.NamedPitch(note_strings[0], octave=int(note_strings[1])))
        else:
            raise ValueError("Please provide a string of format 'G' or 'G4'.")

        return SoundStimulus.generate(
            freq=freq,
            fs=fs,
            duration_ms=duration,
            amplitude=amplitude,
            oscillator=oscillator,
            onramp_ms=onramp_ms,
            offramp_ms=offramp_ms,
            ramp_type=ramp,
            name=name,
        )

    @classmethod
    def from_parselmouth(cls, sound_object, name: str | None = None) -> SoundStimulus:
        """This class method generates a :py:class:`SoundStimulus` object from
        a :class:`parselmouth.Sound` object.

        Parameters
        ----------
        sound_object : :class:`parselmouth.Sound` object
            The to-be imported Parselmouth Sound object.
        name : str, optional
            Optionally, one can provide a name for the SoundStimulus. This is for instance useful
            when distinguishing A and B stimuli. It is used when the SoundStimulus sound is printed,
            written to a file, or when it is plotted.
        """
        if not sound_object.__class__.__name__ == "Sound":
            raise TypeError("Please provide a parselmouth.Sound object.")

        fs = sound_object.sampling_frequency
        # Parselmouth's sampling frequency is always a floating-point number, so we convert and warn if it was not a round number.
        if not fs.is_integer():
            warnings.warn("Sampling frequency was not a round number. It was rounded to the nearest integer.")
        fs = int(sound_object.sampling_frequency)

        samples = sound_object.values.T

        return cls(samples, fs, name)

    def change_amplitude(self, factor: float):
        """This method can be used to change the amplitude of the SoundStimulus sound. A factor
        between 0 and 1 decreases the amplitude; a factor larger than 1 increases the amplitude.

        Parameters
        ----------
        factor
            The factor by which the sound should be amplified.


        Examples
        --------
        >>> sound = SoundStimulus.generate()
        >>> sound.change_amplitude(factor=0.5)  # half as loud
        >>> sound.change_amplitude(factor=2)  # twice as loud
        """

        if not factor > 0:
            raise ValueError("Please provide a 'factor' larger than zero.")

        # set samples to new amplitude
        self.samples = self.samples * factor

    def merge(self, other: thebeat.core.SoundStimulus | list[thebeat.core.SoundStimulus]):
        """Merge this :py:class:`SoundStimulus` object with one or multiple other
        :py:class:`SoundStimulus` objects.

        Returns a new :py:class:`SoundStimulus` object.


        Parameters
        ----------
        other
            A :py:class:`SoundStimulus` object, or a list of :py:class:`SoundStimulus` objects.

        Returns
        -------
        object
            A :py:class:`SoundStimulus` object.
        """
        if isinstance(other, thebeat.SoundStimulus):
            return thebeat.utils.merge_soundstimuli([self, other])

        return thebeat.utils.merge_soundstimuli([self, *other])

    def play(self, loop: bool = False) -> None:
        """This method uses :func:`sounddevice.play` to play the :py:class:`SoundStimulus` sound.

        Parameters
        ----------
        loop
            If ``True``, the SoundStimulus object is played until the :py:meth:`SoundStimulus.stop`
            method is called.

        Examples
        --------
        >>> stim = SoundStimulus.generate()
        >>> stim.play()  # doctest: +SKIP
        """

        sd.play(self.samples, self.fs, loop=loop)
        sd.wait()

    def stop(self) -> None:
        """Stop playing the :py:class:`SoundStimulus` sound. Calls :func:`sounddevice.stop`.

        Examples
        --------
        >>> import time  # doctest: +SKIP
        >>> stim = SoundStimulus.generate()  # doctest: +SKIP
        >>> stim.play()  # doctest: +SKIP
        >>> time.sleep(secs=1)  # doctest: +SKIP
        >>> stim.stop()  # doctest: +SKIP
        """

        sd.stop()

    def plot_waveform(self, **kwargs) -> tuple[plt.Figure, plt.Axes]:
        """This method plots the waveform of the :py:class:`SoundStimulus` sound.

        Parameters
        ----------
        **kwargs
            Additional parameters (e.g. 'title', 'dpi' are passed to
            :py:func:`thebeat.helpers.plot_waveform`).

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

        if self.name:
            kwargs.setdefault("title", self.name)

        fig, ax = thebeat.helpers.plot_waveform(
            samples=self.samples, fs=self.fs, n_channels=self.n_channels, **kwargs
        )

        return fig, ax

    # Stats
    @property
    def duration_s(self) -> np.float64:
        """The duration of the SoundStimulus sound in seconds."""
        return np.float64(self.samples.shape[0] / self.fs)

    @property
    def duration_ms(self) -> np.float64:
        """The duration of the SoundStimulus sound in milliseconds."""
        return np.float64(self.samples.shape[0] / self.fs * 1000)

    def write_wav(self, filepath: str | os.PathLike, dtype: str | np.dtype = np.int16) -> None:
        """Save the :py:class:`SoundStimulus` sound to disk as a wave file.

        Parameters
        ----------
        filepath
            The output destination for the .wav file. Either pass e.g. a ``Path`` object, or a
            string. Of course be aware of OS-specific filepath conventions.
        dtype
            The data type of the samples. Defaults to ``np.int16``, meaning that the
            samples are saved as 16-bit integers.

        Examples
        --------
        >>> stim = SoundStimulus.generate()
        >>> stim.write_wav('my_stimulus.wav')  # doctest: +SKIP
        """

        thebeat.helpers.write_wav(
            samples=self.samples, fs=self.fs, filepath=filepath, dtype=dtype, metronome=False
        )
