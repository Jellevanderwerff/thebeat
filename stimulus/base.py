import numpy as np
from scipy.signal import resample, square, hanning
from scipy.io import wavfile
import sounddevice as sd
import matplotlib.pyplot as plt
from typing import Union, Iterator
import subprocess
from mingus.extra import lilypond
from mingus.containers import Track, Bar, Note
import os
from os import PathLike
import skimage
import parselmouth
from collections.abc import Iterable
from pathlib import Path
import re


class Stimulus:
    """
    Stimulus class that holds a Numpy 1-D array of sound that is either generated, or read from a .wav file.

    Attributes
    ----------
    fs : int
        Sampling frequency of the sound.
    samples : Numpy 1-D array (float32)
        Contains the samples of the sound.
    dtype : Numpy data type object
        Contains the Numpy data type object. Hard-coded as np.float32. If a read .wav file has a different dtype,
        the samples will be converted to np.float32.
    stim : Numpy 1-D array
        Contains the stimulus samples. In most cases this will be the same as Stimulus.samples.

    Class methods
    -------------
    from_wav(wav_filepath, new_fs=None)
        Read a .wav file from disk.
    generate(freq=440, fs=44100, duration=50, amplitude=0.8, osc='sine', onramp=10, offramp=10)
        Generate a sound using a sine or square oscillator.

    Methods
    -------
    change_amplitude(factor)
        Change the amplitude of the Stimulus by 'factor'. E.g. 2 will be twice as loud, 0.5 will be half as loud.
    play(loop=False)
        Play the Stimulus using sounddevice.
    stop()
        Stop sounddevice playback.
    plot()
        Plot the Stimulus's waveform using matplotlib.
    get_duration()
        Get the duration of the Stimulus in seconds.
    write_wav(out_path)
        Write the Stimulus to disk as a .wav file.


    """

    def __init__(self,
                 samples: np.ndarray,
                 fs: int,
                 name: str = None,
                 known_pitch: int = None):
        # check number of dimensions
        if samples.ndim == 1:
            n_channels = 1
        elif samples.ndim == 2:
            n_channels = 2
        else:
            raise ValueError("Wrong number of dimensions in given samples. Can only be 1 (mono) or 2 (stereo).")

        # Save all attributes
        self.samples = samples
        self.fs = fs
        self.dtype = samples.dtype
        self.pitch = known_pitch
        self.n_channels = n_channels
        self.stim_name = name

    def __str__(self):
        if self.stim_name:
            name = self.stim_name
        else:
            name = "Not provided"

        if self.pitch is not None:
            pitch = f"{self.pitch} Hz"
        else:
            pitch = "Unknown"

        return f"Object of type Stimulus.\n\nStimulus name: {name}\nStimulus duration: {self.duration_ms} milliseconds." \
               f"\nPitch frequency: {pitch}"

    @classmethod
    def from_wav(cls,
                 filepath: Union[PathLike, str],
                 name=None,
                 new_fs: int = None,
                 known_pitch: float = None,
                 extract_pitch: bool = False):
        """

        This method loads a stimulus from a PCM .wav file, and reads in the samples.
        It additionally converts .wav files with dtype int16 or int32 to float32.

        Parameters
        ----------
        filepath: str or PathLike object
        extract_pitch: bool
        new_fs : int
            If resampling is required, you can provide the target sampling frequency

        Returns
        ----------
        Does not return anything.
        """

        # Read in the sampling frequency and all the samples from the wav file
        samples, fs, pitch = _read_wavfile(filepath, new_fs, known_pitch, extract_pitch)

        return cls(samples, fs, name, pitch)

    @classmethod
    def generate(cls,
                 name=None,
                 freq=440,
                 fs=44100,
                 duration=50,
                 amplitude=1.0,
                 osc='sine',
                 onramp=0,
                 offramp=0,
                 ramp='linear'):
        """
        """
        t = duration / 1000
        samples = np.linspace(0, t, int(fs * t), endpoint=False, dtype=np.float32)
        if osc == 'sine':
            signal = amplitude * np.sin(2 * np.pi * freq * samples)
        elif osc == 'square':
            signal = amplitude * square(2 * np.pi * freq * samples)
        else:
            raise ValueError("Choose existing oscillator (for now only 'sin')")

        signal, fs = _make_ramps(signal, fs, onramp, offramp, ramp)

        # Return class, and save the used frequency
        return cls(signal, fs, name, known_pitch=freq)

    @classmethod
    def rest(cls,
             name=None,
             duration=50,
             fs=44100):
        samples = np.zeros(duration // (1000 * fs), dtype='float32')

        return cls(samples, fs, name)

    @classmethod
    def from_parselmouth(cls, snd_obj, stim_name=None, extract_pitch=False):
        if not snd_obj.__class__.__name__ == "Sound":
            raise ValueError("Please provide a parselmouth.Sound object.")

        fs = snd_obj.sampling_frequency

        if snd_obj.samples.ndim == 1:
            samples = snd_obj.values[0]
        elif snd_obj.samples.ndim == 2:
            samples = snd_obj.values
        else:
            raise ValueError("Incorrect number of dimensions in samples. Should be 1 (mono) or 2 (stereo).")

        if extract_pitch is True:
            pitch = _extract_pitch(samples, fs)
        else:
            pitch = None

        return cls(samples, fs, stim_name, pitch)

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
            if self.stim_name:
                title = f"Waveform of {self.stim_name}"
            else:
                title = "Waveform of Stimulus"

        _plot_waveform(self.samples, self.fs, self.n_channels, title)

    # Stats
    @property
    def duration_s(self) -> float:
        return self.samples.shape[0] / self.fs

    @property
    def duration_ms(self) -> float:
        return self.samples.shape[0] / self.fs * 1000

    # Out
    def write_wav(self, out_path: Union[str, PathLike]):
        """
        Writes audio to disk.
        """
        out_path = str(out_path)
        if out_path.endswith('.wav'):
            path, filename = os.path.split(out_path)
        elif os.path.isdir(out_path):
            path = out_path
            filename = f"{self.stim_name}.wav"
        else:
            raise ValueError("Wrong out_path specified. Please provide a directory or a complete filepath.")

        wavfile.write(filename=os.path.join(path, filename), rate=self.fs, data=self.samples)


class Stimuli:
    """Class that contains multiple stimuli"""

    def __init__(self,
                 stim_objects: Iterable[Stimulus]):

        stim_objects = list(stim_objects)

        # Check consistency
        all_fs = [snd.fs for snd in stim_objects]
        all_dtypes = [snd.dtype for snd in stim_objects]

        # Check whether fs's are the same across the list
        if not all(x == all_fs[0] for x in all_fs):
            raise ValueError("The Stimulus objects in the passed list have different sampling frequencies!")
        else:
            fs = all_fs[0]

        # Check whether dtypes are the same
        if not all(x == all_dtypes[0] for x in all_dtypes):
            raise ValueError("The Stimulus objects in the passed list have different dtypes!")
        else:
            dtype = all_dtypes[0]

        # Check equal number of channels
        if not all(x.n_channels == stim_objects[0].n_channels for x in stim_objects):
            raise ValueError("The Stimulus objects in the passed list have differing number of channels!")
        else:
            n_channels = stim_objects[0].n_channels

        # Make list of stimulus samples Numpy arrays.
        samples = [stim.samples for stim in stim_objects]

        # Make array of pitch frequencies
        pitch = np.array([x.pitch for x in stim_objects])

        # Save attributes
        self.samples = samples
        self.fs = fs
        self.dtype = dtype
        self.pitch = pitch
        self.n_channels = n_channels
        self.n = len(samples)
        self.stim_names = [stim.stim_name for stim in stim_objects]

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i != len(self.samples):
            stim_obj = Stimulus(self.samples[self.i], self.fs, self.pitch[self.i])
            self.i += 1
            return stim_obj

        else:
            raise StopIteration

    def __len__(self):
        return self.n

    def __str__(self):
        return f"Object of type Stimuli.\nNumber of stimuli: {self.n}\nStimulus names: " \
               f"{self.stim_names}\nSampling frequency: {self.fs} Hz\nPitch frequencies: {self.pitch} " \
               f"Hz\nNumber of channels: {self.n_channels}"

    @classmethod
    def from_stim(cls, stim: Stimulus, repeats: int):

        return cls([Stimulus(stim.samples, stim.fs, stim.stim_name, stim.pitch)] * repeats)

    @classmethod
    def from_stims(cls,
                   stims: Iterable[Stimulus],
                   repeats: int = 1):

        stims = list(stims)

        out_stims = stims * int(repeats)

        return cls(out_stims)

    @classmethod
    def from_dir(cls,
                 dir_path: Union[str, PathLike],
                 new_fs: int = None,
                 known_pitches: Iterable = None,
                 extract_pitch=False):

        dir_path = Path(dir_path)
        filenames = os.listdir(dir_path)

        # Pitch
        if known_pitches:
            known_pitches = list(known_pitches)
            if len(known_pitches) != len(filenames):
                raise ValueError("Please provide an equal number of pitches as .wav files.")
        else:
            known_pitches = [None] * len(filenames)

        # Generate list of Stimulus objects
        stim_objects = []

        for i, filename in enumerate(filenames):
            if not filename.endswith('.wav'):
                raise ValueError('Directory should only contain .wav files.')

            wav_filepath = os.path.join(dir_path, filename)
            samples, fs, pitch = _read_wavfile(wav_filepath, new_fs, known_pitches[i], extract_pitch)

            stim_objects.append(Stimulus(samples, fs, known_pitch=pitch))

        return cls(stim_objects)

    @classmethod
    def from_notes(cls, notes_str, event_duration=50, onramp=0, offramp=0, ramp='linear',
                   stim_names: Iterable[str] = None):
        """
        Get stimulus objects on the basis of a provided string of notes.
        For instance: 'CDEC' returns a list of four Stimulus objects.
        Alternatively, one can use 'C4D4E4C4'. In place of
        silences one can use an 'X'.

        """
        notes = re.findall(r"[A-Z][0-9]?", notes_str)

        if stim_names is None:
            stim_names = notes
        else:
            stim_names = list(stim_names)
            if len(stim_names) != len(notes):
                raise ValueError("Please provide an equal number of stim_names as the number of notes.")
            else:
                stim_names = stim_names

        freqs = []

        for note in notes:
            if len(note) > 1:
                note, num = tuple(note)
                freqs.append(Note(note, int(num)).to_hertz())
            else:
                if note == 'X':
                    freqs.append(None)
                else:
                    freqs.append(Note(note).to_hertz())

        stims = []

        freq_names = list(zip(freqs, stim_names))

        for freq, stim_name in freq_names:
            if freq is None:
                stims.append(Stimulus.rest(name=stim_name, duration=event_duration))
            else:
                stims.append(Stimulus.generate(freq=freq,
                                               duration=event_duration,
                                               onramp=onramp,
                                               offramp=offramp,
                                               ramp=ramp,
                                               name=stim_name))

        return cls(stims)

    def randomize(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        else:
            rng = rng

        zipped = list(zip(self.samples, self.stim_names, self.pitch))
        rng.shuffle(zipped)

        samples, stim_names, pitch = zip(*zipped)
        self.samples, self.stim_names, self.pitch = list(samples), np.array(stim_names), np.array(pitch)

    def write_wavs(self, path: Union[str, PathLike], filenames: list[str] = None):

        if filenames is None:
            if all(name is None for name in self.stim_names):
                filenames = [f"{str(i)}.wav" for i in range(1, len(self.samples) + 1)]
            else:
                filenames = []
                for i, name in enumerate(self.stim_names):
                    filenames.append(f"{i + 1}-{name}.wav")
        else:
            filenames = filenames

        for samples, filename in zip(self.samples, filenames):
            wavfile.write(os.path.join(path, filename), self.fs, samples)


class BaseSequence:
    """Base Sequence class that holds the most basic methods and attributes. """

    def __init__(self, iois, metrical=False, played=None):
        self.iois = iois
        # If metrical=True, that means there's an additional IOI for the final event.
        self.metrical = metrical
        self.played = played

        # Deal with 'played'
        if played is None:
            self.played = [True] * len(self.onsets)
        elif len(played) == len(self.onsets):
            self.played = played
        else:
            raise ValueError("The 'played' list should contain an equal number of "
                             "booleans as the number of onsets.")

        if any(ioi < 0 for ioi in iois):
            raise ValueError("IOIs cannot be negative.")
        else:
            self.iois = np.array(iois, dtype=np.float32)

    @property
    def onsets(self):
        """Get the event onsets. These is the cumulative sum of Sequence.iois, with 0 additionally prepended.
        """

        if self.metrical is True:
            return np.cumsum(np.append(0, self.iois[:-1]), dtype=np.float32)
        else:
            return np.cumsum(np.append(0, self.iois), dtype=np.float32)


class Sequence(BaseSequence):
    """
    Sequence class that holds a sequence of inter-onset intervals (IOIs) and stimulus onsets.
    Additionally has class methods that can be used for generating a new sequence.

    Attributes
    ----------

    iois : Numpy 1-D array
        A list of the sequence's inter-onset intervals.

    Class methods
    -------

    generate_random_normal(n, mu, sigma, rng=None)
        Generate a random sequence using the normal distribution.
    generate_random_uniform(n, a, b, rng=None)
        Generate a random sequence using a uniform distribution.
    generate_random_poisson(n, lam, rng=None)
        Generate a random sequence using a Poisson distribution.
    generate_random_exponential(n, lam, rng=None)
        Generate a random sequence using an exponential distribution.
    generate_isochronous(n, ioi)
        Generate an isochronous sequence using an exponential distribution.

    Methods
    -------


    """

    def __init__(self, iois, metrical=False):

        # Call super init method
        BaseSequence.__init__(self, iois=iois, metrical=metrical, played=None)

    def __str__(self):
        if self.metrical:
            return f"Object of type Sequence (metrical version):\n{len(self.onsets)} events\nIOIs: {self.iois}\nOnsets:{self.onsets}\nOnsets played: {self.played}"
        else:
            return f"Object of type Sequence (non-metrical version):\n{len(self.onsets)} events\nIOIs: {self.iois}\nOnsets:{self.onsets}\nOnsets played: {self.played} "

    def __add__(self, other):
        return join_sequences([self, other])

    @classmethod
    def generate_random_normal(cls, n: int, mu: int, sigma: int, rng=None, metrical=False):
        """

        Class method that generates a sequence of random inter-onset intervals based on the normal distribution.
        Note that there will be n-1 IOIs in a sequence.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        mu : int
            The mean of the normal distribution.
        sigma : int
            The standard deviation of the normal distribution.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)
        metrical : boolean
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
            grid)

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        if metrical:
            n_iois = n
        elif not metrical:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.normal(loc=mu, scale=sigma, size=n_iois))

        return cls(round_iois, metrical=metrical)

    @classmethod
    def generate_random_uniform(cls, n: int, a: int, b: int, rng=None, metrical=False):
        """

        Class method that generates a sequence of random inter-onset intervals based on a uniform distribution.
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        a : int
            The left bound of the uniform distribution.
        b : int
            The right bound of the normal distribution.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)
        metrical : boolean
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
            grid)

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        if metrical:
            n_iois = n
        elif not metrical:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.uniform(low=a, high=b, size=n_iois))

        return cls(round_iois, metrical=metrical)

    @classmethod
    def generate_random_poisson(cls, n: int, lam: int, rng=None, metrical=False):
        """

        Class method that generates a sequence of random inter-onset intervals based on a Poisson distribution.
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        lam : int
            The desired value for lambda.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)
        metrical : boolean
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
                grid)

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        if metrical:
            n_iois = n
        elif not metrical:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.poisson(lam=lam, size=n_iois))

        return cls(round_iois, metrical=metrical)

    @classmethod
    def generate_random_exponential(cls, n: int, lam: int, rng=None, metrical=False):
        """

        Class method that generates a sequence of random inter-onset intervals based on an exponential distribution.
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        lam : int
           The desired value for lambda.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)
        metrical : boolean
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
            grid)

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        if metrical:
            n_iois = n
        elif not metrical:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.exponential(scale=lam, size=n_iois))

        return cls(round_iois, metrical=metrical)

    @classmethod
    def generate_isochronous(cls, n: int, ioi: int, metrical=False):
        """

        Class method that generates a sequence of isochronous inter-onset intervals.
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.


        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        ioi : int
            The inter-onset interval to be used between all events.
        metrical : boolean
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
            grid)

        Returns
        -------
        Returns an object of class Sequence.

        """

        if metrical:
            n_iois = n
        elif not metrical:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        return cls(np.round([ioi] * n_iois), metrical=metrical)

    # Manipulation methods
    def change_tempo(self, factor):
        """
        Change the tempo of the sequence.
        A factor of 1 or bigger increases the tempo (resulting in smaller IOIs).
        A factor between 0 and 1 decreases the tempo (resulting in larger IOIs).
        """
        if factor > 0:
            self.iois /= factor
        else:
            raise ValueError("Please provide a factor larger than 0.")

    def change_tempo_linearly(self, total_change):
        """
        This function can be used for creating a ritardando or accelerando effect.
        You provide the total change over the entire sequence.
        So, total change of 2 results in a final IOI that is
        twice as short as the first IOI.
        """
        self.iois /= np.linspace(1, total_change, self.iois.size)

    # Descriptive methods
    def get_stats(self):
        return {
            'ioi_mean': np.mean(self.iois),
            'ioi_median': np.median(self.iois),
            'ioi_q1': np.quantile(self.iois, 0.25),
            'ioi_q2': np.quantile(self.iois, 0.5),
            'ioi_q3': np.quantile(self.iois, 0.75),
            'ioi_sd': np.std(self.iois),
            'ioi_min': np.min(self.iois),
            'ioi_max': np.max(self.iois)
        }


class StimTrial(BaseSequence):
    """
    StimSequence class which inherits only the most basic functions from BaseSequence
    """

    def __init__(self,
                 stimuli: Stimuli,
                 seq_obj,
                 name: str = None):

        # Type checking for stimuli
        if isinstance(stimuli, Stimulus):
            raise ValueError("Please provide a Stimuli object instead of a Stimulus object as the first argument.")
        elif not isinstance(stimuli, Stimuli):
            raise ValueError("Please provide a Stimuli object as the first argument.")
        else:
            pass

        # Type checking for sequence
        if not seq_obj.__class__.__name__ == "Sequence":
            raise ValueError("Please provide a Sequence or Rhythm object as the second argument.")

        # Save whether passed sequence is metrical or not
        self.metrical = seq_obj.metrical

        # Save fs, dtype, note values, and pitch
        self.fs = stimuli.fs
        self.dtype = stimuli.dtype
        self.n_channels = stimuli.n_channels
        self.pitch = stimuli.pitch
        self.name = name
        self.stim_names = stimuli.stim_names

        # Initialize Sequence class
        BaseSequence.__init__(self, seq_obj.iois, metrical=seq_obj.metrical)

        # Make sound which saves the samples to self.samples
        self.samples = self._make_sound(stimuli, self.onsets)

        # Then save list of Stimulus objects for later use
        self.stim = stimuli

    def __str__(self):

        if self.name:
            name = self.name
        else:
            name = "Not provided"

        if not all(pitch is None for pitch in self.pitch):
            pitch = f"{self.pitch} Hz"
        else:
            pitch = "Unknown"

        if self.stim_names:
            stim_names = self.stim_names
        else:
            stim_names = "Not provided"

        if self.metrical:
            return f"""
Object of type StimTrial (metrical version):
StimTrial name: {name}
{len(self.onsets)} events
IOIs: {self.iois}
Onsets:{self.onsets}
Stimulus names: {stim_names}
Pitch frequencies: {pitch}
Stimulus played: {self.played}
            """
        else:
            return f"""
Object of type StimTrial (non-metrical version):
StimTrial name: {name}
{len(self.onsets)} events
IOIs: {self.iois}
Onsets:{self.onsets}
Stimulus names: {stim_names}
Pitch frequencies: {pitch}
Stimulus played: {self.played}
            """

    @property
    def mean_ioi(self):
        return np.mean(self.iois)

    def _make_sound(self, stimuli, onsets):
        # Check for overlap
        for i in range(len(onsets)):
            stim_duration = stimuli.samples[i].shape[0] / self.fs * 1000
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
            array_length = int((onsets[-1] + self.iois[-1]) / 1000 * self.fs)
        elif not self.metrical:
            array_length = int((onsets[-1] / 1000 * self.fs) + stimuli.samples[-1].shape[0])
        else:
            raise ValueError("Error during calculation of array_length")

        if self.n_channels == 1:
            samples = np.zeros(array_length, dtype=self.dtype)
        else:
            samples = np.zeros((array_length, 2), dtype=self.dtype)

        samples_with_onsets_played = list(zip(stimuli.samples, onsets))

        for stimulus, onset in samples_with_onsets_played:
            start_pos = int(onset * self.fs / 1000)
            end_pos = int(start_pos + stimulus.shape[0])
            if self.n_channels == 1:
                samples[start_pos:end_pos] = stimulus
            elif self.n_channels == 2:
                samples[start_pos:end_pos, :2] = stimulus

        # return sound
        return samples

    def play(self, loop=False, metronome=False, metronome_amplitude=1):
        _play_samples(self.samples, self.fs, self.mean_ioi, loop, metronome, metronome_amplitude)

    def plot_music(self, filepath=None, key='C', print_staff=True):

        if self.pitch is None:
            raise ValueError("The pitches of the stimuli are unknown. Either"
                             "import using the extract_pitch=True flag, or"
                             "provide the values yourself as StimuliSequence.pitch")

        # create initial bar
        t = Track()
        b = Bar(key=key, meter=self.time_sig)

        # keep track of the index of the note_value
        note_i = 0

        values_freqs_played = list(zip(self.note_values, self.pitch, self.played))

        for note_value, freq, played in values_freqs_played:
            if played is True:
                note = Note()
                note.from_hertz(freq)
                b.place_notes(note.name, self.note_values[note_i])
            elif played is False:
                b.place_rest(self.note_values[note_i])

            # if bar is full, create new bar and add bar to track
            if b.current_beat == b.length:
                t.add_bar(b)
                b = Bar(key=key, meter=self.time_sig)

            note_i += 1

        # If final bar was not full yet, add a rest for the remaining duration
        if b.current_beat % 1 != 0:
            rest_value = 1 / b.space_left()
            if round(rest_value) != rest_value:
                raise ValueError("The rhythm could not be plotted. Most likely because the IOIs cannot "
                                 "be (easily) captured in musical notation. This for instance happens when "
                                 "using one of the tempo manipulation methods.")

            b.place_rest(rest_value)
            t.add_bar(b)

        # Call internal plot method to plot the track
        _plot_lp(t, filepath, print_staff)

    def plot_waveform(self, title=None):
        if title:
            title = title
        else:
            if self.name:
                title = f"Waveform of {self.name}"
            else:
                title = "Waveform of StimSequence"

        _plot_waveform(self.samples, self.fs, self.n_channels, title)

    def write_wav(self, out_path='.',
                  metronome=False,
                  metronome_amplitude=1):
        """
        Writes audio to disk.
        """

        _write_wav(self.samples, self.fs, out_path, self.name, metronome, self.mean_ioi, metronome_amplitude)


def _extract_pitch(samples, fs) -> float:
    pm_snd_obj = parselmouth.Sound(values=samples, sampling_frequency=fs)
    pitch = pm_snd_obj.to_pitch()
    mean_pitch = float(parselmouth.praat.call(pitch, "Get mean...", 0, 0.0, 'Hertz'))
    return mean_pitch


def _get_sound_with_metronome(samples, fs, metronome_ioi, metronome_amplitude):
    sound_samples = samples
    duration = sound_samples.shape[0] / fs * 1000

    n_metronome_clicks = int(duration // metronome_ioi)  # We want all the metronome clicks that fit in the seq.
    onsets = np.concatenate((np.array([0]), np.cumsum([metronome_ioi] * (n_metronome_clicks - 1))))

    metronome_fs, metronome_samples = wavfile.read('metronome.wav')

    # resample if metronome sound has different sampling frequency
    if metronome_fs != fs:
        resample_factor = float(fs) / float(metronome_fs)
        resampled = resample(metronome_samples, int(len(metronome_samples) * resample_factor))
        metronome_samples = resampled

    # change amplitude if necessary
    metronome_samples *= metronome_amplitude

    for onset in onsets:
        start_pos = int(onset * fs / 1000)
        end_pos = int(start_pos + metronome_samples.size)
        new_samples = sound_samples[start_pos:end_pos] + metronome_samples
        sound_samples[start_pos:end_pos] = new_samples  # we add the metronome sound to the existing sound

    return sound_samples


def join_sequences(iterator):
    """
    This function can join metrical Sequence objects.
    """

    # Check whether iterable was passed
    if not hasattr(iterator, '__iter__'):
        raise ValueError("Please pass this function a list or other iterable object.")

    # Check whether all the objects are of the same type
    if not all(isinstance(x, Sequence) for x in iterator):
        raise ValueError("This function can only join multiple Sequence objects.")

    # Sequence objects need to be metrical:
    if not all(x.metrical for x in iterator):
        raise ValueError("Only metrical Sequence objects can be joined. This is intentional.")

    iois = [sequence.iois for sequence in iterator]
    iois = np.concatenate(iois)

    return Sequence(iois, metrical=True)


def _make_ramps(signal, fs, onramp, offramp, ramp):
    # Create onramp
    if onramp > 0:
        onramp_samples_len = int(onramp / 1000 * fs)
        end_point = onramp_samples_len

        if ramp == 'linear':
            onramp_amps = np.linspace(0, 1, onramp_samples_len)

        elif ramp == 'raised-cosine':
            hanning_complete = hanning(onramp_samples_len * 2)
            onramp_amps = hanning_complete[:(hanning_complete.shape[0] // 2)]  # only first half of Hanning window

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
            hanning_complete = hanning(offramp_samples_len * 2)
            offramp_amps = hanning_complete[hanning_complete.shape[0] // 2:]

        signal[start_point:] *= offramp_amps

    elif offramp < 0:
        raise ValueError("Offramp cannot be negative")
    elif offramp == 0:
        pass
    else:
        raise ValueError("Wrong value supplied to offramp argument.")

    return signal, fs


def _play_samples(samples, fs, mean_ioi, loop, metronome, metronome_amplitude):
    if metronome is True:
        samples = _get_sound_with_metronome(samples, fs, mean_ioi,
                                            metronome_amplitude=metronome_amplitude)
    else:
        samples = samples

    sd.play(samples, fs, loop=loop)
    sd.wait()


def _plot_lp(t, filepath, print_staff: bool):
    """
    Internal method for plotting a mingus Track object via lilypond.
    """
    # This is the same each time:
    if filepath:
        location, filename = os.path.split(filepath)
        if location == '':
            location = '.'
    else:
        location = '.'
        filename = 'rhythm.png'

    # make lilypond string
    if print_staff is True:
        lp = '\\version "2.10.33"\n' + lilypond.from_Track(t) + '\n\paper {\nindent = 0\mm\nline-width = ' \
                                                                '110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ' \
                                                                '""\noddFooterMarkup = ""\nevenFooterMarkup = ""\n} '
    elif print_staff is False:
        lp = '\\version "2.10.33"\n' + '{ \stopStaff \override Staff.Clef.color = #white' + lilypond.from_Track(t)[
                                                                                            1:] + '\n\paper {\nindent = 0\mm\nline-width = ' \
                                                                                                  '110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ' \
                                                                                                  '""\noddFooterMarkup = ""\nevenFooterMarkup = ""\n} '
    else:
        raise ValueError("Wrong value specified for print_staff.")

    # write lilypond string to file
    with open(os.path.join(location, filename[:-4] + '.ly'), 'w') as file:
        file.write(lp)

    # run subprocess
    if filename.endswith('.eps'):
        command = f'lilypond -dbackend=eps --silent -dresolution=600 --eps -o {filename[:-4]} {filename[:-4] + ".ly"}'
        to_be_removed = ['.ly']
    elif filename.endswith('.png'):
        command = f'lilypond -dbackend=eps --silent -dresolution=600 --png -o {filename[:-4]} {filename[:-4] + ".ly"}'
        to_be_removed = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi', '.ly']
    else:
        raise ValueError("Can only export .png or .eps files.")

    p = subprocess.Popen(command, shell=True, cwd=location).wait()

    image = skimage.img_as_float(skimage.io.imread(filename))

    # Select all pixels almost equal to white
    # (almost, because there are some edge effects in jpegs
    # so the boundaries may not be exactly white)
    white = np.array([1, 1, 1])
    mask = np.abs(image - white).sum(axis=2) < 0.05

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    out = image[top_left[0]:bottom_right[0],
          top_left[1]:bottom_right[1]]

    # show plot
    if not filepath:
        plt.imshow(out)
        plt.axis('off')
        plt.show()
    elif filename.endswith('.png'):
        plt.imshow(out)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
    else:
        pass

    # remove files
    if filepath:
        filenames = [filename[:-4] + x for x in to_be_removed]
    else:
        to_be_removed = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi', '.ly', '.png']
        filenames = ['rhythm' + x for x in to_be_removed]

    for file in filenames:
        os.remove(os.path.join(location, file))


def _plot_waveform(samples, fs, n_channels, title):
    plt.clf()
    frames = np.arange(samples.shape[0])
    if n_channels == 1:
        alph = 1
    elif n_channels == 2:
        alph = 0.5
    else:
        raise ValueError("Unexpected number of channels.")

    plt.plot(frames, samples, alpha=alph)
    if n_channels == 2:
        plt.legend(["Left channel", "Right channel"], loc=0, frameon=True)
    plt.ylim([-1, 1])
    plt.ylabel("Amplitude")
    plt.xticks(ticks=[0, samples.shape[0]],
               labels=[0, int(samples.size / fs * 1000)])
    plt.xlabel("Time (ms)")
    plt.title(title)
    plt.show()


def _read_wavfile(filepath: Union[str, PathLike],
                  new_fs: int,
                  known_pitch: float = None,
                  extract_pitch: bool = False):
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

    # Extract pitch if necessary
    if extract_pitch is True:
        pitch = _extract_pitch(samples, fs)
    elif known_pitch:
        pitch = known_pitch
    else:
        pitch = None

    return samples, fs, pitch


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


def _write_wav(samples, fs, out_path, name, metronome, metronome_ioi, metronome_amplitude):
    if metronome is True:
        samples = _get_sound_with_metronome(samples, fs, metronome_ioi, metronome_amplitude)
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
            filename = f"stim_sequence.wav"

    else:
        raise ValueError("Wrong out_path specified. Please provide a directory or a complete filepath.")

    wavfile.write(filename=os.path.join(path, filename), rate=fs, data=samples)
