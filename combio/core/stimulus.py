import re
import os
from pathlib import Path
from typing import Union, Iterable
import numpy as np
import parselmouth
from mingus.containers import Note
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import resample
from .helpers import _plot_waveform


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
        self.name = name

    def __str__(self):
        if self.name:
            name = self.name
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
                 filepath: Union[os.PathLike, str],
                 name=None,
                 new_fs: int = None,
                 known_pitch: float = None,
                 extract_pitch: bool = False):
        """

        This method loads a stimulus from a PCM .wav file, and reads in the samples.
        It additionally converts .wav files with dtype int16 or int32 to float32.

        Parameters
        ----------
        name
        known_pitch
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
            signal = amplitude * np.square(2 * np.pi * freq * samples)
        else:
            raise ValueError("Choose existing oscillator (for now only 'sine' or 'square')")

        signal, fs = _make_ramps(signal, fs, onramp, offramp, ramp)

        # Return class, and save the used frequency
        return cls(signal, fs, name, known_pitch=freq)

    @classmethod
    def from_note(cls, note_str, event_duration=50, onramp=0, offramp=0, ramp='linear',
                  name: str = None):
        """
        Get stimulus objects on the basis of a provided string of notes.
        For instance: 'CDEC' returns a list of four Stimulus objects.
        Alternatively, one can use 'C4D4E4C4'. In place of
        silences one can use an 'X'.

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

        return Stimulus.generate(name, freq=freq, duration=event_duration, onramp=onramp, offramp=offramp, ramp=ramp)

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
            if self.name:
                title = f"Waveform of {self.name}"
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
    def write_wav(self, out_path: Union[str, os.PathLike]):
        """
        Writes audio to disk.
        """
        out_path = str(out_path)
        if out_path.endswith('.wav'):
            path, filename = os.path.split(out_path)
        elif os.path.isdir(out_path):
            path = out_path
            filename = f"{self.name}.wav"
        else:
            raise ValueError("Wrong out_path specified. Please provide a directory or a complete filepath.")

        wavfile.write(filename=os.path.join(path, filename), rate=self.fs, data=self.samples)


class Stimuli:
    """Class that contains multiple stimuli"""

    def __init__(self,
                 stim_objects: Iterable[Stimulus]):

        stim_objects = [stim for stim in stim_objects]

        # Check consistency
        all_fs = [snd.fs for snd in stim_objects if snd is not None]
        all_dtypes = [snd.dtype for snd in stim_objects if snd is not None]
        all_n_channels = [snd.n_channels for snd in stim_objects if snd is not None]

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
        if not all(channels == all_n_channels[0] for channels in all_n_channels):
            raise ValueError("The Stimulus objects in the passed list have differing number of channels!")
        else:
            n_channels = all_n_channels[0]

        # Make list of stimulus samples Numpy arrays and array of pitch frequencies.
        samples = []
        pitch = []

        for stim in stim_objects:
            if stim is None:
                samples.append(None)
                pitch.append(None)
            else:
                samples.append(stim.samples)
                pitch.append(stim.pitch)

        # Convert to array
        pitch = np.array(pitch)

        # Save attributes
        self.samples = samples
        self.fs = fs
        self.dtype = dtype
        self.pitch = pitch
        self.n_channels = n_channels
        self.n = len(samples)
        self.names = []
        for stim in stim_objects:
            if stim is None:
                self.names.append(None)
            else:
                self.names.append(stim.name)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i != len(self.samples):
            if self.samples[self.i] is not None:
                stim_obj = Stimulus(self.samples[self.i], self.fs, self.names[self.i], self.pitch[self.i])
                self.i += 1
                return stim_obj
            else:
                self.i += 1
                return None

        else:
            raise StopIteration

    def __len__(self):
        return self.n

    def __str__(self):
        return f"Object of type Stimuli.\nNumber of stimuli: {self.n}\nStimulus names: " \
               f"{self.names}\nSampling frequency: {self.fs} Hz\nPitch frequencies: {self.pitch} " \
               f"Hz\nNumber of channels: {self.n_channels}"

    @classmethod
    def from_stim(cls, stim: Stimulus, repeats: int):
        # todo Consider removing this one.

        return cls([Stimulus(stim.samples, stim.fs, stim.name, stim.pitch)] * repeats)

    @classmethod
    def from_stims(cls,
                   stims: Iterable[Stimulus],
                   repeats: int = 1):

        stims = list(stims)

        out_stims = stims * int(repeats)

        return cls(out_stims)

    @classmethod
    def from_dir(cls,
                 dir_path: Union[str, os.PathLike],
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
                raise ValueError("Please provide an equal number of names as the number of notes.")
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

        zipped = list(zip(self.samples, self.names, self.pitch))
        rng.shuffle(zipped)

        samples, stim_names, pitch = zip(*zipped)
        self.samples, self.names, self.pitch = list(samples), np.array(stim_names), np.array(pitch)

    def write_wavs(self, path: Union[str, os.PathLike], filenames: list[str] = None):

        if filenames is None:
            if all(name is None for name in self.names):
                filenames = [f"{str(i)}.wav" for i in range(1, len(self.samples) + 1)]
            else:
                filenames = []
                for i, name in enumerate(self.names):
                    filenames.append(f"{i + 1}-{name}.wav")
        else:
            filenames = filenames

        for samples, filename in zip(self.samples, filenames):
            wavfile.write(os.path.join(path, filename), self.fs, samples)


# noinspection PyArgumentList,PyTypeChecker
def _extract_pitch(samples, fs) -> float:
    pm_snd_obj = parselmouth.Sound(values=samples, sampling_frequency=fs)
    pitch = pm_snd_obj.to_pitch()
    mean_pitch = float(parselmouth.praat.call(pitch, "Get mean...", 0, 0.0, 'Hertz'))
    return mean_pitch


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
