import numpy as np
from scipy.signal import resample, square
from scipy.io import wavfile
import sounddevice as sd
from mingus.containers import Bar, Track
from mingus.extra import lilypond
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import subprocess


class Stimulus:
    """
    Stimulus class that holds a Numpy 1-D array of sound that is either generated, or read from a .wav file.
    Has some additional fun features.

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

    def __init__(self, samples, fs: int):
        self.dtype = np.float32
        self.stim = samples
        self.samples = samples
        self.fs = fs

    def __str__(self):
        return f"Object of type Stimulus.\nStimulus duration: {self.get_duration()} seconds."

    @classmethod
    def from_wav(cls, wav_filepath, new_fs: int = None):
        """

        This method loads a stimulus from a PCM .wav file, and reads in the samples.
        It additionally converts .wav files with dtype int16 or int32 to float32.

        Parameters
        ----------
        wav_filepath : str or Path object
            The path to the wave file
        new_fs : int
            If resampling is required, you can provide the target sampling frequency

        Returns
        ----------
        Does not return anything.
        """

        # Read in the sampling frequency and all the samples from the wav file
        file_fs, samples = wavfile.read(wav_filepath)

        if len(np.shape(samples)) > 1:
            print("Input file was stereo. Please convert to mono first.")

        # Change dtype so we always have float32
        if samples.dtype == 'int16':
            samples = samples.astype(np.float32) / 32768
        elif samples.dtype == 'int32':
            samples = samples.astype(np.float32) / 2147483648
        elif samples.dtype == 'float32':
            pass
        else:
            raise ValueError("Unknown dtype for wav file. 'int16', 'int32' and 'float32' are supported:'"
                   "https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html")

        if new_fs is None or new_fs == file_fs:
            fs = file_fs
        elif new_fs != file_fs:
            resample_factor = float(new_fs) / float(file_fs)
            resampled = resample(samples, int(len(samples) * resample_factor))
            samples = resampled
            fs = new_fs
        else:
            raise ValueError("Error while comparing old and new sampling frequencies.")

        return cls(samples, fs)

    @classmethod
    def generate(cls, freq=440, fs=44100, duration=50, amplitude=1.0, osc='sine', onramp=0, offramp=0):
        """

        Parameters
        ----------
        freq
        fs
        duration
        amplitude
        osc
        onramp
        offramp

        Returns
        -------

        """
        t = duration / 1000
        samples = np.linspace(0, t, int(fs*t), endpoint=False, dtype=np.float32)
        if osc == 'sine':
            signal = amplitude * np.sin(2 * np.pi * freq * samples)
        elif osc == 'square':
            signal = amplitude * square(2 * np.pi * freq * samples)
        else:
            raise ValueError("Choose existing oscillator (for now only 'sin')")
        # Create onramp
        if onramp > 0:
            onramp_amps = np.linspace(0, 1, int(onramp / 1000 * fs))
            signal[:len(onramp_amps)] *= onramp_amps
        elif onramp < 0:
            raise ValueError("Onramp cannot be negative")
        elif onramp == 0:
            pass

        # Create offramp
        if offramp > 0:
            offramp_amps = np.linspace(1, 0, int(offramp / 1000 * fs))
            signal[-len(offramp_amps):] *= offramp_amps
        elif onramp < 0:
            raise ValueError("Onramp cannot be negative")
        elif onramp == 0:
            pass

        # Return class
        return cls(signal, fs)

    # Manipulation

    def change_amplitude(self, factor):
        # get original frequencies
        self.stim *= factor

    # Visualization

    def play(self, loop=False):
        sd.play(self.samples, self.fs, loop=loop)
        sd.wait()

    def stop(self):
        sd.stop()

    def plot(self, title="Waveform of sound"):
        plt.clf()
        frames = np.arange(self.samples.size)
        plt.plot(frames, self.samples)
        plt.ylim([-1, 1])
        plt.ylabel("Amplitude")
        plt.xticks(ticks=[0, self.samples.size],
                   labels=[0, int(self.samples.size / self.fs * 1000)])
        plt.xlabel("Time (ms)")
        plt.title(title)
        plt.show()

    # Stats

    def get_duration(self):
        return len(self.samples) / self.fs

    # Out

    def write_wav(self, out_path):
        """
        Writes audio to disk.
        """
        wavfile.write(filename=out_path, rate=self.fs, data=self.samples)


class Sequence:
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

    def __init__(self, iois, metrical=False, n_bars=None, time_sig=None, quarternote_ms=None):
        # If metrical=True, that means there's an additional IOI for the final event.

        self.metrical = metrical
        self.time_sig = time_sig  # Used for metrical sequences
        self.quarternote_ms = quarternote_ms  # Used for metrical sequences
        self.n_bars = n_bars

        if any(ioi < 0 for ioi in iois):
            raise ValueError("IOIs cannot be negative.")
        else:
            self.iois = np.array(iois, dtype=np.float32)

    def __str__(self):
        if self.metrical and not self.time_sig:
            return f"Object of type Sequence (metrical version):\n{len(self.onsets)} events\nIOIs: {self.iois}\nOnsets:{self.onsets}\n"
        elif self.metrical and self.time_sig and self.quarternote_ms and self.n_bars:
            return f"Object of type Sequence (metrical version):\nTime signature: {self.time_sig}\nNumber of bars: {self.n_bars}\nQuarternote (ms): {self.quarternote_ms}\nNumber of events: {len(self.onsets)}\nIOIs: {self.iois}\nOnsets:{self.onsets}\n"
        else:
            return f"Object of type Sequence (non-metrical version):\n{len(self.onsets)} events\nIOIs: {self.iois}\nOnsets:{self.onsets}\n"

    @property
    def onsets(self):
        """Get the event onsets. These is the cumulative sum of Sequence.iois, with 0 additionally prepended.
        """

        if self.metrical:
            return np.cumsum(np.append(0, self.iois[:-1]), dtype=np.float32)
        else:
            return np.cumsum(np.append(0, self.iois), dtype=np.float32)

    @property
    def note_values(self):
        if self.metrical and self.time_sig and self.quarternote_ms:
            pass
        else:
            raise ValueError("This is not a metrical sequence. Use class method Sequence.from_note_values or one of "
                             "the metrical sequence generation methods.")

        ratios = self.iois / self.quarternote_ms / 4

        note_values = np.array([1 // ratio for ratio in ratios])

        return note_values

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

    @classmethod
    def from_note_values(cls, note_values, time_signature, quarternote_ms):
        ratios = np.array([1/note * time_signature[1] for note in note_values])

        n_bars = np.sum(ratios) / time_signature[0]

        if n_bars % 1 != 0:
            raise ValueError("The provided note values do not amount to whole bars.")
        else:
            n_bars = int(n_bars)

        iois = ratios * quarternote_ms

        return cls(iois, metrical=True, time_sig=time_signature, quarternote_ms=quarternote_ms, n_bars=n_bars)

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

    def plot_rhythm(self):

        if not self.metrical or not self.time_sig or not self.n_bars:
            raise ValueError('This is not a metrical sequence. Use Sequence.from_note_values or one of the'
                             'random metrical sequence methods to generate a sequence.')

        # We want to split up the notes into bars first

        # e.g. for (4, 4) that's 1, for (4, 8), that's 0.5 etc.
        full_bar = self.time_sig[0] * (1/self.time_sig[1])

        t = Track()

        # create initial bar
        b = Bar(meter=self.time_sig)
        # keep track of the index of the note_value
        note_i = 0
        # loop over the note values of the sequence
        for note_value in self.note_values:
            b.place_notes('G-4', self.note_values[note_i])
            # if bar is full, create new bar and add bar to track
            if b.current_beat == full_bar:
                t.add_bar(b)
                b = Bar(meter=self.time_sig)

            note_i += 1

        lp = '\\version "2.10.33"\n' + lilypond.from_Track(t) + '\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""\noddFooterMarkup = ""\nevenFooterMarkup = ""\n}'

        with open('temp.ly', 'w') as file:
            file.write(lp)

        command = f'lilypond -dbackend=eps --silent -dresolution=600 --png -o "temp" "temp.ly"'

        p = subprocess.Popen(command, shell=True).wait()

        img = mpimg.imread('temp.png')
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        filenames = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi', '.png', '.ly']
        filenames = ['temp' + x for x in filenames]

        for file in filenames:
            os.remove(file)


class StimulusSequence(Stimulus, Sequence):
    """
    StimulusSequence class which inherits from Stimulus and Sequence
    """

    def __init__(self, stimulus_obj, seq_obj):

        # Initialize parent Sequence class, so we can use self.onsets etc.
        Sequence.__init__(self, seq_obj.iois)

        # Save whether passed sequence is metrical or not
        self.metrical = seq_obj.metrical
        self.time_sig = seq_obj.time_sig
        self.quarternote_ms = seq_obj.quarternote_ms
        self.n_bars = seq_obj.n_bars

        # Use internal _make_stim method to combine stimulus_obj and seq_obj
        # It makes stimuli which are a nested 1-D array (i.e. for each onset a 1-D array of sound samples)
        stimuli = self._make_stim(stimulus_obj)

        # Make sound which saves the samples to self.samples
        self._make_sound(stimuli, self.onsets)

        # Initialize the Stimulus parent class
        Stimulus.__init__(self, self.samples, self.fs)

        # Then save stimuli for later use
        self.stim = stimuli

    def __str__(self, ):
        if self.metrical:
            return f"Object of type StimulusSequence (metrical version).\nIOIs: {self.iois}\nOnsets:{self.onsets}\n"
        else:
            return f"Object of type StimulusSequence (non-metrical version).\nIOIs: {self.iois}\nOnsets:{self.onsets}\n"

    def _make_stim(self, stimulus_obj):
        # If list of Stimulus objects was passed: Check a number of things (overlap etc.) and save fs and dtype.
        # The all_stimuli variable will later be used to generate the audio.
        if isinstance(stimulus_obj, list):
            # Check whether length of stimulus_obj is the same as onsets
            if not len(self.onsets) == len(stimulus_obj):
                raise ValueError("The number of Stimulus objects passed does not equal the number of onsets! "
                                 "Remember that you need one more Stimulus than the number of IOIs.")

            all_stimuli = np.array([snd.stim for snd in stimulus_obj])
            all_fs = [snd.fs for snd in stimulus_obj]
            all_dtypes = [snd.dtype for snd in stimulus_obj]

            # Check whether fs's are the same across the list
            if not all(x == all_fs[0] for x in all_fs):
                raise ValueError("The Stimulus objects in the passed list have different sampling frequencies!")
            else:
                self.fs = all_fs[0]
            # Check whether dtypes are the same
            if not all(x == all_dtypes[0] for x in all_dtypes):
                raise ValueError("The Stimulus objects in the passed list have different dtypes!")
            else:
                self.dtype = all_dtypes[0]

        # If a single Stimulus object was passed: Check a number of things (overlap etc.) and save fs and dtype.
        # Then make an all_stimuli variable which holds the samples of the Stimulus object n onsets times.
        elif isinstance(stimulus_obj, Stimulus):
            all_stimuli = np.tile(np.array(stimulus_obj.stim), (len(self.onsets), 1))
            self.fs = stimulus_obj.fs
            self.dtype = stimulus_obj.dtype

        else:
            raise AttributeError("Pass a Stimulus object or a list of Stimulus objects as the second argument.")

        return all_stimuli

    def _make_sound(self, stimuli, onsets):
        # Check for overlap
        for stim in stimuli:
            if any(ioi < len(stim) / self.fs * 1000 for ioi in self.iois):
                raise ValueError(
                    "The duration of the Stimulus is longer than one of the IOIs. The events will overlap: "
                    "either use different IOIs, or use a shorter stimulus sound.")

        # Generate an array of silence that has the length of all the onsets + one final stimulus.
        # In the case of a metrical sequence, we add the final ioi
        # The dtype is important, because that determines the values that the magnitudes can take.
        if self.metrical:
            array_length = int((onsets[-1] + self.iois[-1]) / 1000 * self.fs)
        elif not self.metrical:
            array_length = int((onsets[-1] / 1000 * self.fs) + stimuli[-1].size)
        else:
            raise ValueError("Error during calculation of array_length")

        samples = np.zeros(array_length, dtype=self.dtype)

        stimuli_with_onsets = list(zip(stimuli, onsets))

        if any(stimuli_with_onsets[i][0].size / self.fs * 1000 > np.diff(onsets)[i]
               for i in range(len(stimuli_with_onsets) - 1)):
            raise ValueError("The duration of one of the Stimuluss is longer than one of the IOIs. "
                             "The events will overlap: "
                             "either use different IOIs, or use a shorter Stimulus.")

        for stimulus, onset in stimuli_with_onsets:
            start_pos = int(onset * self.fs / 1000)
            end_pos = int(start_pos + stimulus.size)
            samples[start_pos:end_pos] = stimulus

        # then save the sound
        self.samples = samples
        self.stim = stimuli

    def _get_sound_with_metronome(self, ioi, metronome_amplitude):
        current_samples = self.samples
        duration = self.get_duration() * 1000

        n_metronome_clicks = int(duration // ioi)  # We want all the metronome clicks that fit in the seq.
        onsets = np.concatenate((np.array([0]), np.cumsum([ioi] * (n_metronome_clicks - 1))))


        # todo resample if fs is not same as self.fs (
        fs, metronome_samples = wavfile.read('metronome.wav')

        # change amplitude if necessary
        metronome_samples *= metronome_amplitude

        for onset in onsets:
            start_pos = int(onset * self.fs / 1000)
            end_pos = int(start_pos + metronome_samples.size)
            new_samples = current_samples[start_pos:end_pos] + metronome_samples
            current_samples[start_pos:end_pos] = new_samples   # we add the metronome sound to the existing sound

        return current_samples

    # Override Sequence and Stimulus manipulation methods so sound is regenerated when something changes
    def change_tempo(self, factor):
        super().change_tempo(factor)
        self._make_sound(self.stim, self.onsets)

    def change_tempo_linearly(self, total_change):
        super().change_tempo_linearly(total_change)
        self._make_sound(self.stim, self.onsets)

    def change_amplitude(self, factor):
        super().change_amplitude(factor)
        self._make_sound(self.stim, self.onsets)

    def plot_music(self):
        """
        I think i want a musical=True flag, so that the notes are saved when generating random notes.
        Then combined with metrical=True etc., we know of a sequence whether we can plot it as music.
        """
        pass

    def play(self, loop=False, metronome=False, metronome_amplitude=1):
        if metronome is True and self.time_sig and self.quarternote_ms:
            ioi = int((self.time_sig[1] / 4) * self.quarternote_ms)
            samples = self._get_sound_with_metronome(ioi, metronome_amplitude=metronome_amplitude)
        else:
            samples = self.samples

        sd.play(samples, self.fs, loop=loop)
        sd.wait()

    def write_wav(self, out_path, metronome=False, metronome_amplitude=1):
        """
        Writes audio to disk.
        """
        if metronome is True and self.time_sig and self.quarternote_ms:
            ioi = int((self.time_sig[1] / 4) * self.quarternote_ms)
            samples = self._get_sound_with_metronome(ioi, metronome_amplitude=metronome_amplitude)
        else:
            samples = self.samples

        wavfile.write(filename=out_path, rate=self.fs, data=samples)


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

    # Sequence and StimulusSequence objects need to be metrical:
    if not all(x.metrical for x in iterator):
        raise ValueError("Only metrical Sequence objects can be joined. This is intentional.")

    iois = [sequence.iois for sequence in iterator]
    iois = np.concatenate(iois)

    return Sequence(iois, metrical=True)
