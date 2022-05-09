import numpy as np
from scipy.signal import resample, square
from scipy.io import wavfile
from scipy.stats import entropy
import sounddevice as sd
import matplotlib.pyplot as plt


class Sound:
    """
    The Sound class.
    """

    def __init__(self, samples, fs):
        self.dtype = np.float32
        self.stim = samples
        self.samples = samples
        self.fs = fs

    def __str__(self):
        return f"Object of type Sound.\n"

    @classmethod
    def from_wav(cls, wav_filepath, new_fs: int = None):
        """

        This method loads a stimulus from a .wav file, reads in the samples and sets the
        Sound.samples and Sound.fs properties.

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
        # todo check whether this works.
        if samples.dtype == 'int16':
            samples = np.cast['float32'](samples)
            samples /= 32768
        elif samples.dtype == 'int32':
            samples = np.cast['float32'](samples)
            samples /= 2147483648
        elif samples.dtype == 'float32':
            pass
        else:
            print("Unknown dtype for wav file. Check out scipy documentation here:")
            print("https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html")

        if new_fs is None or new_fs == file_fs:
            fs = file_fs
        elif new_fs != file_fs:
            resample_factor = float(new_fs) / float(file_fs)
            resampled = resample(samples, int(len(samples) * resample_factor))
            samples = resampled
            fs = new_fs
        else:
            print("Error occurred when comparing sampling frequencies.")

        return cls(samples, fs)

    @classmethod
    def generate(cls, freq=440, fs=44100, duration=50, amplitude=0.8, osc='sine', onramp=10, offramp=10):
        samples = np.linspace(0, duration / 1000, int(fs * duration / 1000), endpoint=False, dtype=np.float32)
        if osc == 'sine':
            signal = amplitude * np.sin(2 * np.pi * freq * samples)
            signal = np.float32(signal)
        elif osc == 'square':
            signal = amplitude * square(2 * np.pi * freq * samples)
        else:
            raise ValueError("Choose existing oscillator (for now only 'sin')")

        # Create onramp
        onramp_amps = np.linspace(0, 1, int(onramp / 1000 * fs))
        signal[:len(onramp_amps)] *= onramp_amps

        # Create offramp
        offramp_amps = np.linspace(1, 0, int(offramp / 1000 * fs))
        signal[-len(offramp_amps):] *= offramp_amps

        # Set attributes
        return cls(signal, fs)

    # Manipulate

    def change_amplitude(self, factor):
        # get original frequencies
        self.stim *= factor

    def change_pitch(self, factor):
        """
        How to do it?
        """
        fourier = np.fft.rfft(self.stim)
        print(fourier)

    # Visualization

    def play(self, loop=False):
        # todo Check why this doesn't work.
        sd.play(self.samples, self.fs, loop=loop)
        # we need to wait explicitly so it also works in a script
        sd.wait()

    def stop(self):
        sd.stop()

    def plot(self):
        plt.clf()
        frames = np.arange(self.samples.size)
        plt.plot(frames, self.samples)
        plt.ylim([-1, 1])
        plt.ylabel("Amplitude")
        plt.xticks(ticks=[0, self.samples.size],
                   labels=[0, int(self.samples.size / self.fs * 1000)])
        plt.xlabel("Time (ms)")
        plt.title("Waveform of sound")
        plt.show()

    # Stats

    def get_duration(self):
        return len(self.samples) / self.fs

    # Out

    def write_wav(self, out_path):
        """
        Writes audio to disk.
        """
        wavfile.write(filename=out_path, rate=self.fs, data=self.stim)

    def write_ogg(self, out_path):
        pass


class Sequence:
    """
    Sequence class that holds a sequence of inter-onset intervals (IOIs) and stimulus onsets.
    Additionally has class methods that can be used for generating a new sequence.

    Attributes
    ----------

    iois : list of integers
        A list of the sequence's inter-onset intervals.
    stats : dict
        A dictionary containing some useful statistics about the sequence.
    onsets : list of numpy.int64's
        A list of the events' onset values (i.e. t values). t=0 is additionally added here.

    Methods
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

    """

    def __init__(self, iois):
        if any(ioi < 0 for ioi in iois):
            raise ValueError("IOIs cannot be negative.")
        else:
            self.iois = np.array(iois, dtype=np.float32)

    def __str__(self):
        """
        How do we want to print the object?
        """
        return f"Object of type Sequence.\nIOIs: {self.iois}\nOnsets:{self.onsets}\n"

    @property
    def onsets(self):
        return np.cumsum(np.append(0, self.iois), dtype=np.float32)  # The onsets calculated from the IOIs.

    @classmethod
    def generate_random_normal(cls, n: int, mu: int, sigma: int, rng=None):
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

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        round_iois = np.round(rng.normal(loc=mu, scale=sigma, size=n - 1))

        return cls(round_iois)

    @classmethod
    def generate_random_uniform(cls, n: int, a: int, b: int, rng=None):
        """

        Class method that generates a sequence of random inter-onset intervals based on a uniform distribution.
        Note that there will be n-1 IOIs in a sequence.

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

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        round_iois = np.round(rng.uniform(low=a, high=b, size=n - 1))

        return cls(round_iois)

    @classmethod
    def generate_random_poisson(cls, n: int, lam: int, rng=None):
        """

        Class method that generates a sequence of random inter-onset intervals based on a Poisson distribution.
        Note that there will be n-1 IOIs in a sequence.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        lam : int
            The desired value for lambda.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        round_iois = np.round(rng.poisson(lam=lam, size=n - 1))

        return cls(round_iois)

    @classmethod
    def generate_random_exponential(cls, n: int, lam: int, rng=None):
        """

        Class method that generates a sequence of random inter-onset intervals based on an exponential distribution.
        Note that there will be n-1 IOIs in a sequence.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        lam : int
           The desired value for lambda.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        round_iois = np.round(rng.exponential(scale=lam, size=n - 1))

        return cls(round_iois)

    @classmethod
    def generate_isochronous(cls, n: int, ioi: int):
        """

        Class method that generates a sequence of isochronous inter-onset intervals.
        Note that there will be n-1 IOIs in a sequence.


        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        ioi : int
            The inter-onset interval to be used between all events.

        Returns
        -------
        Returns an object of class Sequence.

        """

        return cls(np.round([ioi] * (n - 1)))

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
            'ioi_max': np.max(self.iois),
            'ioi_entropy': entropy(self.iois)
        }


class SoundSequence(Sound, Sequence):
    """
    SoundSequence class which inherits from Sound and Sequence
    """

    def __init__(self, sound_obj, seq_obj):

        # Initialize parent Sequence class, so we can use self.onsets etc.
        Sequence.__init__(self, seq_obj.iois)

        # If list of Sound objects was passed: Check a number of things (overlap etc.) and save fs and dtype.
        # The all_stimuli variable will later be used to generate the audio.
        if isinstance(sound_obj, list):
            # Check whether length of sound_obj is the same as onsets
            if not len(self.onsets) == len(sound_obj):
                raise ValueError("The number of Sound objects passed does not equal the number of onsets! "
                                 "Remember that you need one more Sound than the number of IOIs.")

            all_stimuli = np.array([snd.stim for snd in sound_obj])
            all_fs = [snd.fs for snd in sound_obj]
            all_dtypes = [snd.dtype for snd in sound_obj]

            # Check whether fs's are the same across the list
            if not all(x == all_fs[0] for x in all_fs):
                raise ValueError("The Sound objects in the passed list have different sampling frequencies!")
            else:
                self.fs = all_fs[0]
            # Check whether dtypes are the same
            if not all(x == all_dtypes[0] for x in all_dtypes):
                raise ValueError("The Sound objects in the passed list have different dtypes!")
            else:
                self.dtype = all_dtypes[0]

        # If a single Sound object was passed: Check a number of things (overlap etc.) and save fs and dtype.
        # Then make an all_stimuli variable which holds the samples of the Sound object n onsets times.
        elif isinstance(sound_obj, Sound):
            if any(x < len(sound_obj.stim) / sound_obj.fs * 1000 for x in self.iois):
                raise ValueError(
                    "The duration of the Sound is longer than one of the IOIs. The events will overlap: "
                    "either use different IOIs, or use a shorter stimulus sound.")
            else:
                # todo This list comprehension doesn't make sense (though it works), make better!
                all_stimuli = [np.array(sound_obj.stim) for x in range(len(self.onsets))]
                all_stimuli = np.array(all_stimuli)  # This is now an array with 10 rows, and X (in the example 2205)
                                                     # columns.
                self.fs = sound_obj.fs
                self.dtype = sound_obj.dtype

        else:
            raise AttributeError("Pass a Sound object or a list of Sound objects as the second argument.")

        # Make sound which saves the samples to self.samples
        self._make_sound(all_stimuli, self.onsets)

        # Initialize the Sound parent class
        Sound.__init__(self, self.samples, self.fs)

        # And save the newly created stimuli for later use
        self.stim = all_stimuli  # todo Check whether this one is necessary, because this also happens in _make_sound

    def __str__(self, ):
        return f"Object of type SoundSequence.\nIOIs: {self.iois}\nOnsets:{self.onsets}\n"

    def _make_sound(self, stimuli, onsets):
        # todo Check for overlap, which now only happens when SoundSequence object is initialize, not when manipulating.
        # Generate an array of silence that has the length of all the onsets + one final stimulus.
        # The dtype is important, because that determines the values that the magnitudes can take.
        array_length = (max(onsets) / 1000 * self.fs) + stimuli[-1][:].size  # Total duration + duration of one stimulus
        samples = np.zeros(int(array_length), dtype=self.dtype)

        stimuli_with_onsets = list(zip(stimuli, onsets))

        if any(stimuli_with_onsets[i][0].size / self.fs * 1000 > np.diff(onsets)[i]
               for i in range(len(stimuli_with_onsets) - 1)):
            raise ValueError("The duration of one of the Sounds is longer than one of the IOIs. "
                             "The events will overlap: "
                             "either use different IOIs, or use a shorter Sound.")

        for stimulus, onset in stimuli_with_onsets:
            start_pos = int(onset * self.fs / 1000)
            end_pos = int(start_pos + stimulus.size)
            samples[start_pos:end_pos] = stimulus

        # then save the sound
        self.samples = samples
        self.stim = stimuli

    # Override Sequence and Sound manipulation methods so sound is regenerated when something changes
    def change_tempo(self, factor):
        super().change_tempo(factor)
        self._make_sound(self.stim, self.onsets)

    def change_tempo_linearly(self, total_change):
        super().change_tempo_linearly(total_change)
        self._make_sound(self.stim, self.onsets)

    def change_amplitude(self, factor):
        super().change_amplitude(factor)
        self._make_sound(self.stim, self.onsets)

    def change_pitch(self, factor):
        super().change_pitch(factor)
        self._make_sound(self.stim, self.onsets)


# Example usage
if __name__ == "__main__":

    # Example of a sequence
    sequence = Sequence.generate_random_uniform(n=5, a=200, b=600)
    print(sequence)
    sequence.change_tempo(2)
    print(sequence)

    # Example of a sound
    sound = Sound.from_wav('click01.wav')
    sound.plot()

    # Example of a sound sequence with the same sound used throughout
    sound_sequence = SoundSequence(sound, sequence)
    sound_sequence.plot()
    sound_sequence.write_wav('sequence_samesound.wav')

    # Example of a sound sequence with different sounds for each event (we pass a list of Sound objects of equal length)
    sequence = Sequence.generate_isochronous(n=5, ioi=500)

    tone_heights = [500, 300, 600, 100, 300]
    sounds = [Sound.generate(freq=tone_height) for tone_height in tone_heights]

    sound_sequence = SoundSequence(sounds, sequence)
    sound_sequence.plot()
    sound_sequence.write_wav('sound_sequence.wav')

    # All Sequence and Sound manipulation methods you can also use for SoundSequence objects:
    sound_sequence = SoundSequence(Sound.generate(freq=440, onramp=10, offramp=10),
                                   Sequence.generate_isochronous(n=5, ioi=500))
    sound_sequence.plot()

    sound_sequence.change_amplitude(factor=0.01)
    sound_sequence.plot()

    print(sound_sequence)
    sound_sequence.change_tempo(factor=2)
    print(sound_sequence)

    sound_sequence.play()