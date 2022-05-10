import numpy as np
from scipy.signal import resample, square
from scipy.io import wavfile
from scipy.stats import entropy
import sounddevice as sd
import matplotlib.pyplot as plt


class Sound:
    """
    Sound class that holds a Numpy 1-D array of sound that is either generated, or read from a .wav file.
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
        Contains the stimulus samples. In most cases this will be the same as Sound.samples.

    Class methods
    -------------
    from_wav(wav_filepath, new_fs=None)
        Read a .wav file from disk.
    generate(freq=440, fs=44100, duration=50, amplitude=0.8, osc='sine', onramp=10, offramp=10)
        Generate a sound using a sine or square oscillator.

    Methods
    -------
    change_amplitude(factor)
        Change the amplitude of the Sound by 'factor'. E.g. 2 will be twice as loud, 0.5 will be half as loud.
    play(loop=False)
        Play the Sound using sounddevice.
    stop()
        Stop sounddevice playback.
    plot()
        Plot the Sound's waveform using matplotlib.
    get_duration()
        Get the duration of the Sound in seconds.
    write_wav(out_path)
        Write the Sound to disk as a .wav file.


    """

    def __init__(self, samples, fs: int):
        self.dtype = np.float32
        self.stim = samples
        self.samples = samples
        self.fs = fs

    def __str__(self):
        return f"Object of type Sound.\nSound duration: {self.get_duration()} seconds."

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
            print("Error occurred when comparing sampling frequencies.")

        return cls(samples, fs)

    @classmethod
    def generate(cls, freq=440, fs=44100, duration=50, amplitude=0.8, osc='sine', onramp=0, offramp=0):
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
        wavfile.write(filename=out_path, rate=self.fs, data=self.stim)


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

    def __init__(self, iois):
        if any(ioi < 0 for ioi in iois):
            raise ValueError("IOIs cannot be negative.")
        else:
            self.iois = np.array(iois, dtype=np.float32)

    def __str__(self):
        return f"Object of type Sequence.\nIOIs: {self.iois}\nOnsets:{self.onsets}\n"

    @property
    def onsets(self):
        """Get the event onsets. These is the cumulative sum of Sequence.iois, with 0 additionally prepended.
        """
        return np.cumsum(np.append(0, self.iois), dtype=np.float32)

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
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.

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
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.

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
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.


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

        # Use internal _make_stim method to combine sound_obj and seq_obj
        # It makes stimuli which are a nested 1-D array (i.e. for each onset a 1-D array of sound samples)
        stimuli = self._make_stim(sound_obj)

        # Make sound which saves the samples to self.samples
        self._make_sound(stimuli, self.onsets)

        # Initialize the Sound parent class
        Sound.__init__(self, self.samples, self.fs)

        # Then save stimuli for later use
        self.stim = stimuli

    def __str__(self, ):
        return f"Object of type SoundSequence.\nIOIs: {self.iois}\nOnsets:{self.onsets}\n"

    def _make_stim(self, sound_obj):
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
            all_stimuli = np.tile(np.array(sound_obj.stim), (len(self.onsets), 1))
            self.fs = sound_obj.fs
            self.dtype = sound_obj.dtype

        else:
            raise AttributeError("Pass a Sound object or a list of Sound objects as the second argument.")

        return all_stimuli

    def _make_sound(self, stimuli, onsets):
        # Check for overlap
        for stim in stimuli:
            if any(ioi < len(stim) / self.fs * 1000 for ioi in self.iois):
                raise ValueError(
                    "The duration of the Sound is longer than one of the IOIs. The events will overlap: "
                    "either use different IOIs, or use a shorter stimulus sound.")


        # Generate an array of silence that has the length of all the onsets + one final stimulus.
        # The dtype is important, because that determines the values that the magnitudes can take.
        array_length = (max(onsets) / 1000 * self.fs) + stimuli[-1].size  # Total duration + duration of one stimulus
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


# Example usage
if __name__ == "__main__":
    # Example of a sequence
    sequence = Sequence.generate_random_uniform(n=5, a=200, b=600)
    print(sequence)
    sequence.change_tempo(2)
    print(sequence)

    # Example of a sound
    sound = Sound.from_wav('click01.wav')
    sound.plot(title="Waveform for click01.wav")

    # Example of a sound sequence with the same sound used throughout
    sound_sequence = SoundSequence(sound, sequence)
    sound_sequence.plot("SoundSequence with same sound throughout")
    sound_sequence.write_wav('sequence_samesound.wav')

    # Example of a sound sequence with different sounds for each event (we pass a list of Sound objects of equal length)
    sequence = Sequence.generate_isochronous(n=5, ioi=500)

    tone_heights = [500, 300, 600, 100, 300]
    sounds = [Sound.generate(freq=tone_height) for tone_height in tone_heights]

    sound_sequence = SoundSequence(sounds, sequence)
    sound_sequence.plot("SoundSequence with different sounds")
    sound_sequence.write_wav('sound_sequence.wav')

    # All Sequence and Sound manipulation methods you can also use for SoundSequence objects:
    sound_sequence = SoundSequence(Sound.generate(freq=440, onramp=10, offramp=10),
                                   Sequence.generate_isochronous(n=5, ioi=500))

    sound_sequence.change_amplitude(factor=0.01)
    sound_sequence.plot("Manipulation changed amplitude")

    # Accelerando
    sequence = Sequence.generate_isochronous(n=10, ioi=500)
    sound = Sound.generate(freq=440, osc='sine', duration=10)
    sound_sequence = SoundSequence(sound, sequence)
    sound_sequence.plot("Accelerando before")
    sound_sequence.change_tempo_linearly(total_change=2)
    sound_sequence.plot("Accelerando after")
    # sound_sequence.play()

    # Ritardando
    sound_sequence = SoundSequence(Sound.generate(freq=440, duration=50, onramp=10, offramp=10),
                                   Sequence.generate_isochronous(n=10, ioi=500))
    sound_sequence.plot("Ritardando before")
    sound_sequence.change_tempo_linearly(total_change=0.5)
    sound_sequence.plot("Ritardando after")
    #sound_sequence.play()