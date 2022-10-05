import os
import importlib.resources as pkg_resources
import warnings

import scipy.signal

import combio.resources
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample
import sounddevice as sd
from typing import Union, Optional
from combio._decorators import requires_lilypond
import tempfile
import shutil
import subprocess
import matplotlib.image as mpimg

try:
    import abjad
except ImportError:
    abjad


# todo Fix type hints
def all_possibilities(nums, target):
    """
    I stole this code
    """

    res = []
    nums.sort()

    def dfs(left, path):
        if not left:
            res.append(np.array(path))
        else:
            for val in nums:
                if val > left:
                    break
                current_path = np.append(path, val)
                dfs(left - val, current_path)

    dfs(target, np.array([]))

    return np.array(res, dtype=object)


# todo Fix type hints
def all_rhythmic_ratios(allowed_note_values: Union[list, np.ndarray],
                        time_signature: tuple[int]):
    # Find common denominator so we can work with integers, rather than floats
    common_denom = np.lcm(np.lcm.reduce(allowed_note_values), time_signature[1])  # numpy.int64

    allowed_numerators = common_denom // np.array(allowed_note_values)

    full_bar = time_signature[0] * (1 / time_signature[1])
    # The target is the desired length that should be returned by the deep first search.
    target = int(full_bar * common_denom)

    possibilities = all_possibilities(allowed_numerators, target)
    out_array = possibilities / common_denom

    return out_array


def check_for_overlap(stimulus_durations, onsets):
    for i in range(len(onsets)):
        try:
            ioi_after_onset = onsets[i + 1] - onsets[i]
            if ioi_after_onset < stimulus_durations[i]:
                raise ValueError(
                    "The duration of one or more stimuli is longer than its respective IOI. "
                    "The events will overlap: either use different IOIs, or use a shorter stimulus sound.")
        except IndexError:
            pass


# todo Fix type hints

def get_major_scale(tonic: str,
                    octave: int):
    if abjad is None:
        raise ImportError("This requires the abjad package")

    intervals = "M2 M2 m2 M2 M2 M2 m2".split()
    intervals = [abjad.NamedInterval(_) for _ in intervals]

    pitches = []

    pitch = abjad.NamedPitch(tonic, octave=octave)

    pitches.append(pitch)

    for interval in intervals:
        pitch = pitch + interval

        pitches.append(pitch)

    return pitches


def get_sound_with_metronome(samples: np.ndarray,
                             fs: int,
                             metronome_ioi: int,
                             metronome_amplitude: float) -> np.ndarray:
    """This helper function adds a metronome sound to a NumPy array of sound samples.
     It works for both mono and stereo sounds."""

    sound_samples = samples
    duration_s = sound_samples.shape[0] / fs * 1000

    n_metronome_clicks = int(duration_s // metronome_ioi)  # We want all the metronome clicks that fit in the seq.
    onsets = np.concatenate((np.array([0]), np.cumsum(np.repeat(metronome_ioi, n_metronome_clicks - 1))))

    with pkg_resources.path(combio.resources, 'metronome.wav') as metronome_path:
        metronome_fs, metronome_samples = wavfile.read(metronome_path)

    # resample metronome sound if provided sound has different sampling frequency
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


# todo Use NumPy functions
def join_rhythms(iterator):
    """
    This function can join multiple Rhythm objects.
    """

    # Check whether iterable was passed
    if not hasattr(iterator, '__iter__'):
        raise ValueError("Please pass this function a list or other iterable object.")

    # Check whether all the objects are of the same type
    if not all(isinstance(rhythm, combio.rhythm.Rhythm) for rhythm in iterator):
        raise ValueError("You can only join multiple Rhythm objects.")

    if not all(rhythm.time_signature == iterator[0].time_signature for rhythm in iterator):
        raise ValueError("Provided rhythms should have the same time signatures.")

    if not all(rhythm.beat_ms == iterator[0].beat_ms for rhythm in iterator):
        raise ValueError("Provided rhythms should have same tempo (beat_ms).")

    iois = [rhythm.iois for rhythm in iterator]
    iois = np.concatenate(iois)
    n_bars = int(np.sum([rhythm.n_bars for rhythm in iterator]))

    return combio.rhythm.Rhythm(iois, time_signature=iterator[0].time_signature, beat_ms=iterator[0].beat_ms)


def make_ramps(samples, fs, onramp, offramp, ramp_type):
    """Internal function used to create on- and offramps. Supports 'linear' and 'raised-cosine' ramps."""
    # Create onramp
    if onramp > 0:
        onramp_samples_len = int(onramp / 1000 * fs)
        end_point = onramp_samples_len

        if ramp_type == 'linear':
            onramp_amps = np.linspace(0, 1, onramp_samples_len)

        elif ramp_type == 'raised-cosine':
            hanning_complete = np.hanning(onramp_samples_len * 2)
            onramp_amps = hanning_complete[:(hanning_complete.shape[0] // 2)]  # only first half of Hanning window

        else:
            raise ValueError("Unknown ramp type. Use 'linear' or 'raised-cosine'")

        samples[:end_point] *= onramp_amps

    elif onramp < 0:
        raise ValueError("Onramp cannot be negative")
    elif onramp == 0:
        pass
    else:
        raise ValueError("Wrong value supplied to onramp argument.")

    # Create offramp
    if offramp > 0:
        offramp_samples_len = int(offramp / 1000 * fs)
        start_point = samples.shape[0] - offramp_samples_len

        if ramp_type == 'linear':
            offramp_amps = np.linspace(1, 0, int(offramp / 1000 * fs))
        elif ramp_type == 'raised-cosine':
            hanning_complete = np.hanning(offramp_samples_len * 2)
            offramp_amps = hanning_complete[hanning_complete.shape[0] // 2:]
        else:
            raise ValueError("Unknown ramp type. Use 'linear' or 'raised-cosine'")

        samples[start_point:] *= offramp_amps

    elif offramp < 0:
        raise ValueError("Offramp cannot be negative")
    elif offramp == 0:
        pass
    else:
        raise ValueError("Wrong value supplied to offramp argument.")

    return samples


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """This helper function normalizes audio based on the absolute max amplitude from the provided sound samples."""
    # todo Allow stereo
    samples = samples / np.max(np.abs(samples), axis=0)
    return samples


@requires_lilypond
def plot_lp(lp,
            filepath: Union[os.PathLike, str] = None,
            suppress_display: bool = False):
    # If a filepath is given, we'll use its format. If none is given, we'll use .png as the format to
    # eventually show.
    if filepath:
        save_format = os.path.splitext(filepath)[1]
        if save_format not in ['.eps', '.png']:
            raise ValueError("Can only export .png or .eps files.")
    else:
        save_format = None

    # lilypond needs to write some temporary files to disk. These include a .eps and .png file.
    # if we want to keep that file, we copy it from the temporary files.
    with tempfile.TemporaryDirectory() as tmp_dir:

        with open(os.path.join(tmp_dir, 'rhythm.ly'), 'w') as file:
            file.write(lp)

        command = ['lilypond', '-dbackend=eps', '--silent', '-dresolution=600', f'--png', '-o',
                   'rhythm', 'rhythm.ly']

        subprocess.run(command, cwd=tmp_dir, check=True)

        # read the png as image
        result_path_png = os.path.join(tmp_dir, 'rhythm.png')
        image = mpimg.imread(result_path_png)

        # crop the png
        white = np.array([1, 1, 1])
        mask = np.abs(image - white).sum(axis=2) < 0.05
        coords = np.array(np.nonzero(~mask))
        top_left = np.min(coords, axis=1)
        bottom_right = np.max(coords, axis=1)
        img_cropped = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        # the eps we cannot crop that easily unfortunately, so we use the one created
        # by lilypond if an .eps is desired.
        if filepath and save_format == '.eps':
            path_to_file_for_saving = os.path.join(tmp_dir, 'rhythm-1.eps')
            shutil.copy(path_to_file_for_saving, filepath)

    plt.imshow(img_cropped)
    plt.axis('off')

    # show plot if necessary
    if not suppress_display:
        plt.show()

    # Save cropped .png if necessary
    if filepath and save_format == '.png':
        plt.imsave(filepath, img_cropped)

    return plt.gca(), plt.gcf()


def plot_sequence_single(onsets: Union[list, np.ndarray],
                         style: str = 'seaborn',
                         title: Optional[str] = None,
                         linewidths: Optional[Union[list, np.ndarray, None]] = None,
                         figsize: Optional[tuple] = None,
                         suppress_display: bool = False):
    """This helper function returns a sequence plot."""

    # Make onsets array
    onsets = np.array(list(onsets))

    # X axis
    x_start = 0
    max_onset_ms = np.max(onsets)

    # Above 10s we want seconds on the x axis, otherwise milliseconds
    if max_onset_ms > 10000:
        onsets = onsets / 1000
        linewidths = np.array(linewidths) / 1000
        x_end = (max_onset_ms / 1000) + linewidths[-1]
        x_label = "Time (s)"
    else:
        onsets = onsets
        x_end = max_onset_ms + linewidths[-1]
        x_label = "Time (ms)"

    # Make plot
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.axes.set_xlabel(x_label)
        ax.set_ylim(0, 1)
        ax.set_xlim(x_start, x_end)
        ax.barh(0.5, width=linewidths, height=1.0, left=onsets)
        ax.axes.set_title(title)
        ax.axes.yaxis.set_visible(False)

    # Show plot
    if suppress_display is False:
        plt.show()

    # Additionally return plot
    return fig, ax


def plot_waveform(samples: np.ndarray,
                  fs: int,
                  n_channels: int,
                  style: str = 'seaborn',
                  title: Optional[str] = None,
                  figsize: Optional[tuple] = None,
                  suppress_display: bool = False) -> tuple[plt.Axes, plt.Figure]:
    """
    This helper function plots a waveform of a sound using matplotlib. It returns a fig and an ax object.

    Parameters
    ----------
    samples
        The array containing the sound samples.
    fs
        Sampling frequency in hertz (e.g. 48000).
    n_channels
        Number of channels (1 for mono, 2 for stereo).
    style
        Style used by matplotlib, defaults to 'seaborn'. See matplotlib docs for other styles.
    title
        If desired, one can provide a title for the plot.
    figsize
        A tuple containing the desired output size of the plot in inches.
    suppress_display
        If 'True', plt.show() is not run. Defaults to 'False'.


    Returns
    -------
    fig
        A matplotlib Figure object
    ax
        A matplotlib Axes object

    """

    # if we have two channels, we want the waveform to be opaque
    if n_channels == 1:
        alph = 1.0
    elif n_channels == 2:
        alph = 0.5
    else:
        raise ValueError("Unexpected number of channels.")

    # Above 10s, we want seconds on the x axis, below that we want milliseconds
    n_frames = samples.shape[0]
    if n_frames / fs > 10:
        frames = np.linspace(start=0,
                             stop=samples.shape[0] / fs,
                             num=samples.shape[0])
        x_label = "Time (s)"
    else:
        frames = np.linspace(start=0,
                             stop=samples.shape[0] / fs * 1000,
                             num=samples.shape[0])
        x_label = "Time (ms)"

    # Plot
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.plot(frames, samples, alpha=alph)
        if n_channels == 2:
            ax.legend(["Left channel", "Right channel"], loc=0, frameon=True)

        ax.set_ylabel("Amplitude")
        ax.set_xlabel(x_label)
        ax.set_title(title)

    if suppress_display is False:
        plt.show()

    return fig, ax


def play_samples(samples: np.ndarray,
                 fs: int,
                 mean_ioi: int,
                 loop: bool,
                 metronome: bool,
                 metronome_amplitude: float) -> None:
    """This helper function uses the sounddevice library to play a sound, either with or without metronome."""
    if metronome is True:
        samples = get_sound_with_metronome(samples, fs, mean_ioi, metronome_amplitude=metronome_amplitude)
    else:
        samples = samples

    sd.play(samples, fs, loop=loop)
    sd.wait()


def synthesize_sound(duration_ms: int,
                     fs: int,
                     freq: Union[int, float],
                     amplitude: float,
                     osc: str) -> np.ndarray:
    # Get duration in s
    t = duration_ms / 1000

    samples = np.linspace(0, t, int(fs * t), endpoint=False, dtype=np.float64)

    if osc == 'sine':
        samples = amplitude * np.sin(2 * np.pi * freq * samples)
        plt.plot(samples)
    elif osc == 'square':
        samples = amplitude * scipy.signal.square(2 * np.pi * freq * samples)
    elif osc == 'sawtooth':
        samples = amplitude * scipy.signal.sawtooth(2 * np.pi * freq * samples)
    else:
        raise ValueError("Choose existing oscillator (for now only 'sine' or 'square')")

    return samples


def write_wav(samples: np.ndarray,
              fs: int,
              filepath: Union[str, os.PathLike],
              metronome: bool,
              metronome_ioi: int,
              metronome_amplitude: float) -> None:
    """
    This helper function writes the provided sound samples to disk as a wave file.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html for more info.
    """
    if metronome is True:
        samples = get_sound_with_metronome(samples, fs, metronome_ioi, metronome_amplitude)

    filepath = str(filepath)

    if not filepath.endswith('.wav'):
        warnings.warn("File saved with extension other than .wav")

    wavfile.write(filename=filepath, rate=fs, data=samples)
