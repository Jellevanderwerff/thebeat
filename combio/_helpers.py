import os
import importlib.resources as pkg_resources
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
    import mingus
    from mingus.extra import lilypond as mingus_lilypond
except ImportError:
    mingus = None
    mingus_lilypond = None


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
                        time_signature: tuple[int],
                        target_n: Optional[int] = None):
    # Find common denominator so we can work with integers, rather than floats
    common_denom = np.lcm(np.lcm.reduce(allowed_note_values), time_signature[1])  # numpy.int64

    allowed_numerators = common_denom // np.array(allowed_note_values)

    full_bar = time_signature[0] * (1 / time_signature[1])
    # The target is the desired length that should be returned by the deep first search.
    target = int(full_bar * common_denom)

    possibilities = all_possibilities(allowed_numerators, target)
    out_array = possibilities / common_denom

    if target_n is not None:
        out_array = np.where(len(out_array) == target_n)
        if len(out_array) == 0:
            raise ValueError("No random rhythms exist that adhere to these parameters. "
                             "Try providing different parameters.")

    return out_array


# todo Fix type hints
def get_lp_from_track(t, print_staff: bool):
    """
    Internal method for plotting a mingus Track object via lilypond.
    """

    remove_footers = """\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""
oddFooterMarkup = ""\nevenFooterMarkup = ""\n} """
    remove_staff = '{ \stopStaff \override Staff.Clef.color = #white'

    if print_staff is True:
        lp = '\\version "2.10.33"\n' + mingus_lilypond.from_Track(t) + remove_footers
    elif print_staff is False:
        lp = '\\version "2.10.33"\n' + remove_staff + mingus_lilypond.from_Track(t)[1:] + remove_footers
    else:
        raise ValueError("Wrong value specified for print_staff.")

    return lp


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

    return combio.rhythm.Rhythm(iois,
                                n_bars=n_bars,
                                time_signature=iterator[0].time_signature,
                                beat_ms=iterator[0].beat_ms)


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """This helper function normalizes audio based on the absolute max amplitude from the provided sound samples."""
    samples /= np.max(np.abs(samples), axis=0)
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
        save_format = None  # Just to get my IDE to stop bugging me

    # lilypond needs to write some temporary files to disk. These include a .eps and .png file.
    # if we want to keep that file, we copy it from the temporary files.
    with tempfile.TemporaryDirectory() as tmp_dir:

        command = ['lilypond', '-dbackend=eps', '--silent', '-dresolution=600', f'--png', '-o',
                   'rhythm', 'rhythm.ly']

        with open(os.path.join(tmp_dir, 'rhythm.ly'), 'w') as file:
            file.write(lp)

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

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.plot_rhythm(frames, samples)
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


def write_wav(samples: np.ndarray,
              fs: int,
              out_path: Union[str, os.PathLike],
              name: str,
              metronome: bool,
              metronome_ioi: int,
              metronome_amplitude: float) -> None:
    """
    This helper function writes the provided sound samples to disk as a wave file.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html for more info.
    """
    if metronome is True:
        samples = get_sound_with_metronome(samples, fs, metronome_ioi, metronome_amplitude)
    else:
        samples = samples

    out_path = str(out_path)

    if out_path.endswith('.wav'):
        path, filename = os.path.split(out_path)
    elif os.path.isdir(out_path):
        path = out_path
        # If a name was provided, use that one if no filepath was provided.
        if name:
            filename = f"{name}.wav"
        # If none of the above, simply write 'sequence.wav'.
        else:
            filename = f"sequence.wav"
    else:
        raise ValueError("Wrong out_path specified. Please provide a directory or a complete filepath that"
                         "ends in '.wav'.")

    wavfile.write(filename=os.path.join(path, filename), rate=fs, data=samples)
