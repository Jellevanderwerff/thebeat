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

import importlib.resources as pkg_resources
import os
import shutil
import subprocess
import sys
import tempfile
import warnings

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.signal
import sounddevice as sd
from scipy.io import wavfile

import thebeat.resources
from thebeat._decorators import requires_lilypond

try:
    import abjad
except ImportError:
    abjad = None


def all_possibilities(numbers: list, target: float) -> np.ndarray:
    """
    Use a deep-first search to find all possible combinations of 'numbers' that sum to 'target'.
    Returns a NumPy array of NumPy arrays (with dtype=object to allow nested array).
    """

    res = []
    numbers.sort()

    def dfs(left, path):
        if not left:
            res.append(np.array(path))
        else:
            for val in numbers:
                if val > left:
                    break
                current_path = np.append(path, val)
                dfs(left - val, current_path)

    dfs(target, np.array([]))

    return np.array(res, dtype=object)


def all_rhythmic_ratios(allowed_note_values: list | np.ndarray,
                        time_signature: tuple[int, int]):
    # Find common denominator so we can work with integers, rather than floats
    common_denom = np.lcm(np.lcm.reduce(allowed_note_values), time_signature[1])  # numpy.int64

    # Which numerators are allowed?
    allowed_numerators = common_denom // np.array(allowed_note_values)

    # How much do we need for one bar?
    full_bar = time_signature[0] * (1 / time_signature[1])

    # The target is the desired length that should be returned by the deep first search.
    target = int(full_bar * common_denom)

    # Get all combinations of allowed_numerators that sum to target
    possibilities = all_possibilities(allowed_numerators, target)

    # Convert integers back to floats
    out_array = possibilities / common_denom

    return out_array


def check_for_overlap(sounds_durations, onsets):
    # todo rewrite this function to something more sensible
    for i in range(len(onsets)):
        try:
            ioi_after_onset = onsets[i + 1] - onsets[i]
            if ioi_after_onset < sounds_durations[i]:
                raise ValueError(
                    "The duration of one or more stimuli is longer than its respective IOI. "
                    "The events will overlap: either use different IOIs, or use a shorter stimulus sound.")
        except IndexError:
            pass


def check_sound_properties_sameness(objects: np.typing.ArrayLike):
    """This helper function checks whether the objects in the passed iterable have the same sound properties.
    Raises errors if properties are not the same."""
    if not all(obj.fs == objects[0].fs for obj in objects):
        raise ValueError("The objects do not have the same sampling frequency.")
    elif not all(obj.n_channels == objects[0].n_channels for obj in objects):
        raise ValueError("These objects do not have the same number of channels.")
    elif not all(obj.dtype == objects[0].dtype for obj in objects):
        raise ValueError("These objects do not have the same Numpy data type. Check out "
                         "https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html for info.")
    else:
        return True


def get_sound_with_metronome(samples: np.ndarray,
                             fs: int,
                             metronome_ioi: float,
                             metronome_amplitude: float) -> np.ndarray:
    """This helper function adds a metronome sound to a NumPy array of sound samples.
     It works for both mono and stereo sounds."""

    sound_samples = samples
    duration_s = sound_samples.shape[0] / fs * 1000

    n_metronome_clicks = int(duration_s // metronome_ioi)  # We want all the metronome clicks that fit in the seq.
    onsets = np.concatenate((np.array([0]), np.cumsum(np.repeat(metronome_ioi, n_metronome_clicks - 1))))

    metronome_file = 'metronome_mono.wav' if samples.ndim == 1 else 'metronome_stereo.wav'

    if sys.version_info < (3, 9):
        metronome_path_ctxmgr = pkg_resources.path(thebeat.resources, metronome_file)
    else:
        metronome_path_ctxmgr = pkg_resources.as_file(pkg_resources.files(thebeat.resources) / metronome_file)

    with metronome_path_ctxmgr as metronome_path:
        metronome_fs, metronome_samples = wavfile.read(metronome_path)

    # resample metronome sound if provided sound has different sampling frequency
    if metronome_fs != fs:
        resample_factor = float(fs) / float(metronome_fs)
        resampled = resample(metronome_samples, int(len(metronome_samples) * resample_factor))
        metronome_samples = resampled

    # change amplitude if necessary
    if metronome_amplitude:
        metronome_samples = metronome_samples * metronome_amplitude

    for onset in onsets:
        start_pos = int(onset * fs / 1000)
        end_pos = int(start_pos + metronome_samples.shape[0])
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
        raise TypeError("Please pass this function a list or other iterable object.")

    # Check whether all the objects are of the same type
    if not all(isinstance(rhythm, thebeat.music.Rhythm) for rhythm in iterator):
        raise TypeError("You can only join multiple Rhythm objects.")

    if not all(rhythm.time_signature == iterator[0].time_signature for rhythm in iterator):
        raise ValueError("Provided rhythms should have the same time signatures.")

    if not all(rhythm.beat_ms == iterator[0].beat_ms for rhythm in iterator):
        raise ValueError("Provided rhythms should have same tempo (beat_ms).")

    iois = [rhythm.iois for rhythm in iterator]
    iois = np.concatenate(iois)

    return thebeat.music.Rhythm(iois, time_signature=iterator[0].time_signature, beat_ms=iterator[0].beat_ms)


def sequence_to_binary(sequence: thebeat.core.Sequence, resolution: int):
    """
    Converts a sequence of millisecond onsets to a series of zeros and ones.
    Ones for the onsets.
    """
    duration = np.max(sequence.onsets)
    zeros_n = int(duration / resolution)
    signal = np.zeros(zeros_n)

    for onset in sequence.onsets:
        index = 0 if onset == 0 else int(onset / resolution) - 1
        signal[index] = 1

    if sequence.end_with_interval is True:
        signal = np.append(signal, np.zeros(int(sequence.iois[-1] / resolution - 1)))

    return np.array(signal)


def make_ramps(samples, fs, onramp_ms, offramp_ms, ramp_type):
    """Internal function used to create on- and offramps. Supports 'linear' and 'raised-cosine' ramps."""

    # Create onramp
    onramp_samples_len = int(onramp_ms / 1000 * fs)
    offramp_samples_len = int(offramp_ms / 1000 * fs)

    if onramp_ms < 0 or offramp_ms < 0:
        raise ValueError("Ramp duration must be positive.")
    elif onramp_ms == 0 and offramp_ms == 0:
        return samples
    elif onramp_samples_len > len(samples):
        raise ValueError("Onramp is longer than stimulus")
    elif offramp_samples_len > len(samples):
        raise ValueError("Offramp is longer than stimulus")

    # ONRAMP
    if ramp_type == 'linear':
        onramp_amps = np.linspace(0, 1, onramp_samples_len)

    elif ramp_type == 'raised-cosine':
        hann_complete = scipy.signal.windows.hann(onramp_samples_len * 2)
        onramp_amps = hann_complete[:hann_complete.shape[0] // 2]

    else:
        raise ValueError("Unknown ramp type. Use 'linear' or 'raised-cosine'")

    end_point = onramp_samples_len

    if samples.ndim == 1:
        samples[:end_point] *= onramp_amps
    elif samples.ndim == 2:
        samples[:end_point, 0] *= onramp_amps
        samples[:end_point, 1] *= onramp_amps

    # OFFRAMP
    start_point = len(samples) - offramp_samples_len

    if ramp_type == 'linear':
        offramp_amps = np.linspace(1, 0, int(offramp_ms / 1000 * fs))
    elif ramp_type == 'raised-cosine':
        hann_complete = scipy.signal.windows.hann(offramp_samples_len * 2)
        offramp_amps = hann_complete[hann_complete.shape[0] // 2:]
    else:
        raise ValueError("Unknown ramp type. Use 'linear' or 'raised-cosine'")

    if samples.ndim == 1:
        samples[start_point:] *= offramp_amps
    elif samples.ndim == 2:
        samples[start_point:, 0] *= offramp_amps
        samples[start_point:, 1] *= offramp_amps

    return samples


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """This helper function normalizes audio based on the absolute max amplitude from the provided sound samples."""
    samples = samples / np.max(np.abs(samples), axis=0)
    return samples


def overlay_samples(samples_arrays: np.typing.ArrayLike) -> np.ndarray:
    """Overlay all the samples in the iterable"""

    # Find longest stimulus, which will be used as the 'background'
    n_frames = np.max([len(obj) for obj in samples_arrays])
    # Make empty sound
    output = np.zeros(n_frames)
    # Overlay all samples
    for samples in samples_arrays:
        output[:len(samples)] += samples

    # Check whether to normalize and return
    if np.max(np.abs(output)) > 1:
        warnings.warn(thebeat._warnings.normalization)
        return thebeat.helpers.normalize_audio(output)
    else:
        return output


@requires_lilypond
def plot_lp(lp,
            filepath: os.PathLike | str = None,
            suppress_display: bool = False,
            title: str | None = None,
            dpi: int = 300,
            ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """
    This function plots a LilyPond object.

    Parameters
    ----------
    lp
        A LilyPond string.
    filepath
        If provided, the plot will be saved to this path. Has to end with either .png or .pdf.
    suppress_display
        If True, the plot will not be displayed using :func:`matplotlib.figure.Figure.show`.
    title
        Title of the plot.
    dpi
        The resolution of the plot in dots per inch.
    ax
        If provided, the plot will be drawn on this axis.

    """
    # If a filepath is given, we'll use its format. If none is given, we'll use .png as the format to
    # eventually show.
    if filepath:
        save_format = os.path.splitext(filepath)[1]
        if save_format not in ('.pdf', '.png', '.eps'):
            raise ValueError("Can only export .png, .pdf, and .eps files.")
    else:
        save_format = None

    # lilypond needs to write some temporary files to disk. These include a .eps and .png file.
    # if we want to keep that file, we copy it from the temporary files.
    with tempfile.TemporaryDirectory() as tmp_dir:

        with open(os.path.join(tmp_dir, 'rhythm.ly'), 'w') as file:
            file.write(lp)

        if save_format == '.png' or save_format is None:
            command = ['lilypond', '-dbackend=eps', '--silent', f'-dresolution={dpi}', '--png', '-o',
                       'rhythm', 'rhythm.ly']
        else:
            command = ['lilypond', '-dbackend=eps', '--silent', f'-dresolution={dpi}', '--png', f'--{save_format[1:]}',
                       '-o', 'rhythm', 'rhythm.ly']

        subprocess.run(command, cwd=tmp_dir, check=True)

        # read the png as image
        result_path_png = os.path.join(tmp_dir, 'rhythm.png')
        image = mpimg.imread(result_path_png)

        # crop the png
        white = np.array([1, 1, 1])
        mask = np.abs(image - white).sum(axis=2) < 0.05
        coords = np.array(np.nonzero(~mask))
        top_left = np.min(coords, axis=1) + 1
        bottom_right = np.max(coords, axis=1) + 1
        img_cropped = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

        # the eps we cannot crop that easily unfortunately, so we use the one created
        # by lilypond if an .eps is desired.
        if filepath and save_format in ('.eps', '.pdf'):
            path_to_file_for_saving = os.path.join(tmp_dir, f'rhythm{save_format}')
            shutil.copy(path_to_file_for_saving, filepath)

    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)
        ax_provided = False
    else:
        fig = ax.get_figure()
        ax_provided = True
    ax.imshow(img_cropped)
    ax.set_axis_off()
    ax.set_title(title)

    # show plot if necessary
    if not suppress_display and not ax_provided:
        fig.show()

    # Save cropped .png if necessary
    if filepath and save_format == '.png':
        plt.imsave(filepath, img_cropped)

    return fig, ax


def plot_single_sequence(onsets: list | np.ndarray,
                         end_with_interval: bool,
                         final_ioi: float | None = None,
                         style: str = 'seaborn-v0_8',
                         title: str | None = None,
                         x_axis_label: str = "Time",
                         linewidths: list[float] | npt.NDArray[float] | float | None = None,
                         figsize: tuple | None = None,
                         dpi: int = 100,
                         suppress_display: bool = False,
                         ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """
    This function plots the onsets of a Sequence or SoundSequence object in an event plot.

    Parameters
    ----------
    onsets
        The onsets (i.e. t values) of the sequence.
    end_with_interval
        Indicates whether sequence ends with an interval (``True``) or event (``False``).
    final_ioi
        The final inter-onset interval of the sequence. This is only used if the sequence ends with an interval.
    linewidths
        The desired width of the bars (events). Defaults to 1/10th of the smallest inter-onset interval (IOI).
        Can be a single value that will be used for each onset, or a list or array of values
        (i.e with a value for each respective onsets).
    style
        Matplotlib style to use for the plot. Defaults to 'seaborn'.
        See `matplotlib style sheets reference
        <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    title
        If desired, one can provide a title for the plot. This takes precedence over using the
        name of the object as the title of the plot (if the object has one).
    x_axis_label
        A label for the x axis.
    figsize
        A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
        This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.subplots`.
    dpi
        The number of dots per inch. This refers to the ``dpi`` parameter in :class:`matplotlib.figure.Figure`.
    suppress_display
        If ``True``, the plot is only returned, and not displayed via :func:`matplotlib.pyplot.show`.
    ax
        If desired, you can provide an existing :class:`matplotlib.axes.Axes` object onto which to plot.
        See the Examples of the different plotting functions to see how to do this
        (e.g. :py:meth:`~thebeat.core.Sequence.plot_sequence` ).
        If an existing :class:`matplotlib.axes.Axes` object is supplied, this function returns the original
        :class:`matplotlib.figure.Figure` and :class:`matplotlib.axes.Axes` objects.

    """

    # Make onsets array
    onsets = np.array(onsets)

    # Get matplotlib default size
    if not figsize:
        cur_size = plt.rcParams["figure.figsize"]
        figsize = (cur_size[0], cur_size[1] / 2)

    # Make plot
    with plt.style.context(style):
        # If an existing Axes object was passed, do not create new Figure and Axes.
        # Else, get the current Figure object from the Axes object.
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True, dpi=dpi)
            ax_provided = False
        else:
            fig = ax.get_figure()
            ax_provided = True

        ax.axes.set_xlabel(x_axis_label)
        ax.set_ylim(0, 1)
        left_x_lim = min(onsets[0], 0)
        right_x_lim = onsets[-1] + final_ioi if end_with_interval else onsets[-1] + linewidths[-1]
        ax.set_xlim(left_x_lim, right_x_lim)
        ax.barh(0.5, width=linewidths, height=1.0, left=onsets)
        ax.axes.set_title(title)
        ax.axes.yaxis.set_visible(False)

    # Show plot if desired, and if no existing Axes object was passed.
    if suppress_display is False and ax_provided is False:
        fig.show()

    return fig, ax


def plot_waveform(samples: np.ndarray,
                  fs: int,
                  n_channels: int,
                  style: str = 'seaborn-v0_8',
                  title: str | None = None,
                  figsize: tuple | None = None,
                  dpi: int = 100,
                  suppress_display: bool = False,
                  ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes]:
    """
    This helper function plots a waveform of a sound using matplotlib.

    Parameters
    ----------
    samples
        The array containing the sound samples.
    fs
        Sampling frequency in hertz (e.g. 48000).
    n_channels
        Number of channels (1 for mono, 2 for stereo).
    style
        Style used by matplotlib. See `matplotlib style sheets reference
        <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
    title
        If desired, one can provide a title for the plot.
    figsize
        A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
        This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.figure`.
    dpi
        The resolution of the plot in dots per inch. This refers to the ``dpi`` parameter in
        :func:`matplotlib.pyplot.figure`.
    suppress_display
        If ``True``, :meth:`matplotlib.pyplot.Figure.show` is not run.
    ax
        If desired, you can provide an existing :class:`matplotlib.axes.Axes` object onto which to plot.
        See the Examples of the different plotting functions to see how to do this
        (e.g. :py:meth:`~thebeat.core.Sequence.plot_sequence` ).
        If an existing :class:`matplotlib.axes.Axes` object is supplied, this function returns the original
        :class:`matplotlib.figure.Figure` and :class:`matplotlib.axes.Axes` objects.
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
        x_right_lim = samples.shape[0] / fs
    else:
        frames = np.linspace(start=0,
                             stop=samples.shape[0] / fs * 1000,
                             num=samples.shape[0])
        x_label = "Time (ms)"
        x_right_lim = samples.shape[0] / fs * 1000

    # Plot
    with plt.style.context(style):
        # If an Axes object is provided, we use that one. Save whether that was done to know
        # if we need to return a newly created Figure and Axes.
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, tight_layout=True, dpi=dpi)
            ax_provided = False
        else:
            fig = ax.get_figure()
            ax_provided = True

        ax.set_xlim(0, x_right_lim)
        ax.plot(frames, samples, alpha=alph)
        if n_channels == 2:
            ax.legend(["Left channel", "Right channel"], loc=0, frameon=True)
        ax.set_ylabel("Amplitude")
        ax.set_xlabel(x_label)
        ax.set_title(title)

    if suppress_display is False and not ax_provided:
        fig.show()

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


def read_wav(filepath: str | os.PathLike,
             new_fs: int | None):
    """Internal function used to read a wave file. Returns the wave file's samples and the sampling frequency.
    If dtype is different from np.float64, it converts the samples to that."""
    file_fs, samples = scipy.io.wavfile.read(filepath)

    # Change dtype so we always have float64
    if samples.dtype == 'int16':
        samples = samples.astype(np.float64, order='C') / 2 ** 15
    elif samples.dtype == 'int32':
        samples = samples.astype(np.float64, order='C') / 2 ** 31
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
        samples, fs = thebeat.helpers.resample(samples, file_fs, new_fs)

    return samples, fs


def resample(samples, input_fs, output_fs):
    """Internal function used to resample sounds. Uses scipy.signal.resample"""
    if output_fs == input_fs:
        return samples, input_fs

    resample_factor = float(output_fs) / float(input_fs)
    resampled = scipy.signal.resample(samples, int(len(samples) * resample_factor))

    return resampled, output_fs


def rhythm_to_binary(rhythm: thebeat.music.Rhythm,
                     smallest_note_value: int = 16):
    """This helper function converts a rhythm to a binary representation."""

    n_positions = smallest_note_value / rhythm.time_signature[1] * rhythm.time_signature[0]
    if not n_positions.is_integer():
        raise ValueError("Something went wrong while making the rhythmic grid. Try supplying a different "
                         "'smallest_note_value'.")

    # Create empty zeros array
    signal = np.zeros(int(n_positions))

    # We multiply each fraction by the total length of the zeros array to get the respective positions
    # and add zero for the first onset
    indices = np.append(0, np.cumsum(rhythm.fractions) * n_positions)

    # Check if any of the indices are not integers
    if np.any(indices % 1 != 0):
        raise ValueError("The smallest_note_value that you provided is longer than the shortest note in the "
                         "rhythm. Please provide a shorter note value as the smallest_note_value (i.e. a larger "
                         "number).")

    for index, is_played in zip(indices, rhythm.is_played):
        if is_played:
            signal[int(index)] = 1

    return signal


def synthesize_sound(duration_ms: float,
                     fs: int,
                     freq: float,
                     n_channels: int,
                     amplitude: float,
                     oscillator: str) -> np.ndarray:
    # Get duration in s
    t = duration_ms / 1000

    # Check whether frame locations were rounded off
    n_samples = fs * t
    if not n_samples.is_integer():
        warnings.warn(thebeat._warnings.framerounding_soundsynthesis)
    n_samples = int(np.ceil(n_samples))

    samples = np.arange(n_samples, dtype=np.float64) / fs

    if n_channels == 2:
        empty = np.empty(shape=(n_samples, 2))
        empty[:, 0] = samples
        empty[:, 1] = samples
        samples = empty

    if oscillator == 'sine':
        samples = amplitude * np.sin(2 * np.pi * freq * samples)
    elif oscillator == 'square':
        samples = amplitude * scipy.signal.square(2 * np.pi * freq * samples)
    elif oscillator == 'sawtooth':
        samples = amplitude * scipy.signal.sawtooth(2 * np.pi * freq * samples)
    else:
        raise ValueError("Choose existing oscillator (for now only 'sine' or 'square')")

    return samples


def write_wav(samples: np.ndarray,
              fs: int,
              filepath: str | os.PathLike,
              dtype: str | np.dtype = np.int16,
              metronome: bool = False,
              metronome_ioi: float | None = None,
              metronome_amplitude: float | None = None) -> None:
    """
    This helper function writes the provided sound samples to disk as a wave file.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html for more info.
    """
    if metronome is True:
        samples = get_sound_with_metronome(samples, fs, metronome_ioi, metronome_amplitude)

    # Construct dtype object
    dtype = np.dtype(dtype)

    # Change dtypes if necessary (we default to int16)
    if dtype == np.int16:
        samples = np.clip(samples * (2 ** 15), -2**15, 2**15-1).astype(np.int16)
    elif dtype == np.int32:
        samples = np.clip(samples * (2 ** 31), -2**31, 2**31-1).astype(np.int32)
    elif dtype == np.float32:
        samples = samples.astype(np.float64)
    elif dtype == np.float64:
        pass
    else:
        raise ValueError("Unknown dtype for wav file. 'int16', 'int32', 'float32' and 'float64' are supported:'"
                         "https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html")

    filepath = str(filepath)

    if not filepath.endswith('.wav'):
        warnings.warn("File saved with extension other than .wav even though it is a .wav file.")

    wavfile.write(filename=filepath, rate=fs, data=samples)
