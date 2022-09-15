import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample
import sounddevice as sd
from typing import Union


def get_sound_with_metronome(samples: np.ndarray,
                             fs: int,
                             metronome_ioi: int,
                             metronome_amplitude: float):
    """This helper function adds a metronome sound to a NumPy array of sound samples.
     It works for both mono and stereo sounds."""

    sound_samples = samples
    duration_s = sound_samples.shape[0] / fs * 1000

    n_metronome_clicks = int(duration_s // metronome_ioi)  # We want all the metronome clicks that fit in the seq.
    onsets = np.concatenate((np.array([0]), np.cumsum([metronome_ioi] * (n_metronome_clicks - 1))))

    metronome_path = os.path.join(sys.path[1], 'combio', 'resources', 'metronome.wav')
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


def normalize_audio(samples: np.ndarray):
    """This helper function normalizes audio based on the absolute max amplitude from the provided sound samples."""
    samples /= np.max(np.abs(samples), axis=0)
    return samples


def plot_waveform(samples: np.ndarray,
                  fs: int,
                  n_channels: int,
                  style: str = 'seaborn',
                  title: str = None,
                  figsize: tuple = None,
                  suppress_display: bool = False):
    """
    This helper function plots a waveform of a sound using matplotlib. It returns a fig and an ax object.

    Parameters
    ----------
    samples : ndarray
        The array containing the sound samples.
    fs : int
        Sampling frequency in hertz (e.g. 48000).
    n_channels : int
        Number of channels (1 for mono, 2 for stereo).
    style : str, optional
        Style used by matplotlib, defaults to 'seaborn'. See matplotlib docs for other styles.
    title : str, optional
        If desired, one can provide a title for the plot.
    figsize : tuple, optional
        A tuple containing the desired output size of the plot in inches.
    suppress_display : bool, optional
        If 'True', plt.show() is not run. Defaults to 'False'.


    Returns
    -------
    fig : a matplotlib Figure object
    ax : a matplotlib Axes object

    """
    frames = np.arange(samples.shape[0])

    # if we have two channels, we want the waveform to be opaque
    if n_channels == 1:
        alph = 1
    elif n_channels == 2:
        alph = 0.5
    else:
        raise ValueError("Unexpected number of channels.")

    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize, tight_layout=True)
        ax.plot(frames, samples, alpha=alph)
        if n_channels == 2:
            ax.legend(["Left channel", "Right channel"], loc=0, frameon=True)
        ax.set_ylabel("Amplitude")
        ax.set_xticks(ticks=[0, samples.shape[0]],
                      labels=[0, int(samples.size / fs * 1000)])
        ax.set_xlabel("Time (ms)")
        ax.set_title(title)

    if suppress_display is False:
        plt.show()

    return fig, ax


def play_samples(samples: np.ndarray,
                 fs: int,
                 mean_ioi: int,
                 loop: bool,
                 metronome: bool,
                 metronome_amplitude: float):
    """This helper function uses the sounddevice library to play a sound, either with or without metronome.
    It does not return anything."""
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
              metronome_amplitude: float):
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
