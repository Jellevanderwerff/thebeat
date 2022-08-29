import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample
import sounddevice as sd


def get_sound_with_metronome(samples, fs, metronome_ioi, metronome_amplitude):
    sound_samples = samples
    duration = sound_samples.shape[0] / fs * 1000

    n_metronome_clicks = int(duration // metronome_ioi)  # We want all the metronome clicks that fit in the seq.
    onsets = np.concatenate((np.array([0]), np.cumsum([metronome_ioi] * (n_metronome_clicks - 1))))

    metronome_path = os.path.join(sys.path[1], 'stimulus', 'resources', 'metronome.wav')
    metronome_fs, metronome_samples = wavfile.read(metronome_path)

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


def normalize_audio(samples):
    samples /= np.max(np.abs(samples), axis=0)
    return samples


def plot_waveform(samples, fs, n_channels, title):
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
    plt.ylabel("Amplitude")
    plt.xticks(ticks=[0, samples.shape[0]],
               labels=[0, int(samples.size / fs * 1000)])
    plt.xlabel("Time (ms)")
    plt.title(title)
    plt.show()


def play_samples(samples, fs, mean_ioi, loop, metronome, metronome_amplitude):
    if metronome is True:
        samples = get_sound_with_metronome(samples, fs, mean_ioi, metronome_amplitude=metronome_amplitude)
    else:
        samples = samples

    sd.play(samples, fs, loop=loop)
    sd.wait()


def write_wav(samples, fs, out_path, name, metronome, metronome_ioi, metronome_amplitude):
    if metronome is True:
        samples = get_sound_with_metronome(samples, fs, metronome_ioi, metronome_amplitude)
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
