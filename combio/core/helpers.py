import numpy as np
import matplotlib.pyplot as plt


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
    plt.ylabel("Amplitude")
    plt.xticks(ticks=[0, samples.shape[0]],
               labels=[0, int(samples.size / fs * 1000)])
    plt.xlabel("Time (ms)")
    plt.title(title)
    plt.show()
