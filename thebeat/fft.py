from scipy.fft import rfft, rfftfreq
from thebeat.helpers import make_binary_timeseries
from thebeat.core import Sequence
import thebeat
import matplotlib.pyplot as plt
import numpy as np


def fft_plot_hz(seq: thebeat.core.Sequence, resolution: int = 1, max_fs: int = 20):
    fs = 1000
    timeseries = make_binary_timeseries(seq.onsets, resolution=resolution)
    duration = max(seq.onsets) / 1000
    n = int(fs * duration) + 1

    yf = rfft(timeseries)
    xf = rfftfreq(n, d=1 / fs)

    fig, ax = plt.subplots()
    ax.plot(xf, np.abs(yf))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_xlim((0, max_fs))
    fig.show()


seq = Sequence.generate_random_normal(n=1000, mu=500, sigma=20)
fft_plot_hz(seq, resolution=1)


