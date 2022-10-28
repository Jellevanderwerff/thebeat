from scipy.fft import rfft, rfftfreq
from thebeat.helpers import make_binary_timeseries
from thebeat.core import Sequence
import thebeat
import matplotlib.pyplot as plt
import numpy as np


def fft_plot(seq: thebeat.core.Sequence,
             fs: int = 1000,
             max_freq: int = 20,
             style: str = 'seaborn'):
    timeseries = make_binary_timeseries(seq.onsets, resolution=1000 / fs)
    duration = max(seq.onsets) / 1000
    n = int(fs * duration) + 1

    yf = rfft(timeseries)[1:]
    xf = rfftfreq(n, d=1 / fs)[1:]

    with plt.style.context(style):
        fig, ax = plt.subplots()
        ax.plot(xf, np.abs(yf))
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Absolute power')
        ax.set_xlim((1, max_freq))
    fig.show()


def fft_plot_ms(seq: thebeat.core.Sequence,
                fs: int = 1000,
                resolution: int = 1,
                min_ioi: int = 20,
                style: str = 'seaborn'):
    if min_ioi < 0:
        raise ValueError("The minimum IOI value needs to be above zero.")

    max_freq = 1000 / min_ioi
    timeseries = make_binary_timeseries(seq.onsets, resolution=resolution)
    duration = max(seq.onsets) / 1000
    n = int(fs * duration) + 1

    yf = rfft(timeseries)[1:]
    xf = rfftfreq(n, d=1 / fs)[1:]

    with plt.style.context(style):
        fig, ax = plt.subplots()
        ax.plot(xf, np.abs(yf))
        ax.set_xlabel('IOI')
        ax.set_ylabel('Absolute power')
        ax.set_xlim((1, max_freq))
        ax.invert_xaxis()
        ax.xaxis.set_major_formatter(lambda x, pos: str(round(1000 / x if not x == 0 else 0, 1)))
    fig.show()


seq = Sequence.generate_random_normal(n=1000, mu=500, sigma=20)
fft_plot_ms(seq, min_ioi=200)
fft_plot(seq, max_freq=10)

