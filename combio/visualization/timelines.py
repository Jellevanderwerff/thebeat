from combio.core import Sequence, StimTrial, Stimulus
import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional


def event_plot(sequence: Union[Sequence, StimTrial],
               style: str = 'seaborn',
               linewidth=10):

    # Input validation and setting line widths
    if isinstance(sequence, Sequence):
        linewidths = linewidth
    elif isinstance(sequence, StimTrial):
        linewidths = sequence.event_durations
    else:
        raise ValueError("Pass either a Sequence or a StimTrial option as the first argument.")

    # Make plot
    with plt.style.context(style):
        fig, ax = plt.subplots()
        ax.set_ylim(0, 1)
        ax.barh(0.5, width=linewidths, height=0.3, left=sequence.onsets)
        ax.axes.yaxis.set_visible(False)
        ax.axes.set_xlabel("Time (ms)")

    # Show plot
    plt.show()

    # Additionally return plot
    return fig, ax


seq = Sequence([500, 200, 1000])
stims = [Stimulus.generate(duration=150),
         Stimulus.generate(duration=10),
         Stimulus.generate(duration=200),
         Stimulus.generate(duration=200)]
trial = StimTrial(stims, seq)
event_plot(trial, style='seaborn')

