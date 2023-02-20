<img src=docs/source/thebeat_logo.png height="350" width="350">

**Note that *thebeat* is still under development. Please report any issues by clicking the 'Issues' tab in the respository.**

# *thebeat*: Rhythms in Python for Science

*thebeat* is a Python package for working with temporal sequences and rhythms in the behavioural and cognitive sciences. It provides functionality for creating stimuli, and for visualizing and analyzing temporal data.

*thebeat* will save you time when creating experiments or analyzing data. 
It is a collection of widely accepted methods for use in timing research. 
*thebeat* is an open-source, on-going, and collaborative project, 
integrating easily with the existing Python ecosystem, and with your own scripts. 
The package was specifically designed to be useful for both skilled and novice programmers.

# Documentation
The package documentation is available from [https://docs.the-beat.info](https://docs.the-beat.info).

# Installation

A development version of *thebeat* can be installed by downloading this repository as a ``.zip`` file (click the green 'Code' button > 'Download ZIP'), and installing via pip, by typing the following in a terminal where you've navigated to the folder containing the downloaded ``.zip`` file:

```bash
pip install thebeat-main.zip
```

# Getting started
The code below illustrates how we might create a simple trial for use in an experiment:

```python
from thebeat import Sequence, SoundStimulus, SoundSequence

seq = Sequence.generate_isochronous(n_events=10, ioi=500)
sound = SoundStimulus.generate(freq=440, duration_ms=50, onramp_ms=10, offramp_ms=10)
trial = SoundSequence(sound, seq)

trial.play()  # play sound over loudspeakers
trial.plot_waveform()  # plot as sound waveform
trial.plot_sequence()  # plot as an event plot
trial.write_wav('example_trial.wav')  # save file to disk
```

# License
*thebeat* is distributed under the GPL-3 license. You are free to distribute or modify the code, both for non-commercial and commercial use. See [here](https://choosealicense.com/licenses/gpl-3.0/) for more info.

# Collaborators
The package was developed by the Comparative Bioacoustics Group at the Max Planck Institute for Psycholinguistics.

The collaborators were: Jelle van der Werff, Andrea Ravignani, and Yannick Jadoul.

