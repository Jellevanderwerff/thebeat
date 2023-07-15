[![PyPI version](https://img.shields.io/pypi/v/thebeat.svg)](https://pypi.python.org/pypi/thebeat)
[![PyPI Python versions](https://img.shields.io/pypi/pyversions/thebeat.svg)](https://pypi.python.org/pypi/thebeat)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Jellevanderwerff/thebeat/main.svg)](https://results.pre-commit.ci/latest/github/Jellevanderwerff/thebeat/main)
[![GitHub CI status](https://github.com/jellevanderwerff/thebeat/actions/workflows/ci.yml/badge.svg)](https://github.com/Jellevanderwerff/thebeat)
[![License](https://img.shields.io/pypi/l/praat-parselmouth.svg)](https://github.com/YannickJadoul/Parselmouth/blob/master/LICENSE)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Jellevanderwerff/thebeat/HEAD?labpath=docs%2Fsource%2Fgettingstarted.ipynb)


<img src="https://raw.githubusercontent.com/Jellevanderwerff/thebeat/main/docs/source/thebeat_logo.png" height="350" width="350">

# *thebeat*: Rhythms in Python for Science

*thebeat* is a Python package for working with temporal sequences and rhythms in the behavioural and cognitive sciences. It provides functionality for creating stimuli, and for visualizing and analyzing temporal data.

As a collection of accepted methods for use in music and timing research,
*thebeat* will save you time when creating experiments or analyzing data.

*thebeat* is an open-source, on-going, and collaborative project,
integrating easily with the existing Python ecosystem, and with your own scripts.
The package was specifically designed to be useful for both skilled and novice programmers.

# Documentation
The package documentation is available from
[https://thebeat.readthedocs.io](https://thebeat.readthedocs.io). The documentation contains
detailed descriptions of all package functionality, as well as a large number of (copyable)
examples.

# Installation
*thebeat* is available through PyPI, and can be installed using:

```bash
pip install thebeat
```

Note that if you want use *thebeat*'s functionality for plotting musical notation,
you have to install it using:

```bash
pip install thebeat[music_notation]
```

This will install *thebeat* with the optional dependencies [abjad](https://abjad.github.io)
and [Lilypond](https://lilypond.org).

*thebeat* is actively tested on Linux, macOS, and Windows. We aim to provide support for all
[supported versions](https://devguide.python.org/versions/) of Python (3.8 and higher).

# Try directly via Binder
If you first would like to try *thebeat*, or of you wish to use it in, for instance, an
educational setting, you can use
[this link](https://mybinder.org/v2/gh/Jellevanderwerff/thebeat/HEAD?labpath=docs%2Fsource%2Fgettingstarted.ipynb)
to try *thebeat* in a Binder environment.


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

# Open discussion
One of the reasons for creating *thebeat* was the lack of a collection of standardized/accepted
methods for use in rhythm and timing research. Therefore, an important part of *thebeat*'s merit
lies in opening discussions about the methods that are included. As an example, there are different
ways of calculating phase differences and integer ratios, and we imagine people to have different
opinions about which method to use. Where possible, we have included
references to the literature in the package documentation. But, we encourage anyone with an opinion
to openly question the methods that *thebeat* provides.

There are two places where you can go with comments and/or questions:

- You can click the 'Issues' tab at the top of this GitHub page, and start a thread. Note that
this place is mostly for questioning methods, or for reporting bugs.
- You can drop by in our [Gitter chatroom](https://app.gitter.im/#/room/#thebeat:gitter.im). This is likely
the best place to go to with questions about how *thebeat* works.

# License
*thebeat* is distributed under the GPL-3 license. You are free to distribute or modify the code, both for non-commercial and commercial use. See [here](https://choosealicense.com/licenses/gpl-3.0/) for more info.

# Collaborators
The package was developed by the Comparative Bioacoustics Group at the Max Planck Institute for
Psycholinguistics, in Nijmegen, the Netherlands.

The collaborators were: Jelle van der Werff, Andrea Ravignani, and Yannick Jadoul.
