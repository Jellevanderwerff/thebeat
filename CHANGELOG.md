# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - DATE
### Added
TENTATIVE - Support for Apple Silicon has been added for all functions, and we have included it in our test suite. This did not use to be the case because of compatibility issues with Lilypond [(#125)](https://github.com/Jellevanderwerff/thebeat/pull/125).

### Fixed
- Notes are now tied (mostly) correctly in the ``thebeat.music.Rhythm.plot_rhythm()`` and ``thebeat.music.Melody.plot_melody()`` functions. There are some situations that still will not be handled correctly, e.g. the note sequence ``[1/16, 1/16, 1/16, 5/16]`` is plotted with first a quarternote that is tied to a sixteenth note (rather than a sixteenth note tied to a quarternote) [(#102)](https://github.com/Jellevanderwerff/thebeat/pull/102).
- We have fixed some bugs that were due to an update in one of our dependencies (Abjad). These had the effect that the Rhythm and Melody classes were not working if one installed the latest version of Abjad. [(#125)](https://github.com/Jellevanderwerff/thebeat/pull/125)
- The optional depencies are now named ``[music-notation]`` instead of ``[music_notation]``. Nothing changes for the user since ``pip`` already normalized those names [(#93)](https://github.com/Jellevanderwerff/thebeat/pull/124).
- A number of bugs have been fixed in the Fourier transform functions (such as ``thebeat.stats.fft_plot()``). They now also return power instead of amplitude (i.e. amplitude squared). Also an explicit DC ``remove_dc`` parameter has been added to the functions [(#95)](https://github.com/Jellevanderwerff/thebeat/pull/95).
- Small bug w.r. to plot titles has been fixed [(#67)](https://github.com/Jellevanderwerff/thebeat/pull/67).

### Removed
- Support for Python 3.8 and 3.9 has been dropped [(#124)](https://github.com/Jellevanderwerff/thebeat/pull/124).
- We removed the function ``Sequence.generate_random_poisson()``, seeing as actually it is an exponential distribution that underlied this function (a Poisson distribution of onsets results in an exponential distribution of IOIs; we sampled IOIs). The ``Sequence.generate_random_exponential()`` has been updated accordingly [(#108)](https://github.com/Jellevanderwerff/thebeat/pull/108).
- Previously, one would have to specifically provide ``suppress_display=True`` to all plotting functions to hide plotting output. This functionality has now been reconsidered, and it is now necessary to explicitly call ``matplotlib`` functions ``fig.show()`` or ``plt.show()``. This was done such because the previous behaviour did not work similarly in all environments (i.e. interactive vs. non-interactive) [(#102)](https://github.com/Jellevanderwerff/thebeat/pull/102).
- ``matplotlib`` styles are now not forced on the user in the plotting functions. This has the effect that by default plots will use the standard ``matplotlib`` style. Instructions have been added to [the documentation](https://thebeat.readthedocs.io/en/stable/examples/tipstricks/manipulate_plots.html#Adding-a-style/theme-to-the-plot) on how to use styles [(#99)](https://github.com/Jellevanderwerff/thebeat/pull/99).

## [0.2.0] - 2023-12-22
### Added
- `Sequence.from_binary_string()` (#32)
- Split off `stats.get_fft_values()` from `stats.fft_plot()` (#40)
- Added `SoundSequence.write_multichannel_wav()` to write multichannel wav files (#51)

### Fixed
- Fixed issue with Lilypond wrapper (#28)
- `stats.fft_plot()` does not discard first value anymore (#39)
- Fixed `visualization.plot_multiple_sequences` plotting multiple sequences with the same name on top of each other (#46)
- Fixed mistake in nPVI calculation, returning the correct value now
- Fixed calculation of phase differences, now in `stats.get_phase_differences()` (#63)

### Changed
- Changed `Sequence.quantize()` to `Sequence.quantize_iois()`, now using IOIs instead of onsets (#33)
- Changed entropy calculation function to require a given 'resolution' instead of calculating the bins from the mean IOI (#10, #11)
- `sequence_to_binary()` returns an integer array
- Various improvements to the documentation and examples (#55, #57)

## [0.1.0] - 2023-07-15
### Added
- Initial release with core classes Sequence, SoundStimulus, SoundSequence, Rhythm, and Melody
- Several utility modules: linguistic, stats, visualization
- First version of the online documentation
- Basic CI setup

[Unreleased]: https://github.com/Jellevanderwerff/thebeat/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Jellevanderwerff/thebeat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Jellevanderwerff/thebeat/releases/tag/v0.1.0
