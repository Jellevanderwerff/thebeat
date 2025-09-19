# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Added tests for Apple Silicon support (#88).
- Added and tested support for Python 3.12, 3.13, and 3.14 (#91, #138).
- Added three functions to `utils` module (`concatenate_rhythms`, `sequence_to_binary`, `rhythm_to_binary`), which were previously only internal in `thebeat.helpers` (#136).

### Changed
- The optional dependencies for music notation are now named `[music-notation]` instead of `[music_notation]`, following Python packaging conventions (#93). Nothing changes for the user since `pip` already normalized those names.
- Fourier transform functions, such as `thebeat.stats.fft_plot()` now return power instead of amplitude, and have an explicit `remove_dc` parameter (#95)
- `matplotlib` styles are not automatically applied by all plotting functions, but are instead left to be manually applied by the user (#99). The new approach is illustrated in [the documentation](https://thebeat.readthedocs.io/en/stable/examples/tipstricks/manipulate_plots.html#Adding-a-style/theme-to-the-plot).
- Changed meaning of `note_value` naming throughout the Rhythm and Melody API: `note_values` are now `fractions.Fraction`s and correspond to the musical definitions of eighth/quarter/half/whole/... notes (#130). This provides more flexibility instead of the previous use (where integer denominators were used instead of actual fractions).
- Removed internal `thebeat.helpers` module from API reference in documentation (#136).
- Reviewed and changed all rounding behavior to be consistent when discretizing continuous sequences (i.e., when generating sound samples and in `sequence_to_binary`) (#140).

### Removed
- Remove `suppress_display` arguments from all plotting functions, in favor of explicitly calling `matplotlib` functions `fig.show()` or `plt.show()` by the user, increasing consistency between Python enviroments (i.e., interactive vs. non-interactive) (#106).
- Removed `Sequence.generate_random_poisson`, potentially generating invalid sequences (as the Poisson distribution is a discrete distribution); refer to `Sequence.generate_random_exponetial` instead to generate sequences with Poisson process-distributed IOIs (#108).
- Dropped support for end-of-life versions of Python, 3.8 and 3.9 (#124).
- Removed ambiguously defined `Rhythm.from_fractions` and `Rhythm.fractions` in favor of reworked `Rhythm.from_note_values` and `Rhythm.note_values` (#130)

### Fixed
- Fixed another bug in the nPVI calculation, which overestimated nPVI by a factor of 2.
- Fixed plot titles to use the name of the seqeuencey object by default, as originally intended (#67).
- Various small fixes related to updated dependencies (#88).
- Fixed a number of bugs in the Fourier transform functions, such as `thebeat.stats.fft_plot()` (#95).
- Fixed automatic ties between notes in `Rhythm.plot_rhythm()` and `Melody.plot_melody()` methods (#102). Note that the order of tied notes will not always align with the metre implied by the `Rhythm` or `Melody` objects`'s time signature.
- Fixed interpretation of `Sequence.generate_random_exponential`'s `lam` parameter, to match common interpretation of lambda parameters of exponential and Poisson distributions (#108).
- Fixed several incompatibilities with breaking changes in new versions of Abjad (#125).
- Fixed several bugs in `SoundStimulus.from_parselmouth` and `SoundStimulus.__init__` (#139).

## [0.2.0] - 2023-12-22
### Added
- `Sequence.from_binary_string()` (#32).
- Split off `stats.get_fft_values()` from `stats.fft_plot()` (#40).
- Added `SoundSequence.write_multichannel_wav()` to write multichannel wav files (#51).

### Fixed
- Fixed issue with Lilypond wrapper (#28).
- `stats.fft_plot()` does not discard first value anymore (#39).
- Fixed `visualization.plot_multiple_sequences` plotting multiple sequences with the same name on top of each other (#46).
- Fixed mistake in nPVI calculation, returning the correct value now.
- Fixed calculation of phase differences, now in `stats.get_phase_differences()` (#63).

### Changed
- Changed `Sequence.quantize()` to `Sequence.quantize_iois()`, now using IOIs instead of onsets (#33).
- Changed entropy calculation function to require a given 'resolution' instead of calculating the bins from the mean IOI (#10, #11).
- `sequence_to_binary()` returns an integer array.
- Various improvements to the documentation and examples (#55, #57).

## [0.1.0] - 2023-07-15
### Added
- Initial release with core classes Sequence, SoundStimulus, SoundSequence, Rhythm, and Melody.
- Several utility modules: linguistic, stats, visualization.
- First version of the online documentation.
- Basic CI setup.

[Unreleased]: https://github.com/Jellevanderwerff/thebeat/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Jellevanderwerff/thebeat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Jellevanderwerff/thebeat/releases/tag/v0.1.0
