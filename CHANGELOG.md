# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- Fixed interpretation of `Sequence.generate_random_exponential`'s `lam` parameter, to match common interpretation of lambda parameters of exponential and Poisson distributions (#108)

### Removed
- Removed `Sequence.generate_random_poisson`, potentially generating invalid sequences; refer to `Sequence.generate_random_exponetial` instead (#108)

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
