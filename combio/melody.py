import os
import re
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import warnings

try:
    import abjad
except ImportError:
    abjad = None
from combio._decorators import requires_lilypond
import combio._warnings
import combio.rhythm
import combio._helpers
import sounddevice
import numpy.typing as npt


class Melody(combio.core.sequence.BaseSequence):

    def __init__(self,
                 rhythm: combio.rhythm.Rhythm,
                 pitch_hz: Union[npt.NDArray[Union[int, float]], list[Union[int, float]]],
                 is_played: Optional[list] = None,
                 name: Optional[str] = None):
        # Initialize namedtuple. The namedtuple template is saved as an attribute.
        self.Event = namedtuple('event', 'onset_ms duration_ms note_value tone_height_hz is_played')

        # Make is_played if None supplied
        if is_played is None:
            is_played = [True] * len(rhythm.onsets)

        # Add initial events
        self.events = self._make_events(rhythm=rhythm,
                                        iois=rhythm.iois,
                                        note_values=rhythm.get_note_values,
                                        pitch_hz=pitch_hz,
                                        is_played=is_played)

        # Save rhythm attributes
        self.time_signature = rhythm.time_signature  # Used for metrical sequences
        self.beat_ms = rhythm.beat_ms  # Used for metrical sequences

        # Check whether the provided IOIs result in a sequence only containing whole bars
        n_bars = np.sum(rhythm.iois) / self.time_signature[0] / self.beat_ms
        if not n_bars.is_integer():
            raise ValueError("The provided inter-onset intervals do not amount to whole bars.")
        # Save number of bars as an attribute
        self.n_bars = n_bars

        # Call BaseSequence constructor
        combio.core.sequence.BaseSequence.__init__(self, iois=rhythm.iois, metrical=True, name=name)

    @classmethod
    def from_hertz(cls):
        pass

    @classmethod
    def from_notes(cls,
                   rhythm: combio.rhythm.Rhythm,
                   notes: Union[npt.NDArray[str], list[str], str],
                   octave: int = None,
                   is_played: Optional[Union[npt.NDArray[bool], list[bool]]] = None,
                   name: Optional[str] = None):

        if isinstance(notes, str):
            notes = re.split(r"([A-Z|a-z][0-9]?)", notes)
            notes = list(filter(None, notes))

        pitch_hz = [abjad.NamedPitch(note, octave=octave).hertz for note in notes]

        return cls(rhythm, pitch_hz=pitch_hz, is_played=is_played, name=name)

    @classmethod
    def generate_random_melody(cls,
                               n_bars: int = 1,
                               beat_ms: int = 500,
                               time_signature: tuple = (4, 4),
                               key: str = 'C',
                               octave: int = 4,
                               n_rests: int = 0,
                               allowed_note_values: list = None,
                               rng: np.random.Generator = None,
                               name: Optional[str] = None):
        if abjad is None:
            raise ImportError("This method requires the 'mingus' Python package."
                              "Install it, for instance by typing 'pip install mingus' into your terminal.")

        if rng is None:
            rng = np.random.default_rng()

        if allowed_note_values is None:
            allowed_note_values = [4, 8, 16]

        # Generate random rhythm and random tone_heights
        rhythm = combio.rhythm.Rhythm.generate_random_rhythm(allowed_note_values=allowed_note_values, n_bars=n_bars,
                                                             time_signature=time_signature, beat_ms=beat_ms, rng=rng)
        pitches = [pitch.hertz for pitch in combio._helpers.get_major_scale(tonic=key, octave=octave)]

        pitch_hz = list(rng.choice(pitches, size=len(rhythm.onsets)))

        if n_rests > len(rhythm.onsets):
            raise ValueError("The provided number of rests is higher than the number of sounds.")

        # Make the rests and shuffle
        is_played = n_rests * [False] + (len(rhythm.onsets) - n_rests) * [True]
        rng.shuffle(is_played)

        return cls(rhythm=rhythm, pitch_hz=pitch_hz, is_played=is_played, name=name)

    @property
    def get_note_values(self):
        """
        Get note values from the IOIs, based on beat_ms.
        """

        # todo check this, I don't understand what the '4' means.

        ratios = self.iois / self.beat_ms / 4

        note_values = np.array([int(1 // ratio) for ratio in ratios])

        return note_values

    @requires_lilypond
    def plot_melody(self,
                    filepath: Optional[Union[os.PathLike, str]] = None,
                    key: str = 'C',
                    print_staff: bool = False,
                    suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:

        lp = self._get_lp_from_events(time_signature=self.time_signature, key=key, print_staff=print_staff)

        fig, ax = combio._helpers.plot_lp(lp, filepath=filepath, suppress_display=suppress_display)

        return fig, ax

    def synthesize_and_return(self,
                              event_durations: Optional[Union[np.ndarray, list]] = None,
                              fs: int = 48000,
                              amplitude: float = 1.0,
                              oscillator: str = 'sine',
                              onramp: int = 0,
                              offramp: int = 0,
                              ramp_type: str = 'linear',
                              metronome: bool = False,
                              metronome_amplitude: float = 1.0):
        """Here people can supply an event duration, which can differ from the IOIs (otherwise you may get one long
        sound)."""

        samples = self._make_melody_sound(fs=fs, oscillator=oscillator, amplitude=amplitude, onramp=onramp,
                                          offramp=offramp, ramp_type=ramp_type, event_durations=event_durations)

        if metronome is True:
            samples = combio._helpers.get_sound_with_metronome(samples=samples, fs=fs, metronome_ioi=self.beat_ms,
                                                               metronome_amplitude=metronome_amplitude)

        return samples, fs

    def synthesize_and_play(self,
                            event_durations: Optional[Union[np.ndarray, list]] = None,
                            fs: int = 48000,
                            amplitude: float = 1.0,
                            oscillator: str = 'sine',
                            onramp: int = 0,
                            offramp: int = 0,
                            ramp_type: str = 'linear',
                            metronome: bool = False,
                            metronome_amplitude: float = 1.0):
        """Here people can supply an event duration, which can differ from the IOIs (otherwise you may get one long
        sound)."""

        samples, _ = self.synthesize_and_return(event_durations=event_durations, fs=fs, amplitude=amplitude,
                                                oscillator=oscillator, onramp=onramp, offramp=offramp,
                                                ramp_type=ramp_type, metronome=metronome,
                                                metronome_amplitude=metronome_amplitude)

        sounddevice.play(samples, samplerate=fs)
        sounddevice.wait()

    def synthesize_and_write(self,
                             filepath: Union[str, os.PathLike],
                             event_durations: Optional[Union[np.ndarray, list]] = None,
                             fs: int = 48000,
                             amplitude: float = 1.0,
                             oscillator: str = 'sine',
                             onramp: int = 0,
                             offramp: int = 0,
                             ramp_type: str = 'linear',
                             metronome: bool = False,
                             metronome_amplitude: float = 1.0):
        """Here people can supply an event duration, which can differ from the IOIs (otherwise you may get one long
        sound)."""

        samples, _ = self.synthesize_and_return(event_durations=event_durations, fs=fs, amplitude=amplitude,
                                                oscillator=oscillator, onramp=onramp, offramp=offramp,
                                                ramp_type=ramp_type, metronome=metronome,
                                                metronome_amplitude=metronome_amplitude)

        if metronome is True:
            samples = combio._helpers.get_sound_with_metronome(samples=samples, fs=fs, metronome_ioi=self.beat_ms,
                                                               metronome_amplitude=metronome_amplitude)

        combio._helpers.write_wav(samples=samples, fs=fs, filepath=filepath, metronome=metronome,
                                  metronome_ioi=self.beat_ms, metronome_amplitude=metronome_amplitude)

    def _make_events(self,
                     rhythm,
                     iois,
                     note_values,
                     pitch_hz,
                     is_played) -> list:
        events = []

        for event in zip(rhythm.onsets, iois, note_values, pitch_hz, is_played):
            entry = self.Event(event[0], event[1], event[2], event[3], event[4])
            events.append(entry)

        return events

    def _make_melody_sound(self,
                           fs: int,
                           oscillator: str,
                           amplitude: float,
                           onramp: int,
                           offramp: int,
                           ramp_type: str,
                           event_durations: Optional[Union[list, np.ndarray]] = None):
        # todo Allow stereo (but also in the other functions)

        # Calculate required number of frames
        total_duration_ms = np.sum(self.iois)
        n_frames = total_duration_ms / 1000 * fs

        # Avoid rounding issues
        if not n_frames.is_integer():
            warnings.warn(combio._warnings.framerounding)
        n_frames = int(np.ceil(n_frames))

        # Create empty array with length n_frames
        samples = np.zeros(n_frames)

        # Set event durations to the IOIs if no event durations were supplied (i.e. use full length notes)
        if event_durations is None:
            event_durations = self.iois
        # If a single integer is passed, use that value for all the events
        elif isinstance(event_durations, int):
            event_durations = np.tile(event_durations, len(self.events))

        # Loop over the events, synthesize event sound, and add all of them to the samples array at the appropriate
        # times.
        for event, duration_ms in zip(self.events, event_durations):
            if event.is_played is True:
                event_samples = combio._helpers.synthesize_sound(duration_ms=duration_ms, fs=fs,
                                                                 freq=event.tone_height_hz, amplitude=amplitude,
                                                                 osc=oscillator)
                if onramp or offramp:
                    event_samples = combio._helpers.make_ramps(samples=event_samples, fs=fs, onramp=onramp,
                                                               offramp=offramp, ramp_type=ramp_type)

                # Calculate start- and end locations for inserting the event into the output array
                # and warn if the location in terms of frames was rounded off.
                start_pos = event.onset_ms / 1000 * fs
                end_pos = start_pos + event_samples.shape[0]
                if not start_pos.is_integer() or not end_pos.is_integer():
                    warnings.warn(combio._warnings.framerounding)
                start_pos = int(np.ceil(start_pos))
                end_pos = int(np.ceil(end_pos))

                # Add event samples to output array
                samples[start_pos:end_pos] = samples[start_pos:end_pos] + event_samples

            else:
                pass

        if np.max(samples) > 1:
            warnings.warn(combio._warnings.normalization)
            samples = combio._helpers.normalize_audio(samples)

        return samples

    def _get_lp_from_events(self,
                            key: str,
                            time_signature: tuple,
                            print_staff: bool = True):

        # Set up what we need
        note_maker = abjad.makers.NoteMaker()
        time_signature = abjad.TimeSignature(time_signature)
        key = abjad.KeySignature(key)
        remove_footers = """\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""
        oddFooterMarkup = ""\nevenFooterMarkup = ""\n} """
        remove_staff = """\stopStaff \override Staff.Clef.color = #white'"""

        tone_heights = [event.tone_height_hz for event in self.events]
        note_values = [event.note_value for event in self.events]
        is_played = [event.is_played for event in self.events]

        notes = []

        for tone_height, note_value, is_played in zip(tone_heights, note_values, is_played):
            duration = abjad.Duration((1, int(note_value)))
            if is_played is True:
                pitch = abjad.NamedPitch.from_hertz(tone_height)
                note = note_maker(pitch, duration)
            else:
                note = abjad.Rest(duration)
            notes.append(note)

        voice = abjad.Voice(notes)
        abjad.attach(time_signature, voice[0])
        abjad.attach(key, voice[0])

        staff = abjad.Staff([voice])
        score = abjad.Score([staff])
        score_lp = abjad.lilypond(score)

        if print_staff is False:
            lp = [remove_staff, remove_footers, score_lp]
        else:
            lp = [remove_footers, score_lp]

        lpf = abjad.LilyPondFile(lp)
        lpf_str = abjad.lilypond(lpf)

        return lpf_str
