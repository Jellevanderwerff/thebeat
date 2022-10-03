import os
from typing import Union, Optional
import numpy as np
from collections import namedtuple

from _pytest import warnings

try:
    import mingus
    import mingus.core.scales
except ImportError:
    mingus = None
from _decorators import requires_lilypond
import combio.rhythm
import combio._helpers
import sounddevice
import numpy.typing as npt


class Melody(combio.rhythm.Rhythm):

    def __init__(self,
                 rhythm: combio.rhythm.Rhythm,
                 tone_heights,
                 event_durations: Optional[Union[npt.NDArray[int], list[int], int]] = None,
                 is_played: Optional[Union[list[bool], npt.NDArray[bool]]] = None,
                 name: Optional[str] = None):

        # Initialize namedtuple. The namedtuple template is saved as an attribute.
        self.Event = namedtuple('event', 'layer onset_ms duration_ms tone_height_hz is_played')

        # Make is_played if None supplied
        if is_played is None:
            is_played = np.tile(True, len(rhythm.onsets))

        # Set event durations to the IOIs if no event durations were supplied (i.e. full length notes)
        if event_durations is None:
            event_durations = rhythm.iois

        # If a single integer is passed, use that value for all the events
        if isinstance(event_durations, int):
            event_durations = np.tile(event_durations, len(rhythm.onsets))

        # Add initial events
        self.events = self._make_events(layer_id=np.tile(0, len(rhythm.onsets)),
                                        rhythm=rhythm,
                                        event_durations=event_durations,
                                        tone_heights=tone_heights,
                                        is_played=is_played)

        # Save attributes
        self.n_layers = 1  # Currently, we have one layer at index 0
        self.name = name

        # Call Rhythm constructor
        super().__init__(iois=rhythm.iois,
                         n_bars=rhythm.n_bars,
                         time_signature=rhythm.time_signature,
                         beat_ms=rhythm.beat_ms)

    @classmethod
    def generate_random_melody(cls,
                               key: str = 'C',
                               n_rests: Optional[int] = None,
                               rng: np.random.Generator = None,
                               name: Optional[str] = None):
        if mingus is None:
            raise ImportError("This method requires the 'mingus' Python package."
                              "Install it, for instance by typing 'pip install mingus' into your terminal.")

        if rng is None:
            rng = np.random.default_rng()

        # Generate random rhythm and tone_heights
        rhythm = combio.rhythm.Rhythm.generate_random_rhythm([4, 8, 16])
        possible_notes = mingus.core.scales.Major(key, octaves=2).ascending()
        tone_heights = rng.choice(possible_notes, size=len(rhythm.onsets))

        # Do a random number of rests if none is given
        if n_rests is None:
            n_rests = rng.integers(low=0, high=len(rhythm.onsets))
        else:
            n_rests = n_rests

        is_played = n_rests * [False] + (len(rhythm.onsets) - n_rests) * [True]

        return cls(rhythm=rhythm, tone_heights=tone_heights, is_played=is_played, name=name)

    def synthesize_and_play(self,
                            fs: int = 48000,
                            amplitude: float = 1.0,
                            oscillator: str = 'sine',
                            onramp: int = 0,
                            offramp: int = 0,
                            ramp_type: str = 'linear',
                            metronome: bool = False,
                            metronome_amplitude: float = 1.0):

        samples = self._make_sound(self.events, fs=fs, oscillator=oscillator, amplitude=amplitude, onramp=onramp,
                                   offramp=offramp, ramp_type=ramp_type)

        if metronome is True:
            samples = combio._helpers.get_sound_with_metronome(samples=samples, fs=fs, metronome_ioi=self.beat_ms,
                                                               metronome_amplitude=metronome_amplitude)

        sounddevice.play(samples, samplerate=fs)
        sounddevice.wait()

    def synthesize_and_write(self,
                             filepath: Union[str, os.PathLike],
                             fs: int = 48000,
                             amplitude: float = 1.0,
                             oscillator: str = 'sine',
                             onramp: int = 0,
                             offramp: int = 0,
                             ramp_type: str = 'linear',
                             metronome: bool = False,
                             metronome_amplitude: float = 1.0):

        samples = self._make_sound(self.events, fs=fs, oscillator=oscillator, amplitude=amplitude, onramp=onramp,
                                   offramp=offramp, ramp_type=ramp_type)

        if metronome is True:
            samples = combio._helpers.get_sound_with_metronome(samples=samples, fs=fs, metronome_ioi=self.beat_ms,
                                                               metronome_amplitude=metronome_amplitude)

        combio._helpers.write_wav(samples=samples, fs=fs, filepath=filepath, metronome=metronome,
                                  metronome_ioi=self.beat_ms, metronome_amplitude=metronome_amplitude)

    def _make_events(self, layer_id, rhythm, event_durations, tone_heights, is_played) -> list:

        events = []

        for event in zip(layer_id, rhythm.onsets, event_durations, tone_heights, is_played):
            entry = self.Event(event[0], event[1], event[2], event[3], event[4])
            events.append(entry)

        return events

    def _make_sound(self,
                    events: list,
                    fs: int,
                    oscillator: str,
                    amplitude: float,
                    onramp: int,
                    offramp: int,
                    ramp_type: str):

        # Generate an array of silence that has the size of all the onsets + one final stimulus.
        # Calculate number of frames
        total_duration_ms = np.sum(self.iois)
        n_frames = total_duration_ms / 1000 * fs

        # Avoid rounding issues
        if not n_frames.is_integer():
            warnings.warn("Number of frames was rounded off to nearest integer ceiling. "
                          "This shouldn't be much of a problem.")
        n_frames = int(np.ceil(n_frames))

        # todo Allow stereo (but also in the other functions)

        # Create empty 1-D array
        samples = np.zeros(n_frames)

        # Loop over the events, synthesize event sound, and add all of them to the samples array at the appropriate
        # times.
        for event in events:
            if event.is_played is True:
                event_samples = combio._helpers.synthesize_sound(duration_ms=event.duration_ms, fs=fs,
                                                                 freq=event.tone_height_hz, amplitude=amplitude,
                                                                 osc=oscillator)
                if onramp or offramp:
                    event_samples = combio._helpers.make_ramps(samples=event_samples, fs=fs, onramp=onramp,
                                                               offramp=offramp, ramp_type=ramp_type)

                start_pos = int(event.onset_ms / 1000 * fs)
                end_pos = int(start_pos + event_samples.shape[0])

                samples[start_pos:end_pos] = samples[start_pos:end_pos] + event_samples

            else:
                pass

        if np.max(samples) > 1:
            warnings.warn("Sound was normalized to avoid distortion. If undesirable, change amplitude of the sounds.")
            return combio._helpers.normalize_audio(samples)

        return samples


@requires_lilypond
def _get_lp_from_events(events,
                        n_layers: int,
                        time_signature: tuple,
                        print_staff: bool = True,
                        stem_directions=None):

    # Check whether mingus is installed
    if mingus is None:
        raise ImportError("This function or method requires the mingus package. Install, for instance by typing"
                          " 'pip install mingus' in your terminal.")

    # If custom stem directions are provided (i.e. for each layer in which direction the stems of the notes
    # are, make a list that we'll later use for each layer. Otherwise use bogus setting.
    if stem_directions is None:
        stem_directions = ['', '', '']
    else:
        stem_directions = ['\override Stem.direction = #' + stem_direction for stem_direction in stem_directions]

    if n_layers > 3:
        raise ValueError("Can maximally plot three layers.")

    if print_staff is True:
        print_staff_lp = ""
    else:
        print_staff_lp = "\\stopStaff \override Staff.Clef.color = #white "

    layers_list = []

    for layer in range(n_layers):
        bars = []
        events = [event for event in events if event.layer == layer]

        bar = ''
        b = mingus.containers.Bar(meter=time_signature)

        for event in events:
            note_value = event.note_value
            b.place_rest(note_value)  # This is only to keep track of the number of notes in a bar
            if event.is_played is True:
                note = str(note_value) + ' '
            else:
                note = 's' + str(note_value) + ' '
            bar += note
            if b.current_beat == b.length:
                bars.append("{ " + bar + "}\n")
                b = Bar(meter=time_signature)
                bar = ''

        layers_list.append(bars)

    voice_names = ['voiceOne', 'voiceTwo', 'voiceThree']
    layer_names = ['uno', 'dos', 'tres']

    string_firstbit = ''
    string_secondbit = '\\new DrumStaff << '

    for layer_i in range(len(layers_list)):
        bars = ' '.join(layers_list[layer_i])
        bars = print_staff_lp + bars
        layer_string = f"{layer_names[layer_i]} = \drummode {{ {stem_directions[layer_i]} {bars} }} \n"
        string_firstbit += layer_string
        staves_string = "\\new DrumVoice { \\%s \\%s }\n" % (voice_names[layer_i], layer_names[layer_i])
        string_secondbit += staves_string

    string_secondbit += ' >>'

    out_string = string_firstbit + string_secondbit

    remove_footers = """\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""
oddFooterMarkup = ""\nevenFooterMarkup = ""\n} """

    lp = '\\version "2.10.33"\n' + out_string + remove_footers

    return lp
