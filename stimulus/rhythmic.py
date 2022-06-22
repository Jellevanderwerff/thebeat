from mingus.containers import Bar, Track, Note
import numpy as np
from stimulus.base import BaseSequence, Stimuli, _plot_lp, _plot_waveform, _normalize_audio, _play_samples, _write_wav
import random
from scipy.io import wavfile
from collections import namedtuple
import warnings


class Rhythm(BaseSequence):

    def __init__(self, iois, n_bars, time_sig, beat_ms, played=None):
        # Save attributes
        self.time_sig = time_sig  # Used for metrical sequences
        self.beat_ms = beat_ms  # Used for metrical sequences
        self.n_bars = n_bars
        self.played = played

        # Call initializer of super class
        BaseSequence.__init__(self, iois, metrical=True, played=played)

    def __str__(self):
        return f"Object of type Rhythm.\nTime signature: {self.time_sig}\nNumber of bars: {self.n_bars}\nQuarternote (" \
               f"ms): {self.beat_ms}\nNumber of events: {len(self.onsets)}\nIOIs: {self.iois}\n" \
               f"Onsets:{self.onsets}\nOnsets played: {self.played}"

    def __add__(self, other):
        return join_rhythms([self, other])

    @property
    def note_values(self):
        """
        Get note values from the IOIs, based on beat_ms.
        """
        if not self.metrical or not self.time_sig or not self.beat_ms:
            raise ValueError("This is not a rhythmic sequence. Use class method Sequence.from_note_values or e.g."
                             "random_rhythmic_sequence(). Alternatively, you can set the following properties manually: "
                             "Sequence.metrical (boolean), Sequence.time_sig (tuple), Sequence.n_bars (int).")

        ratios = self.iois / self.beat_ms / 4

        note_values = np.array([1 // ratio for ratio in ratios])

        return note_values

    @classmethod
    def from_note_values(cls, note_values, time_signature=(4, 4), beat_ms=500, played=None):
        """
        Almost same as standard initialization, except that we don't provide the number of bars but calculate those.

        """
        if played is None:
            played = [True] * len(note_values)
        elif len(played) == len(note_values):
            played = played
        else:
            raise ValueError("The 'played' argument should contain an equal number of "
                             "booleans as the number of note_values.")

        ratios = np.array([1 / note * time_signature[1] for note in note_values])

        n_bars = np.sum(ratios) / time_signature[0]

        if n_bars % 1 != 0:
            raise ValueError("The provided note values do not amount to whole bars.")
        else:
            n_bars = int(n_bars)

        iois = ratios * beat_ms

        return cls(iois,
                   time_sig=time_signature,
                   beat_ms=beat_ms,
                   n_bars=n_bars,
                   played=played)

    @classmethod
    def from_iois(cls, iois, time_signature, beat_ms, played=None):
        if played is None:
            played = [True] * len(iois)
        elif len(played) == len(iois):
            played = played
        else:
            raise ValueError("The 'played' argument should contain an equal number of "
                             "booleans as the number of note_values.")

        note_values = np.array([ioi / beat_ms * time_signature[1] for ioi in iois])

        n_bars = np.sum(note_values) / time_signature[0]

        if n_bars % 1 != 0:
            raise ValueError("The provided note values do not amount to whole bars.")
        else:
            n_bars = int(n_bars)

        return cls(iois,
                   time_sig=time_signature,
                   beat_ms=beat_ms,
                   n_bars=n_bars,
                   played=played)

    @classmethod
    def generate_random_rhythm(cls, n_bars,
                               allowed_note_values,
                               time_signature,
                               beat_ms,
                               events_per_bar=None,
                               n_random_rests=0):
        """
        This function returns a randomly generated integer ratio Sequence on the basis of the provided params.
        """

        iois = np.empty(0)

        for bar in range(n_bars):
            all_rhythmic_ratios = _all_rhythmic_ratios(allowed_note_values, time_signature, target_n=events_per_bar)
            ratios = random.choice(all_rhythmic_ratios)

            new_iois = ratios * 4 * beat_ms

            iois = np.concatenate((iois, new_iois), axis=None)

        if n_random_rests > 0:
            if n_random_rests > len(iois):
                raise ValueError("You asked for more rests than there were events in the sequence.")
            played = [True] * (len(iois) - n_random_rests) + [False] * n_random_rests
            random.shuffle(played)
        else:
            played = [True] * len(iois)

        return cls(iois, time_sig=time_signature, beat_ms=beat_ms, n_bars=n_bars, played=played)

    @classmethod
    def generate_isochronous(cls, n_bars, time_sig, beat_ms, played=None):

        n_iois = time_sig[0] * n_bars

        if played is None:
            played = [True] * n_iois

        iois = n_iois * [beat_ms]

        return cls(iois=iois,
                   n_bars=n_bars,
                   time_sig=time_sig,
                   beat_ms=beat_ms,
                   played=played)

    def plot(self, filepath=None, print_staff=False):

        # create initial bar
        t = Track()
        b = Bar(meter=self.time_sig)

        # keep track of the index of the note_value
        note_i = 0
        # loop over the note values of the sequence

        values_and_played = list(zip(self.note_values, self.played))

        for note_value, played in values_and_played:
            if played:
                b.place_notes('G-4', self.note_values[note_i])
            elif not played:
                b.place_rest(self.note_values[note_i])

            # if bar is full, create new bar and add bar to track
            if b.current_beat == b.length:
                t.add_bar(b)
                b = Bar(meter=self.time_sig)

            note_i += 1

        # If final bar was not full yet, add a rest for the remaining duration
        if b.current_beat % 1 != 0:
            rest_value = 1 / b.space_left()
            if round(rest_value) != rest_value:
                raise ValueError("The rhythm could not be plotted. Most likely because the IOIs cannot "
                                 "be (easily) captured in musical notation. This for instance happens after "
                                 "using one of the tempo manipulation methods.")

            b.place_rest(rest_value)
            t.add_bar(b)

        plt = _plot_lp(t, filepath, print_staff=print_staff)

        return plt


class RhythmTrial:

    def __init__(self,
                 rhythm: Rhythm,
                 stims: Stimuli,
                 name: str = None,
                 first_layer_id: int = 0):

        # Check if right objects were passed
        if not isinstance(rhythm, Rhythm) or not isinstance(stims, Stimuli):
            raise ValueError("Please provide a Rhythm object as the first argument, and a Stimuli object as the second "
                             "argument.")

        # Initialize namedtuple
        self.Event = namedtuple('Event', 'onset ioi duration samples layer')

        # Then add events as namedtuples to self.events
        events = []
        self.events = self._add_events(events, rhythm, stims, first_layer_id)

        # Save provided trial name
        self.name = name

        # Save rhythmic attributes
        self.time_sig = rhythm.time_sig
        self.beat_ms = rhythm.beat_ms
        self.n_bars = rhythm.n_bars
        self.note_values = rhythm.note_values
        self.bar_duration = np.sum(rhythm.iois) / rhythm.n_bars
        self.total_duration = np.sum(rhythm.iois)

        # Save stimulus attributes
        self.fs = stims.fs
        self.n_channels = stims.n_channels

        # Make initial sound
        self.samples = self._make_sound(self.events)

    def _make_sound(self, events):
        array_length = int(self.total_duration / 1000 * self.fs)
        if self.n_channels == 1:
            samples = np.zeros(array_length)
        else:
            samples = np.zeros((array_length, 2))

        for event in events:
            if event.samples is not None:
                start_pos = int(event.onset / 1000 * self.fs)
                end_pos = int(start_pos + (event.duration / 1000 * self.fs))
                if self.n_channels == 1:
                    samples[start_pos:end_pos] += event.samples
                elif self.n_channels == 2:
                    samples[start_pos:end_pos, :2] += event.samples

        if np.max(samples) > 1:
            warnings.warn("\nSound was normalized to avoid distortion. If undesirable, change amplitude of the stims.")
            return _normalize_audio(samples)
        else:
            return samples

    def _add_events(self, current_events, rhythm, stims, layer_id):

        events = current_events

        # Make some additional variables
        layer = [layer_id] * stims.n
        event_duration = []
        for stim in stims:
            if stim is not None:
                event_duration.append(stim.duration_ms)
            else:
                event_duration.append(None)

        # Save each event to self.events as a named tuple
        for event in zip(rhythm.onsets, rhythm.iois, event_duration, stims.samples, layer):
            entry = self.Event(event[0], event[1], event[2], event[3], event[4])
            events.append(entry)

        return events

    def add_layer(self, rhythm: Rhythm, stims: Stimuli, layer_id: int):
        # Check if right objects were passed
        if not isinstance(rhythm, Rhythm) or not isinstance(stims, Stimuli):
            raise ValueError("Please provide a Rhythm object as the first argument, and a Stimuli object as the second "
                             "argument.")

        # todo Add checks to see if time_sig, beat_ms, sampling frequency all that are the same!
        # Add new layer to events

        # todo Check whether layer already exists
        self.events = self._add_events(self.events, rhythm, stims, layer_id)
        # make sound and save to self.samples
        self.samples = self._make_sound(self.events)

    def play(self, loop=False, metronome=False, metronome_amplitude=1):
        _play_samples(self.samples, self.fs, self.beat_ms, loop, metronome, metronome_amplitude)

    def plot_music(self, filepath=None, key='C', print_staff=True):
        pass

        # Call internal plot method to plot the track
        #_plot_lp(t, filepath, print_staff)

    def plot_waveform(self, title=None):
        if title:
            title = title
        else:
            if self.name:
                title = f"Waveform of {self.name}"
            else:
                title = "Waveform of RhythmTrial"

        _plot_waveform(self.samples, self.fs, self.n_channels, title)

    def write_wav(self, out_path='.',
                  metronome=False,
                  metronome_amplitude=1):
        """
        Writes audio to disk.
        """

        _write_wav(self.samples, self.fs, out_path, self.name, metronome, self.beat_ms, metronome_amplitude)


def _all_possibilities(nums, target):
    """
    I stole this code
    """

    res = []
    nums.sort()

    def dfs(left, path):
        if not left:
            res.append(path)
        else:
            for val in nums:
                if val > left:
                    break
                dfs(left - val, path + [val])

    dfs(target, [])

    return res


def _all_rhythmic_ratios(allowed_note_values, time_signature, target_n: int = None):
    common_denom = np.lcm(np.lcm.reduce(allowed_note_values), time_signature[1])

    allowed_numerators = common_denom // np.array(allowed_note_values)
    full_bar = time_signature[0] * (1 / time_signature[1])
    target = full_bar * common_denom

    all_possibilities = _all_possibilities(allowed_numerators, target)

    out_list = [(np.array(result) / common_denom) for result in
                all_possibilities]

    if target_n is not None:
        out_list = [rhythm for rhythm in out_list if len(rhythm) == target_n]
        if len(out_list) == 0:
            raise ValueError("No random rhythms exist that adhere to these parameters. "
                             "Try providing different parameters.")

    return out_list


def join_rhythms(iterator):
    """
    This function can join multiple Rhythm objects.
    """

    # Check whether iterable was passed
    if not hasattr(iterator, '__iter__'):
        raise ValueError("Please pass this function a list or other iterable object.")

    # Check whether all the objects are of the same type
    if not all(isinstance(rhythm, Rhythm) for rhythm in iterator):
        raise ValueError("This function can only join multiple Rhythm objects.")

    if not all(rhythm.time_sig == iterator[0].time_sig for rhythm in iterator):
        raise ValueError("Provided rhythms should have the same time signatures.")

    if not all(rhythm.beat_ms == iterator[0].beat_ms for rhythm in iterator):
        raise ValueError("Provided rhythms should have same tempo (beat_ms).")

    iois = [rhythm.iois for rhythm in iterator]
    iois = np.concatenate(iois)
    n_bars = np.sum([rhythm.n_bars for rhythm in iterator])
    played = [rhythm.played for rhythm in iterator]
    played = list(np.concatenate(played))
    played = [bool(x) for x in played]  # Otherwise we get Numpy booleans

    return Rhythm(iois,
                  n_bars=n_bars,
                  time_sig=iterator[0].time_sig,
                  beat_ms=iterator[0].beat_ms,
                  played=played)
