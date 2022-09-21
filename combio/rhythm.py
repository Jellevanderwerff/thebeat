from collections import namedtuple
from . import core
import os
import shutil
import subprocess
import tempfile
import warnings
import numpy as np
import skimage
from matplotlib import pyplot as plt
from mingus.containers import Bar, Track
from mingus.extra import lilypond
from typing import Union, Iterable


class Rhythm(core.sequence.BaseSequence):

    def __init__(self, iois, n_bars: int, time_signature, beat_ms):
        # Save attributes
        self.time_signature = time_signature  # Used for metrical sequences
        self.beat_ms = beat_ms  # Used for metrical sequences
        self.n_bars = n_bars

        if not np.sum(iois) % (n_bars * time_signature[0] * beat_ms) == 0:
            raise ValueError("The provided inter-onset intervals do not amount to whole bars.")

        # Call initializer of super class
        core.sequence.BaseSequence.__init__(self, iois, metrical=True)

    def __str__(self):
        return f"Object of type Rhythm.\nTime signature: {self.time_signature}\nNumber of bars: {self.n_bars}\nQuarternote (" \
               f"ms): {self.beat_ms}\nNumber of events: {len(self.onsets)}\nIOIs: {self.iois}\n" \
               f"Onsets:{self.onsets}\n"

    def __add__(self, other):
        return _join_rhythms([self, other])

    def __len__(self):
        return len(self.onsets)

    @property
    def note_values(self):
        """
        Get note values from the IOIs, based on beat_ms.
        """

        ratios = self.iois / self.beat_ms / 4

        note_values = np.array([int(1 // ratio) for ratio in ratios])

        return note_values

    @classmethod
    def from_note_values(cls, note_values, time_signature=(4, 4), beat_ms=500):
        """
        Almost same as standard initialization, except that we don't provide the number of bars but calculate those.

        """

        ratios = np.array([1 / note * time_signature[1] for note in note_values])

        n_bars = np.sum(ratios) / time_signature[0]

        if n_bars % 1 != 0:
            raise ValueError("The provided note values do not amount to whole bars.")
        else:
            n_bars = int(n_bars)

        iois = ratios * beat_ms

        return cls(iois,
                   time_signature=time_signature,
                   beat_ms=beat_ms,
                   n_bars=n_bars)

    @classmethod
    def from_iois(cls, iois, time_signature, beat_ms):
        n_bars = np.sum(iois) / time_signature[0] / beat_ms

        if n_bars % 1 != 0:
            raise ValueError("The provided note values do not amount to whole bars.")
        else:
            n_bars = int(n_bars)

        return cls(iois, n_bars, time_signature, beat_ms)

    @classmethod
    def generate_random_rhythm(cls,
                               allowed_note_values,
                               n_bars=1,
                               time_signature=(4, 4),
                               beat_ms=500,
                               events_per_bar=None,
                               rng=None):
        """
        This function returns a randomly generated integer ratio Sequence on the basis of the provided params.
        """

        if rng is None:
            rng = np.random.default_rng()

        iois = np.empty(0)

        for bar in range(n_bars):
            all_ratios = _all_rhythmic_ratios(allowed_note_values,
                                              time_signature,
                                              target_n=events_per_bar)
            ratios = list(rng.choice(all_ratios, 1))

            new_iois = ratios * 4 * beat_ms

            iois = np.concatenate((iois, new_iois), axis=None)

        return cls(iois, time_signature=time_signature, beat_ms=beat_ms, n_bars=n_bars)

    @classmethod
    def generate_isochronous(cls, n_bars, time_signature, beat_ms):

        n_iois = time_signature[0] * n_bars

        iois = n_iois * [beat_ms]

        return cls(iois=iois,
                   n_bars=n_bars,
                   time_signature=time_signature,
                   beat_ms=beat_ms)

    def plot_rhythm(self, filepath=None, print_staff=False, suppress_display=False):
        # create initial bar
        t = Track()
        b = Bar(meter=self.time_signature)

        # keep track of the index of the note_value
        note_i = 0
        # loop over the note values of the sequence

        for note_value in self.note_values:
            b.place_notes('G-4', note_value)

            # if bar is full, create new bar and add bar to track
            if b.current_beat == b.length:
                t.add_bar(b)
                b = Bar(meter=self.time_signature)

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

        lp = _get_lp_from_track(t, print_staff=print_staff)

        _plot_lp(lp, filepath, suppress_display=suppress_display)


class RhythmTrial:

    def __init__(self,
                 stimuli: Union[core.stimulus.Stimulus, Iterable[core.stimulus.Stimulus]],
                 rhythm: Rhythm,
                 name: str = None):

        # Check if correct objects were passed for stimuli
        if isinstance(stimuli, core.stimulus.Stimulus):
            stimuli = [stimuli] * len(rhythm)
        elif hasattr(stimuli, '__iter__'):
            pass
        else:
            raise ValueError("Please provided either a Stimulus object, "
                             "or an iterable of Stimulus objects as the first argument")

        # Check if correct objects was passed for rhythm
        if not isinstance(rhythm, Rhythm):
            raise ValueError("Please provide a Rhythm object as the second argument")

        # Initialize namedtuple
        self.Event = namedtuple('Event', 'layer onset ioi duration note_value samples')

        # Then add events as namedtuples to self.events
        events = []
        layer_id = [0] * len(stimuli)
        self.events = self._add_events(events, rhythm, stimuli, layer_id)

        # Save provided trial name
        self.name = name

        # Save rhythmic attributes
        self.time_signature = rhythm.time_signature
        self.beat_ms = rhythm.beat_ms
        self.n_bars = rhythm.n_bars
        self.note_values = rhythm.note_values
        self.bar_duration = np.sum(rhythm.iois) / rhythm.n_bars
        self.total_duration = np.sum(rhythm.iois)

        # Save stimulus attributes
        self.fs = stimuli[0].fs
        self.n_channels = stimuli[0].n_channels

        # Make initial sound
        self.samples = self._make_sound(self.events)

        # We start with one layer
        self.n_layers = 1

    def _add_events(self, current_events, rhythm, stims, layer_id):

        events = current_events

        # Make some additional variables
        event_durations = [stim.duration_ms for stim in stims]
        stim_samples = [stim.samples for stim in stims]

        # Save each event to self.events as a named tuple
        for event in zip(layer_id, rhythm.onsets, rhythm.iois, event_durations, rhythm.note_values, stim_samples):
            entry = self.Event(event[0], event[1], event[2], event[3], event[4], event[5])
            events.append(entry)

        return events

    def _make_sound(self, provided_events):
        # Generate an array of silence that has the length of all the onsets + one final stimulus.
        # In the case of a metrical sequence, we add the final ioi
        # The dtype is important, because that determines the values that the magnitudes can take.

        length = self.total_duration / 1000 * self.fs
        if int(length) != length:  # let's avoid rounding issues
            warnings.warn("Number of frames was rounded off to nearest integer ceiling. "
                          "This shouldn't be much of a problem.")

        array_length = int(np.ceil(length))

        if self.n_channels == 1:
            samples = np.zeros(array_length)
        else:
            samples = np.zeros((array_length, 2))

        for event in provided_events:
            if event.samples is not None:
                if self.n_channels == 1:
                    start_pos = int(event.onset / 1000 * self.fs)
                    end_pos = int(start_pos + event.samples.shape[0])
                    try:
                        samples[start_pos:end_pos] = samples[start_pos:end_pos] + event.samples
                    except ValueError:
                        raise ValueError("Could not make sound. Probably the final stimulus is longer than the "
                                         "final note value.")
                elif self.n_channels == 2:
                    start_pos = int(event.onset / 1000 * self.fs)
                    end_pos = int(start_pos + (event.duration / 1000 * self.fs))
                    try:
                        samples[start_pos:end_pos, :2] = samples[start_pos:end_pos, :2] + event.samples
                    except ValueError:
                        raise ValueError("Could not make sound. Probably the final stimulus is longer than the "
                                         "final note value.")

        if np.max(samples) > 1:
            warnings.warn("Sound was normalized to avoid distortion. If undesirable, change amplitude of the stims.")
            return core.helpers.normalize_audio(samples)
        else:
            return samples

    def add_layer(self, rhythm: Rhythm, stimuli: Union[core.stimulus.Stimulus, Iterable[core.stimulus.Stimulus]]):

        if self.n_layers > 3:
            raise ValueError("Can, for now, only handle 4 layers.")

        # Check if right objects were passed
        if not isinstance(rhythm, Rhythm):
            raise ValueError("Please provide a Rhythm object as the first argument.")

        if not all(isinstance(stim, core.stimulus.Stimulus) for stim in stimuli) or not isinstance(stimuli,
                                                                                                   core.stimulus.Stimulus):
            raise ValueError("Please provide either an iterable (e.g. a list) with Stimulus objects as the second "
                             "argument, or a single Stimulus object.")

        # multiply if Stimulus object was passed
        if isinstance(stimuli, core.stimulus.Stimulus):
            stimuli = [stimuli] * len(rhythm.onsets)

        # fs
        if not all(stimulus.fs == stimuli[0].fs for stimulus in stimuli):
            raise ValueError("The provided stimuli do not all have the same sampling frequency.")
        elif not stimuli[0].fs == self.fs:
            raise ValueError("The provided stimuli have a different sampling frequency than the stimuli "
                             "currently in this trial.")

        # n channels
        if not all(stimulus.n_channels for stimulus in stimuli):
            raise ValueError("The provided stimuli do not all have the same number of channels.")
        elif not stimuli[0].n_channels == self.n_channels:
            raise ValueError("The provided stimuli do not have the same number of channels as the stimuli "
                             "currently in this trial.")
        # beat ms
        if not rhythm.beat_ms == self.beat_ms:
            raise ValueError("The provided rhythm has a different beat_ms than the rhythm "
                             "currently in this trial.")

        # time signature
        if not rhythm.time_signature == self.time_signature:
            raise ValueError("The provided rhythm has a different time signature than the rhythm "
                             "currently in this trial.")

        # add layer to self.events
        layer_id = [self.n_layers] * len(stimuli)
        self.events = self._add_events(self.events, rhythm, stimuli, layer_id)

        # make sound and save to self.samples
        self.samples = self._make_sound(self.events)

        # add one to the number of layers
        self.n_layers += 1

    def play(self, loop=False, metronome=False, metronome_amplitude=1):
        core.helpers.play_samples(self.samples, self.fs, self.beat_ms, loop, metronome, metronome_amplitude)

    def plot_rhythm(self,
                    filepath=None,
                    print_staff=True,
                    lilypond_percussion_notes=None,
                    stem_directions=None,
                    suppress_display=False):
        """

        Parameters
        ----------
        filepath
        print_staff
        lilypond_percussion_notes:
            List of lilypond percussion notes for each layer. Defaults to ['bd', 'snare', 'hihat'].
            See possible options here: https://lilypond.org/doc/v2.23/Documentation/notation/percussion-notes
        stem_directions
        suppress_display

        Returns
        -------

        """

        lp = _get_lp_from_events(self.events,
                                 self.n_layers,
                                 self.time_signature,
                                 print_staff=print_staff,
                                 lilypond_percussion_notes=lilypond_percussion_notes,
                                 stem_directions=stem_directions)

        _plot_lp(lp, filepath=filepath, suppress_display=suppress_display)

        warnings.warn("Time signatures aren't implemented here yet! Forgot that...")

    def plot_waveform(self, title=None):
        if title:
            title = title
        else:
            if self.name:
                title = f"Waveform of {self.name}"
            else:
                title = "Waveform of RhythmTrial"

        core.helpers.plot_waveform(self.samples, self.fs, self.n_channels, title)

    def write_wav(self, out_path='.',
                  metronome=False,
                  metronome_amplitude=1):
        """
        Writes audio to disk.
        """

        core.helpers.write_wav(self.samples, self.fs, out_path, self.name, metronome, self.beat_ms, metronome_amplitude)


def _join_rhythms(iterator):
    """
    This function can join multiple Rhythm objects.
    """

    # Check whether iterable was passed
    if not hasattr(iterator, '__iter__'):
        raise ValueError("Please pass this function a list or other iterable object.")

    # Check whether all the objects are of the same type
    if not all(isinstance(rhythm, Rhythm) for rhythm in iterator):
        raise ValueError("You can only join multiple Rhythm objects.")

    if not all(rhythm.time_signature == iterator[0].time_signature for rhythm in iterator):
        raise ValueError("Provided rhythms should have the same time signatures.")

    if not all(rhythm.beat_ms == iterator[0].beat_ms for rhythm in iterator):
        raise ValueError("Provided rhythms should have same tempo (beat_ms).")

    iois = [rhythm.iois for rhythm in iterator]
    iois = np.concatenate(iois)
    n_bars = int(np.sum([rhythm.n_bars for rhythm in iterator]))

    return Rhythm(iois, n_bars=n_bars, time_signature=iterator[0].time_signature, beat_ms=iterator[0].beat_ms)


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

    possibilities = _all_possibilities(allowed_numerators, target)

    out_list = [(np.array(result) / common_denom) for result in
                possibilities]

    if target_n is not None:
        out_list = [rhythm for rhythm in out_list if len(rhythm) == target_n]
        if len(out_list) == 0:
            raise ValueError("No random rhythms exist that adhere to these parameters. "
                             "Try providing different parameters.")

    return out_list


def _get_lp_from_track(t, print_staff: bool):
    """
    Internal method for plotting a mingus Track object via lilypond.
    """

    remove_footers = """\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""
oddFooterMarkup = ""\nevenFooterMarkup = ""\n} """
    remove_staff = '{ \stopStaff \override Staff.Clef.color = #white'

    if print_staff is True:
        lp = '\\version "2.10.33"\n' + lilypond.from_Track(t) + remove_footers
    elif print_staff is False:
        lp = '\\version "2.10.33"\n' + remove_staff + lilypond.from_Track(t)[1:] + remove_footers
    else:
        raise ValueError("Wrong value specified for print_staff.")

    return lp


def _get_lp_from_events(provided_events,
                        n_layers: int,
                        time_signature: tuple,
                        print_staff: bool = True,
                        lilypond_percussion_notes=None,
                        stem_directions=None):
    if any(event.samples is None for event in provided_events):
        warnings.warn("'Rests' are plotted as empty spaces, not as rests. Please check manually whether"
                      "the plot makes sense.")

    if lilypond_percussion_notes is None:
        lilypond_percussion_notes = ['bd', 'snare', 'hihat']

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
        events = [event for event in provided_events if event.layer == layer]

        bar = ''
        b = Bar(meter=time_signature)

        for event in events:
            note_value = event.note_value
            b.place_rest(note_value)  # This is only to keep track of the number of notes in a bar
            if event.samples is not None:
                note = lilypond_percussion_notes[layer] + str(note_value) + ' '
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


def _plot_lp(lp, filepath, suppress_display):
    format = os.path.splitext(filepath)[1] if filepath else '.png'

    with tempfile.TemporaryDirectory() as tmp_dir:
        # run subprocess
        if format not in ['.eps', '.png']:
            raise ValueError("Can only export .png or .eps files.")

        command = ['lilypond', '-dbackend=eps', '--silent', '-dresolution=600', f'--{format[1:]}', '-o', 'rhythm', 'rhythm.ly']
        with open(os.path.join(tmp_dir, 'rhythm.ly'), 'w') as file:
            file.write(lp)

        subprocess.run(command, cwd=tmp_dir, check=True)
        result_path = os.path.join(tmp_dir, 'rhythm' + format)


        if filepath:
            shutil.copy(result_path, filepath)

        image = skimage.img_as_float(skimage.io.imread(result_path))

    # Select all pixels almost equal to white
    # (almost, because there are some edge effects in jpegs
    # so the boundaries may not be exactly white)
    white = np.array([1, 1, 1])
    mask = np.abs(image - white).sum(axis=2) < 0.05

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    out = image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    # show plot
    if not filepath and not suppress_display:
        plt.imshow(out)
        plt.axis('off')
        plt.show()
    elif format == '.png' and not suppress_display:
        plt.imshow(out)
        plt.axis('off')
        # plt.savefig(filename, bbox_inches='tight')  # TODO Jelle needs to figure out what he wants
    else:
        pass
