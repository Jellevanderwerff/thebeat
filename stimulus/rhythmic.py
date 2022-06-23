from mingus.containers import Bar, Track
import numpy as np
from stimulus.base import BaseSequence, Stimuli, _plot_waveform, _normalize_audio, _play_samples, _write_wav
import random
from collections import namedtuple
import warnings
import os
import subprocess
import skimage
import matplotlib.pyplot as plt
from mingus.extra import lilypond


class Rhythm(BaseSequence):

    def __init__(self, iois, n_bars, time_sig, beat_ms):
        # Save attributes
        self.time_sig = time_sig  # Used for metrical sequences
        self.beat_ms = beat_ms  # Used for metrical sequences
        self.n_bars = n_bars

        # Call initializer of super class
        BaseSequence.__init__(self, iois, metrical=True)

    def __str__(self):
        return f"Object of type Rhythm.\nTime signature: {self.time_sig}\nNumber of bars: {self.n_bars}\nQuarternote (" \
               f"ms): {self.beat_ms}\nNumber of events: {len(self.onsets)}\nIOIs: {self.iois}\n" \
               f"Onsets:{self.onsets}\n"

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
                   time_sig=time_signature,
                   beat_ms=beat_ms,
                   n_bars=n_bars)

    @classmethod
    def from_iois(cls, iois, time_signature, beat_ms):

        note_values = np.array([ioi / beat_ms * time_signature[1] for ioi in iois])

        n_bars = np.sum(note_values) / time_signature[0]

        if n_bars % 1 != 0:
            raise ValueError("The provided note values do not amount to whole bars.")
        else:
            n_bars = int(n_bars)

        return cls(iois,
                   time_sig=time_signature,
                   beat_ms=beat_ms,
                   n_bars=n_bars)

    @classmethod
    def generate_random_rhythm(cls,
                               allowed_note_values,
                               n_bars=1,
                               time_signature=(4, 4),
                               beat_ms=500,
                               events_per_bar=None):
        """
        This function returns a randomly generated integer ratio Sequence on the basis of the provided params.
        """

        iois = np.empty(0)

        for bar in range(n_bars):
            all_rhythmic_ratios = _all_rhythmic_ratios(allowed_note_values, time_signature, target_n=events_per_bar)
            ratios = random.choice(all_rhythmic_ratios)

            new_iois = ratios * 4 * beat_ms

            iois = np.concatenate((iois, new_iois), axis=None)

        return cls(iois, time_sig=time_signature, beat_ms=beat_ms, n_bars=n_bars)

    @classmethod
    def generate_isochronous(cls, n_bars, time_sig, beat_ms):

        n_iois = time_sig[0] * n_bars

        iois = n_iois * [beat_ms]

        return cls(iois=iois,
                   n_bars=n_bars,
                   time_sig=time_sig,
                   beat_ms=beat_ms)

    def plot_rhythm(self, filepath=None, print_staff=False):
        # create initial bar
        t = Track()
        b = Bar(meter=self.time_sig)

        # keep track of the index of the note_value
        note_i = 0
        # loop over the note values of the sequence

        for note_value in self.note_values:
            b.place_notes('G-4', note_value)

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

        _plot_rhythm_lp(t, filepath, print_staff=print_staff)


class RhythmTrial:

    def __init__(self,
                 rhythm: Rhythm,
                 stims: Stimuli,
                 name: str = None):

        # Check if right objects were passed
        if not isinstance(rhythm, Rhythm) or not isinstance(stims, Stimuli):
            raise ValueError("Please provide a Rhythm object as the first argument, and a Stimuli object as the second "
                             "argument.")

        # Initialize namedtuple
        self.Event = namedtuple('Event', 'onset ioi duration samples layer')

        # Then add events as namedtuples to self.events
        events = []
        layer_id = [0] * stims.n
        self.events = self._add_events(events, rhythm, stims, layer_id)

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

        # We start with one layer
        self.n_layers = 1

    def _add_events(self, current_events, rhythm, stims, layer_id):

        events = current_events

        # Make some additional variables
        event_duration = []
        for stim in stims:
            if stim is not None:
                event_duration.append(stim.duration_ms)
            else:
                event_duration.append(None)

        # Save each event to self.events as a named tuple
        for event in zip(rhythm.onsets, rhythm.iois, event_duration, stims.samples, layer_id):
            entry = self.Event(event[0], event[1], event[2], event[3], event[4])
            events.append(entry)

        return events

    def _make_mingus_track_from_events(self, events, placable_notes):
        """Conceptually, I can't get my head around how to do this with NoteContainers etc.
        I think it will only be possible if the note durations for the notes that are played simultaneously
        are the same for the multiple layers. I don't think mingus supports different note durations for notes
        that start at the same moment."""
        pass

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
            warnings.warn("Sound was normalized to avoid distortion. If undesirable, change amplitude of the stims.")
            return _normalize_audio(samples)
        else:
            return samples

    def add_layer(self, rhythm: Rhythm, stims: Stimuli):

        if self.n_layers > 3:
            raise ValueError("Can, for now, only handle 4 layers.")

        # Check if right objects were passed
        if not isinstance(rhythm, Rhythm) or not isinstance(stims, Stimuli):
            raise ValueError("Please provide a Rhythm object as the first argument, and a Stimuli object as the second "
                             "argument.")

        if not stims.n_channels == self.n_channels:
            raise ValueError("The provided stims have a different number of channels than the stims "
                             "currently in this trial.")
        if not stims.fs == self.fs:
            raise ValueError("The provided stims have a different sampling frequency than the stims "
                             "currently in this trial.")
        if not rhythm.beat_ms == self.beat_ms:
            raise ValueError("The provided rhythm has a different beat_ms than the rhythm "
                             "currently in this trial.")
        if not rhythm.time_sig == self.time_sig:
            raise ValueError("The provided rhythm has a different time signature than the rhythm "
                             "currently in this trial.")

        layer_id = [self.n_layers] * stims.n  # Because we start counting from 0, but length is 0 + 1

        self.events = self._add_events(self.events, rhythm, stims, layer_id)
        # make sound and save to self.samples
        self.samples = self._make_sound(self.events)

        # add one to the number of layers
        self.n_layers += 1

    def play(self, loop=False, metronome=False, metronome_amplitude=1):
        _play_samples(self.samples, self.fs, self.beat_ms, loop, metronome, metronome_amplitude)

    def plot_rhythm(self, filepath=None, print_staff=True):
        pass

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


def _plot_rhythm_lp(t, filepath, print_staff: bool):
    """
    Internal method for plotting a mingus Track object via lilypond.
    """
    # This is the same each time:
    if filepath:
        location, filename = os.path.split(filepath)
        if location == '':
            location = '.'
    else:
        location = '.'
        filename = 'rhythm.png'

    # make lilypond string
    if print_staff is True:
        lp = '\\version "2.10.33"\n' + lilypond.from_Track(t) + '\n\paper {\nindent = 0\mm\nline-width = ' \
                                                                '110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ' \
                                                                '""\noddFooterMarkup = ""\nevenFooterMarkup = ""\n} '
    elif print_staff is False:
        lp = '\\version "2.10.33"\n' + '{ \stopStaff \override Staff.Clef.color = #white' + lilypond.from_Track(t)[
                                                                                            1:] + '\n\paper {\nindent = 0\mm\nline-width = ' \
                                                                                                  '110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ' \
                                                                                                  '""\noddFooterMarkup = ""\nevenFooterMarkup = ""\n} '
    else:
        raise ValueError("Wrong value specified for print_staff.")

    # write lilypond string to file
    with open(os.path.join(location, filename[:-4] + '.ly'), 'w') as file:
        file.write(lp)

    # run subprocess
    if filename.endswith('.eps'):
        command = f'lilypond -dbackend=eps --silent -dresolution=600 --eps -o {filename[:-4]} {filename[:-4] + ".ly"}'
        to_be_removed = ['.ly']
    elif filename.endswith('.png'):
        command = f'lilypond -dbackend=eps --silent -dresolution=600 --png -o {filename[:-4]} {filename[:-4] + ".ly"}'
        to_be_removed = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi', '.ly']
    else:
        raise ValueError("Can only export .png or .eps files.")

    p = subprocess.Popen(command, shell=True, cwd=location).wait()

    image = skimage.img_as_float(skimage.io.imread(filename))

    # Select all pixels almost equal to white
    # (almost, because there are some edge effects in jpegs
    # so the boundaries may not be exactly white)
    white = np.array([1, 1, 1])
    mask = np.abs(image - white).sum(axis=2) < 0.05

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    out = image[top_left[0]:bottom_right[0],
          top_left[1]:bottom_right[1]]

    # show plot
    if not filepath:
        plt.imshow(out)
        plt.axis('off')
        plt.show()
    elif filename.endswith('.png'):
        plt.imshow(out)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
    else:
        pass

    # remove files
    if filepath:
        filenames = [filename[:-4] + x for x in to_be_removed]
    else:
        to_be_removed = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi', '.ly', '.png']
        filenames = ['rhythm' + x for x in to_be_removed]

    for file in filenames:
        os.remove(os.path.join(location, file))


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

    return Rhythm(iois,
                  n_bars=n_bars,
                  time_sig=iterator[0].time_sig,
                  beat_ms=iterator[0].beat_ms)
