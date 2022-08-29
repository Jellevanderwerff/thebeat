from scipy.io import wavfile
from .sequence import BaseSequence, Sequence
from .stimulus import Stimulus
import numpy as np
from combio.core.helpers import play_samples, plot_waveform, normalize_audio, get_sound_with_metronome
import warnings
import os
from typing import Iterable, Union


class StimTrial(BaseSequence):
    """
    StimSequence class which inherits only the most basic functions from BaseSequence
    """

    def __init__(self,
                 stimuli: Union[Stimulus, Iterable],
                 sequence_object: Sequence,
                 name: str = None):

        # If a single Stimulus object is passed, repeat that stimulus for each onset
        if isinstance(stimuli, Stimulus):
            stimuli = [stimuli] * len(sequence_object.onsets)

        # Type checking for sequence
        if not isinstance(sequence_object, Sequence):
            raise ValueError("Please provide a Sequence object as the second argument.")

        # Check whether fs, dtype, and n_channels are the same for all stimuli
        all_fs = [snd.fs for snd in stimuli]
        all_dtypes = [snd.dtype for snd in stimuli]
        all_n_channels = [snd.n_channels for snd in stimuli]

        if not all(fs == all_fs[0] for fs in all_fs):
            raise ValueError("The Stimulus objects in the passed list have different sampling frequencies!")
        elif not all(dtype == all_dtypes[0] for dtype in all_dtypes):
            raise ValueError("The Stimulus objects in the passed list have different dtypes!")
        elif not all(n_channels == all_n_channels[0] for n_channels in all_n_channels):
            raise ValueError("The Stimulus objects in the passed list have differing number of channels!")

        # Save attributes
        self.fs = stimuli[0].fs
        self.dtype = stimuli[0].dtype
        self.n_channels = stimuli[0].n_channels
        self.name = name
        self.stim_names = [stimulus.name for stimulus in stimuli]
        self.metrical = sequence_object.metrical

        # Initialize Sequence class
        BaseSequence.__init__(self, sequence_object.iois, metrical=sequence_object.metrical)

        # Make sound which saves the samples to self.samples
        self.samples = self._make_sound(stimuli, self.onsets)

    def __str__(self):

        if self.name:
            name = self.name
        else:
            name = "Not provided"

        if all(stim_name is None for stim_name in self.stim_names):
            stim_names = "None provided"
        else:
            stim_names = []
            for stim_name in self.stim_names:
                if stim_name is None:
                    stim_names.append("Unknown")
                else:
                    stim_names.append(stim_name)

        if self.metrical:
            return f"""
Object of type StimTrial (metrical version):
StimTrial name: {name}
{len(self.onsets)} events
IOIs: {self.iois}
Onsets: {self.onsets}
Stimulus names: {stim_names}
            """
        else:
            return f"""
Object of type StimTrial (non-metrical version):
StimTrial name: {name}
{len(self.onsets)} events
IOIs: {self.iois}
Onsets: {self.onsets}
Stimulus names: {stim_names}
            """

    @property
    def mean_ioi(self):
        return np.mean(self.iois)

    def play(self, loop=False, metronome=False, metronome_amplitude=1):
        play_samples(self.samples, self.fs, self.mean_ioi, loop, metronome, metronome_amplitude)

    def plot_waveform(self, title=None):
        if title:
            title = title
        else:
            if self.name:
                title = f"Waveform of {self.name}"
            else:
                title = "Waveform of StimTrial"

        plot_waveform(self.samples, self.fs, self.n_channels, title)

    def write_wav(self, out_path='.',
                  metronome=False,
                  metronome_amplitude=1):
        """
        Writes audio to disk.
        """

        _write_wav(self.samples, self.fs, out_path, self.name, metronome, self.mean_ioi, metronome_amplitude)

    def _make_sound(self, stimuli, onsets):
        # Check for overlap
        for i in range(len(onsets)):
            stim_duration = stimuli[i].samples.shape[0] / self.fs * 1000
            try:
                ioi_after_onset = onsets[i + 1] - onsets[i]
                if ioi_after_onset < stim_duration:
                    raise ValueError(
                        "The duration of one or more stimuli is longer than its respective IOI. "
                        "The events will overlap: either use different IOIs, or use a shorter stimulus sound.")
            except IndexError:
                pass

        # Generate an array of silence that has the length of all the onsets + one final stimulus.
        # In the case of a metrical sequence, we add the final ioi
        # The dtype is important, because that determines the values that the magnitudes can take.
        if self.metrical:
            array_length = int((onsets[-1] + self.iois[-1]) / 1000 * self.fs)
        elif not self.metrical:
            array_length = int((onsets[-1] / 1000 * self.fs) + stimuli[-1].samples.shape[0])
        else:
            raise ValueError("Error during calculation of array_length")

        if self.n_channels == 1:
            samples = np.zeros(array_length, dtype=self.dtype)
        else:
            samples = np.zeros((array_length, 2), dtype=self.dtype)

        samples_with_onsets = list(zip([stimulus.samples for stimulus in stimuli], onsets))

        for stimulus, onset in samples_with_onsets:
            start_pos = int(onset * self.fs / 1000)
            end_pos = int(start_pos + stimulus.shape[0])
            if self.n_channels == 1:
                samples[start_pos:end_pos] = stimulus
            elif self.n_channels == 2:
                samples[start_pos:end_pos, :2] = stimulus

        # return sound
        if np.max(samples) > 1:
            warnings.warn("Sound was normalized")
            return normalize_audio(samples)
        else:
            return samples


def _write_wav(samples, fs, out_path, name, metronome, metronome_ioi, metronome_amplitude):
    if metronome is True:
        samples = get_sound_with_metronome(samples, fs, metronome_ioi, metronome_amplitude)
    else:
        samples = samples

    out_path = str(out_path)

    if out_path.endswith('.wav'):
        path, filename = os.path.split(out_path)
    elif os.path.isdir(out_path):
        path = out_path
        if name:
            filename = f"{name}.wav"
        else:
            filename = f"out.wav"

    else:
        raise ValueError("Wrong out_path specified. Please provide a directory or a complete filepath.")

    wavfile.write(filename=os.path.join(path, filename), rate=fs, data=samples)
