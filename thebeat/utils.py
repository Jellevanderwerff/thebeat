# Copyright (C) 2022-2023  Jelle van der Werff
#
# This file is part of thebeat.
#
# thebeat is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# thebeat is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with thebeat.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import thebeat.core
from thebeat._warnings import phases_t_at_zero

try:
    import abjad
except ImportError:
    abjad = None


def get_ioi_df(sequences: (thebeat.core.Sequence | list[thebeat.core.Sequence] | np.ndarray[thebeat.core.Sequence]),
               additional_functions: list[callable] | None = None):
    """
    This function exports a Pandas :class:`pandas.DataFrame` with information about the provided
    :py:class:`thebeat.core.Sequence` objects in
    `tidy data <https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html>`_ format.
    The DataFrame always has the columns:

    * ``Sequence_index``: The index of the Sequence object in the list of Sequences.
    * ``IOI_i``: The index of the IOI in the Sequence.
    * ``IOI``: The IOI.

    Additionally it has a column ``Sequence_name`` if at least one of the provided Sequence objects
    has a name.

    Moreover, one can provide a list of functions that will be applied to each sequence's IOIs.
    The results will be added as additional columns in the output DataFrame. See under 'Examples' for an
    illustration.

    Parameters
    ----------
    sequences
        The Sequence object(s) to be exported.
    additional_functions
        A list of functions that will be applied to the IOIs for each individual sequence,
        and the results of which will be added as additional columns.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with information about the provided Sequence objects in tidy data format.

    Examples
    --------
    >>> rng = np.random.default_rng(123)
    >>> seqs = [thebeat.core.Sequence.generate_random_normal(n_events=10, mu=500, sigma=25, rng=rng) for _ in range(10)]
    >>> df = get_ioi_df(seqs)
    >>> print(df.head())
       sequence_i  ioi_i         ioi
    0           0      0  475.271966
    1           0      1  490.805334
    2           0      2  532.198132
    3           0      3  504.849360
    4           0      4  523.005772

    >>> import numpy as np
    >>> df = get_ioi_df(seqs, additional_functions=[np.mean, np.std])
    >>> print(df.head())
       sequence_i        mean        std  ioi_i         ioi
    0           0  503.364499  17.923263      0  475.271966
    1           0  503.364499  17.923263      1  490.805334
    2           0  503.364499  17.923263      2  532.198132
    3           0  503.364499  17.923263      3  504.849360
    4           0  503.364499  17.923263      4  523.005772

    """

    # Checks
    if not all(isinstance(sequence, thebeat.core.Sequence) for sequence in sequences):
        raise TypeError("The provided sequences must be Sequence objects.")

    if additional_functions is not None and not all(callable(f) for f in additional_functions):
        raise TypeError("The functions in additional_functions must be callable.")

    # Create output dictionary
    output_df = None

    # Loop over sequences to fill output_dict with the columns that we always have
    for i, sequence in enumerate(sequences):
        # Start with the sequence index and the sequence name if it exists
        sequence_dict = {'sequence_i': i,
                         'sequence_name': sequence.name if sequence.name else np.nan}

        # If functions were provided, add those columns
        if additional_functions is not None:
            for f in additional_functions:
                # todo consider what sorts of error handling to do here
                sequence_dict[f.__name__] = f(sequence.iois)

        # Add the IOI index and the IOI itself
        sequence_dict['ioi_i'] = np.arange(len(sequence.iois))
        sequence_dict['ioi'] = sequence.iois

        # Concatenate the new DataFrame to the output DataFrame
        # If this is the first iteration, we need to create the output DataFrame first
        if output_df is None:
            output_df = pd.DataFrame(sequence_dict)
        else:
            output_df = pd.concat([output_df, pd.DataFrame(sequence_dict)], ignore_index=True)

    # Check if all names are None, if so, drop the column
    if output_df['sequence_name'].isnull().all():
        output_df.drop('sequence_name', axis=1, inplace=True)

    return output_df


def get_major_scale(tonic: str,
                    octave: int) -> list:
    """Get the major scale for a given tonic and octave. Returns a list of :class:`abjad.pitch.NamedPitch` objects.

    Note
    ----
    This function requires abjad to be installed.

    Parameters
    ----------
    tonic
        The tonic of the scale, e.g. 'G'.
    octave
        The octave of the scale, e.g. 4.

    Returns
    -------
    pitches
        A list of :class:`abjad.pitch.NamedPitch` objects.

    """
    if abjad is None:
        raise ImportError("This function requires the abjad package. Install, for instance by typing "
                          "'pip install abjad' into your terminal.")

    intervals = "M2 M2 m2 M2 M2 M2 m2".split()
    intervals = [abjad.NamedInterval(interval) for interval in intervals]

    pitches = []

    pitch = abjad.NamedPitch(tonic, octave=octave)

    pitches.append(pitch)

    for interval in intervals:
        pitch = pitch + interval

        pitches.append(pitch)

    return pitches


def get_phase_differences(test_sequence: thebeat.core.Sequence,
                          reference_sequence: thebeat.core.Sequence | float,
                          circular_unit="degrees"):
    # todo Verify this method of calculating phase differences. I'm not quite sure about using the period of the
    #  test_sequence instead of the reference_sequence

    """Get the phase differences for ``test_sequence`` compared to ``reference_sequence``. If the second argument is a
    number, ``test_sequence`` will be compared with an isochronous sequence with a constant inter-onset interval (IOI)
    of that number and the same length as the test sequence.

    Caution
    -------
    The phase differences are calculated for each onset of ``test_sequence`` compared to the onset with the same
    index of ``reference_sequence``. Missing values are discarded. In addition, if the first onset of the test sequence
    is at t = 0, that phase difference is also discarded.

    Parameters
    ----------
    test_sequence
        The sequence to be compared with the reference sequence. Can either be a single Sequence or
        a list or array of Sequences.
    reference_sequence
        The reference sequence. Can be a Sequence object, a list or array of Sequence objects, or a number.
        In the latter case, the reference sequence will be an isochronous sequence with a constant IOI of that
        number and the same length as ``sequence_1``.
    circular_unit
        The unit of the circular unit. Can be "degrees" or "radians".

    """

    # Input validation
    if not isinstance(test_sequence, thebeat.core.Sequence):
        raise TypeError("Please provide a Sequence object as the left argument.")
    elif isinstance(reference_sequence, (int, float)):
        reference_sequence = thebeat.core.Sequence.generate_isochronous(n_events=len(test_sequence.onsets),
                                                                        ioi=reference_sequence)
    elif isinstance(reference_sequence, thebeat.core.Sequence):
        pass
    else:
        raise TypeError("Please provide a Sequence object as the left-hand argument, and a Sequence object or a "
                        "number as the right-hand argument.")

    # Get onsets once
    test_onsets = test_sequence.onsets
    ref_onsets = reference_sequence.onsets

    # If the first onset is at t=0, raise warning and remove it
    if test_sequence._first_onset == 0:
        warnings.warn(phases_t_at_zero)
        test_onsets = test_onsets[1:]
        ref_onsets = ref_onsets[1:]
        start_at_tzero = False
    else:
        start_at_tzero = True

    # Check length sameness
    if not len(test_onsets) == len(ref_onsets):
        raise ValueError("This function only works if the number of events in the two sequences are equal. For "
                         "missing data, insert np.nan values in the sequence for the missing data.")

    # Output array
    phase_diffs = np.array([])

    # Calculate phase differences
    for i, test_onset in enumerate(test_onsets):

        # For the first event, we use the period of the IOI that follows the event, but only if it was the
        # first onset
        if i == 0 and start_at_tzero is True:
            period_next = test_sequence.iois[0]
            period_prev = period_next
        # For the last event, we use the period of the IOI that precedes the event
        elif i == len(test_onsets) - 1:
            period_prev = test_sequence.iois[i - 1]
            period_next = period_prev
        # For all other events, we need both the previous and the next IOI
        else:
            period_prev = test_sequence.iois[i - 1]
            period_next = test_sequence.iois[i]

        if test_onset > ref_onsets[i]:
            phase_diff = (test_onset - ref_onsets[i]) / period_next
        elif test_onset < ref_onsets[i]:
            phase_diff = (test_onset - ref_onsets[i]) / period_prev
        elif test_onset == ref_onsets[i]:
            phase_diff = 0.0
        elif np.isnan(test_onset) or np.isnan(ref_onsets[i]):
            phase_diff = np.nan
            warnings.warn(thebeat._warnings.missing_values)
        else:
            raise ValueError("Something went wrong during the calculation of the phase differences."
                             "Please check your data.")

        phase_diffs = np.append(phase_diffs, phase_diff)

    # Convert to degrees
    phase_diff_degrees = (phase_diffs * 360) % 360

    # Return
    if circular_unit == "degrees":
        return phase_diff_degrees
    elif circular_unit == "radians":
        return np.deg2rad(phase_diff_degrees)
    else:
        raise ValueError("Please provide a valid circular unit. Either 'degrees' or 'radians'.")


def get_interval_ratios_from_dyads(sequence: np.array | thebeat.core.Sequence | list):
    r"""
    Return sequential interval ratios, calculated as:

    :math:`\textrm{ratio}_k = \frac{\textrm{IOI}_k}{\textrm{IOI}_k + \textrm{IOI}_{k+1}}`.

    Note that for *n* IOIs this property returns *n*-1 ratios.

    Parameters
    ----------
    sequence
        The sequence from which to calculate the interval ratios. Can be a Sequence object, or a list or array of
        IOIs.

    Notes
    -----
    The used method is based on the methodology from :cite:t:`roeskeCategoricalRhythmsAre2020`.

    """
    if isinstance(sequence, thebeat.core.Sequence):
        sequence = sequence.iois

    return sequence[:-1] / (sequence[1:] + sequence[:-1])


def concatenate_sequences(sequences: np.typing.ArrayLike,
                          name: str | None = None):
    """Concatenate an array or list of :py:class:`~thebeat.core.Sequence` objects.

    Note
    ----
    Only works for Sequence objects where all but the last provided object has an
    ``end_with_interval=True`` flag.

    Parameters
    ----------
    sequences
        The to-be-concatenated objects.
    name
        Optionally, you can give the returned Sequence object a name.

    Returns
    -------
    object
        The concatenated Sequence
    """

    if not all(isinstance(obj, thebeat.core.Sequence) for obj in sequences):
        raise TypeError("Please pass only Sequence objects.")

    if not all(obj.end_with_interval for obj in sequences[:-1]):
        raise ValueError("All passed Sequence objects except for the final one need to end with an interval."
                         "Otherwise we miss an interval between the onset of the "
                         "final event in a Sequence and the onset of the first event in the next sequence.")

    if not all(obj.onsets[0] == 0.0 for obj in sequences):
        raise ValueError("Please only pass sequences that have their first event at onset 0.0")

    # Whether the sequence ends with an interval depends only on the final object passed
    end_with_interval = sequences[-1].end_with_interval

    # concatenate iois and create new Sequence
    iois = np.concatenate([obj.iois for obj in sequences])
    return thebeat.core.Sequence(iois, end_with_interval=end_with_interval, name=name)


def concatenate_soundsequences(sound_sequences: np.typing.ArrayLike,
                               name: str | None = None):
    """Concatenate an array or list of :py:class:`~thebeat.core.SoundSequence` objects.

    Note
    ----
    Only works for SoundSequence objects where all but the last provided object has an
    ``end_with_interval=True`` flag.

    Parameters
    ----------
    sound_sequences
        The to-be-concatenated objects.
    name
        Optionally, you can give the returned SoundSequence object a name.

    Returns
    -------
    object
        The concatenated SoundSequence
    """
    if not all(isinstance(obj, thebeat.core.SoundSequence) for obj in sound_sequences):
        raise TypeError("Please pass only SoundSequence objects.")

    if not all(obj.end_with_interval for obj in sound_sequences[:-1]):
        raise ValueError("All passed SoundSequence objects except for the final one need to end with an interval."
                         "Otherwise we miss an interval between the onset of the "
                         "final event in a Sequence and the onset of the first event in the next sequence.")

    # Whether the sequence ends with an interval depends only on the final object passed
    end_with_interval = sound_sequences[-1].end_with_interval

    # concatenate iois and create new Sequence
    iois = np.concatenate([obj.iois for obj in sound_sequences])
    seq = thebeat.core.Sequence(iois, end_with_interval=end_with_interval)

    # concatenate sounds
    all_sounds = [sound_obj for obj in sound_sequences for sound_obj in obj.sound_objects]

    return thebeat.core.SoundSequence(sound=all_sounds, sequence=seq, name=name)


def concatenate_soundstimuli(sound_stimuli: np.ndarray | list,
                             name: str | None = None):
    """Concatenate an array or list of :py:class:`~thebeat.core.SoundStimulus` objects.

    Parameters
    ----------
    sound_stimuli
        The to-be-concatenated objects.
    name
        Optionally, you can give the returned SoundStimulus object a name.

    Returns
    -------
    object
        The concatenated SoundStimulus
    """

    if not all(isinstance(obj, thebeat.core.SoundStimulus) for obj in sound_stimuli):
        raise TypeError("Please pass only SoundStimulus objects.")

    thebeat.helpers.check_sound_properties_sameness(sound_stimuli)

    samples = np.concatenate([obj.samples for obj in sound_stimuli])
    fs = sound_stimuli[0].fs

    return thebeat.core.SoundStimulus(samples, fs, name=name)


def merge_soundstimuli(sound_stimuli: np.typing.ArrayLike[thebeat.SoundStimulus],
                       name: str | None = None):
    """Merge an array or list of :py:class:`~thebeat.core.SoundStimulus` objects.
    The sound samples for each of the objects will be overlaid on top of each other.

    Parameters
    ----------

    sound_stimuli
        The to-be-merged objects.
    name
        Optionally, you can give the returned SoundStimulus object a name.

    Returns
    -------
    object
        The merged SoundStimulus
    """

    if not all(isinstance(obj, thebeat.core.SoundStimulus) for obj in sound_stimuli):
        raise TypeError("Can only overlay another SoundStimulus object on this SoundStimulus object.")

    # Check sameness of number of channels etc.
    thebeat.helpers.check_sound_properties_sameness(sound_stimuli)

    # Overlay sounds
    samples = thebeat.helpers.overlay_samples([obj.samples for obj in sound_stimuli])

    return thebeat.core.SoundStimulus(samples=samples, fs=sound_stimuli[0].fs, name=name)


def merge_sequences(sequences: np.typing.ArrayLike[thebeat.core.Sequence],
                    name: str | None = None):
    """Merge an array or list of :py:class:`~thebeat.core.Sequence` objects.
    The the event onsets in each of the objects will be overlaid on top of each other.

    Parameters
    ----------
    sequences
        The to-be-merged objects.
    name
        Optionally, you can give the returned Sequence object a name.

    Returns
    -------
    object
        The merged Sequence
    """

    # check if only Sequence objects were passed
    if not all(isinstance(obj, thebeat.core.Sequence) for obj in sequences):
        raise TypeError("Please pass only Sequence objects.")

    # concatenate onsets and sort
    onsets = np.concatenate([obj.onsets for obj in sequences])
    onsets.sort()

    # Check for duplicates
    if np.any(onsets[1:] == onsets[:-1]):
        raise ValueError("The merged Sequence object would contain duplicate onsets.")

    return thebeat.core.Sequence.from_onsets(onsets, name=name)


def merge_soundsequences(sound_sequences:
                         list[thebeat.core.SoundSequence],
                         name: str | None = None):
    """Merge a list or array of :py:class:`~thebeat.core.SoundSequence` objects.
    The event onsets in each of the objects will be overlaid on top of each other, after which the sounds

    Parameters
    ----------
    sound_sequences
        The to-be-merged objects.
    name
        Optionally, you can give the returned SoundSequence object a name.

    Returns
    -------
    object
        The merged SoundSequence
    """
    # check if only SoundSequence objects were passed
    if not all(isinstance(obj, thebeat.core.SoundSequence) for obj in sound_sequences):
        raise TypeError("Please pass only SoundSequence objects.")

    # Get all onsets and sounds
    all_onsets = np.concatenate([obj.onsets for obj in sound_sequences])
    all_sounds = [sound_obj for obj in sound_sequences for sound_obj in obj.sound_objects]

    # Sort sounds in same order as onsets
    sounds_sorted = [all_sounds[i] for i in np.argsort(all_onsets)]

    # Sort onsets onsets and create new Sequence
    onsets_sorted = np.sort(all_onsets)
    seq = thebeat.Sequence.from_onsets(onsets_sorted)

    return thebeat.core.SoundSequence(sound=sounds_sorted, sequence=seq, name=name)
