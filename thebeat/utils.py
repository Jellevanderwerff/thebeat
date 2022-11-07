import numpy as np
import thebeat.core
from typing import Optional, Union

try:
    import abjad
except ImportError:
    abjad = None


def get_phase_differences(test_sequence: thebeat.core.Sequence,
                          reference_sequence: Union[thebeat.core.Sequence, float],
                          circular_unit="degrees"):
    """Get the phase differences for ``sequence_1`` compared to ``sequence_2``. If the second argument is a number,
    ``sequence_1`` will be compared with an isochronous sequence with a constant inter-onset interval (IOI) of that
    number and the same length as sequence 1.

    The phase differences are calculated for each onset of ``sequence 1`` compared to the onset with the same
    index of sequence 2. Note, this means that this function does not handle missing values well.

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
        raise ValueError("Please provide a Sequence object as the left argument.")
    elif isinstance(reference_sequence, (int, float)):
        reference_sequence = thebeat.core.Sequence.generate_isochronous(n=len(test_sequence.onsets), ioi=reference_sequence)
    elif isinstance(reference_sequence, thebeat.core.Sequence):
        pass
    else:
        raise ValueError("Please provide a Sequence object as the left-hand argument, and a Sequence object or a "
                         "number as the right-hand argument.")

    # Get onsets once
    seq_1_onsets = test_sequence.onsets
    seq_2_onsets = reference_sequence.onsets

    # Check length sameness
    if not len(seq_1_onsets) == len(seq_2_onsets):
        raise ValueError("This function only works if the number of events in the two sequences are equal.")

    # Output array
    phase_diffs = np.array([])

    # Calculate phase differences
    for i, seq_1_onset in enumerate(seq_1_onsets):

        # For the first event, we use the period of the IOI that follows the event
        if i == 0:
            period_next = test_sequence.iois[0]
            period_prev = period_next
        # For the last event, we use the period of the IOI that precedes the event
        elif i == len(seq_1_onsets) - 1:
            period_prev = test_sequence.iois[i - 1]
            period_next = period_prev
        # For all other events, we need both the previous and the next IOI
        else:
            period_next = test_sequence.iois[i]
            period_prev = test_sequence.iois[i - 1]

        if seq_1_onset > seq_2_onsets[i]:
            phase_diff = (seq_1_onset - seq_2_onsets[i]) / period_next
        elif seq_1_onset < seq_2_onsets[i]:
            phase_diff = (seq_1_onset - seq_2_onsets[i]) / period_prev
        elif seq_1_onset == seq_2_onsets[i]:
            phase_diff = 0.0
        else:
            raise ValueError("Something went wrong during the calculation of the phase differences.")

        phase_diffs = np.append(phase_diffs, phase_diff)

    # Convert to degrees
    phase_diff_degrees = (phase_diffs * 360) % 360

    # Return
    if circular_unit == "degrees":
        return phase_diff_degrees
    elif circular_unit == "radians":
        return np.deg2rad(phase_diff_degrees)
    else:
        raise ValueError("Please provide a valid circular unit. Either degrees or radians.")


def get_major_scale(tonic: str,
                    octave: int) -> list[abjad.NamedPitch]:
    """Get the major scale for a given tonic and octave. Returns a list of abjad NamedPitch objects.

    Note
    ----
    This function requires abjad to be installed.

    Parameters
    ----------
    tonic
        The tonic of the scale, e.g. 'G'.
    octave
        The octave of the tonic, e.g. 4.

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


def join(objects: np.typing.ArrayLike,
         name: Optional[str] = None):
    """Join an array or list of :py:`Sequence` or :py:`StimSequence` objects.

    Note
    ----
    Only works for Sequence or StimSequence objects where all but the last provided object has a ``metrical=True`` flag.

    Parameters
    ----------
    objects
        The to-be-joined objects.
    name
        Optionally, you can give the returned Sequence or StimSequence object a name.

    Returns
    -------
    object
        The joined Sequence or StimSequence
    """

    if not all(isinstance(obj, thebeat.core.Sequence) for obj in objects) and not all(
            isinstance(obj, thebeat.core.StimSequence) for obj in objects):
        raise ValueError("Please pass only Sequence or only StimSequence objects.")

    if not all(obj.metrical for obj in objects[:-1]):
        raise ValueError("All passed Sequences or StimSequences need to be metrical, except for the final one. "
                         "Otherwise we miss an interval between the onset of the "
                         "final event in a Sequence and the onset of the first event in the next sequence.")

    if not all(obj.onsets[0] == 0.0 for obj in objects):
        raise ValueError("Please only pass sequences that have their first event at onset 0.0")

    # the metricality of the sequence depends only on the final object passed
    metricality = objects[-1].metrical

    # concatenate iois and create new Sequence
    iois = np.concatenate([obj.iois for obj in objects])
    seq = thebeat.core.Sequence(iois, metrical=metricality)

    # For Sequence objects we're done:
    if isinstance(objects[0], thebeat.core.Sequence):
        return seq

    # Otherwise we get the stimuli from the StimSequence object, join them and return a new StimSequence
    if isinstance(objects[0], thebeat.core.StimSequence):
        all_stimuli = []
        for obj in objects:
            all_stimuli += obj.stim_objects
        stimseq = thebeat.core.StimSequence(stimulus=all_stimuli, sequence=seq, name=name)
        return stimseq
