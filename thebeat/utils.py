import numpy as np
from thebeat.core import Sequence, StimSequence
from typing import Optional


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

    if not all(isinstance(obj, Sequence) for obj in objects) and not all(
            isinstance(obj, StimSequence) for obj in objects):
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
    seq = Sequence(iois, metrical=metricality)

    # For Sequence objects we're done:
    if isinstance(objects[0], Sequence):
        return seq

    # Otherwise we get the stimuli from the StimSequence object, join them and return a new StimSequence
    if isinstance(objects[0], StimSequence):
        all_stimuli = []
        for obj in objects:
            all_stimuli += obj.stim_objects
        stimseq = StimSequence(stimulus=all_stimuli, sequence=seq, name=name)
        return stimseq
