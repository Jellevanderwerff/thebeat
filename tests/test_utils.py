from thebeat import Sequence
import thebeat.utils
import numpy as np


def test_phasedifferences():
    seq = Sequence([499, 501, 505, 501])

    diffs = thebeat.utils.get_phase_differences(seq, 500)
    assert np.all(diffs == np.array([0., -0.72, 0., 3.6, 4.32]))
