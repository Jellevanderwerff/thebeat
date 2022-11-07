from thebeat import Sequence
import thebeat.utils
import numpy as np


def test_phasedifferences():
    seq = Sequence([499, 501, 505, 501])

    diffs = thebeat.utils.get_phase_differences(seq, 500)
    print(list(diffs))

    assert np.all(diffs == [0.0, 359.2785571142285, 0.0, 3.592814371257485, 4.311377245508982])
