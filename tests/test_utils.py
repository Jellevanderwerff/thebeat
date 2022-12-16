from thebeat import Sequence
import thebeat.utils
import numpy as np


def test_phasedifferences():
    seq = Sequence([499, 501, 505, 501])

    diffs = list(thebeat.utils.get_phase_differences(seq, 500))

    assert diffs == [359.2814371257485, 0.0, 3.5643564356435644, 4.2772277227722775]
