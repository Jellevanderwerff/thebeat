from combio.rhythm import *
import numpy as np
import pytest


def test_rhythm():
    # generate random rhythm
    rrhythm = Rhythm.generate_random_rhythm([4, 8])
    assert rrhythm

    # combine two rhythms

    rhythm1 = Rhythm.generate_random_rhythm([4, 8])
    rhythm2 = Rhythm.generate_random_rhythm([4, 8])
    combined_rhythm = rhythm1 + rhythm2
    assert len(combined_rhythm) == len(rhythm1) + len(rhythm2)

    rhythm = Rhythm.from_note_values([4, 4, 4, 4])
    assert len(rhythm) == 4
    assert np.all(rhythm.iois == [500, 500, 500, 500])

    rhythm = Rhythm.from_iois([500, 1000, 500], (4, 4), 500)
    assert rhythm.beat_ms == 500

    with pytest.raises(ValueError):
        rhythm = Rhythm.from_iois([250, 250, 500, 250, 250, 250], (4, 4), 500)
