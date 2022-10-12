import thebeat
import numpy as np
import pytest


def test_rhythm():
    # generate random rhythm
    rrhythm = thebeat.rhythm.Rhythm.generate_random_rhythm(allowed_note_values=[4, 8])
    assert rrhythm

    # combine two rhythms

    rhythm1 = thebeat.rhythm.Rhythm.generate_random_rhythm(allowed_note_values=[4, 8])
    rhythm2 = thebeat.rhythm.Rhythm.generate_random_rhythm(allowed_note_values=[4, 8])
    combined_rhythm = rhythm1 + rhythm2
    assert len(combined_rhythm) == len(rhythm1) + len(rhythm2)

    rhythm = thebeat.rhythm.Rhythm.from_note_values([4, 4, 4, 4])
    assert len(rhythm) == 4
    assert np.all(rhythm.iois == [500, 500, 500, 500])

    rhythm = thebeat.rhythm.Rhythm([500, 1000, 500], (4, 4), 500)
    assert rhythm.beat_ms == 500

    with pytest.raises(ValueError):
        rhythm = thebeat.rhythm.Rhythm([250, 250, 500, 250, 250, 250], (4, 4), 500)
