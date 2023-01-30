import thebeat
import numpy as np
import pytest


def test_rhythm():
    # generate random rhythm
    rrhythm = thebeat.music.Rhythm.generate_random_rhythm(allowed_note_values=[4, 8])
    assert rrhythm

    # combine two rhythms
    rhythm1 = thebeat.music.Rhythm.generate_random_rhythm(allowed_note_values=[4, 8])
    rhythm2 = thebeat.music.Rhythm.generate_random_rhythm(allowed_note_values=[4, 8])
    combined_rhythm = rhythm1 + rhythm2
    assert sum(combined_rhythm.iois) == sum(rhythm1.iois) + sum(rhythm2.iois)

    rhythm = thebeat.music.Rhythm.from_note_values([4, 4, 4, 4])
    assert np.all(rhythm.iois == [500, 500, 500, 500])

    rhythm = thebeat.music.Rhythm([500, 1000, 500], (4, 4), 500)
    assert rhythm.beat_ms == 500

    with pytest.raises(ValueError):
        rhythm = thebeat.music.Rhythm([250, 250, 500, 250, 250, 250], (4, 4), 500)

    rhythm = thebeat.music.Rhythm([500, 500, 500, 500])
    rhythm *= 4
    assert rhythm.n_bars == 4
    assert len(rhythm.iois) == 16


@pytest.mark.mpl_image_compare
def test_rhythm_plot():
    rhythm = thebeat.music.Rhythm([500, 500, 500, 500])
    fig, ax = rhythm.plot_rhythm(suppress_display=True)
    return fig
