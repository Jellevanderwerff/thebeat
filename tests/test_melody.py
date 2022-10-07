import combio.melody
import combio.rhythm
import numpy as np


def test_melody():
    mel = combio.melody.Melody.generate_random_melody(n_bars=2, key='G', octave=5)
    fig, ax = mel.plot_melody(suppress_display=True)
    assert fig, ax

    r = combio.rhythm.Rhythm.from_note_values([4, 4, 4, 4])
    m = combio.melody.Melody(r, 'CEGC', octave=3)
    assert list(m.note_values) == [4, 4, 4, 4]
