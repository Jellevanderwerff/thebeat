import combio.melody
import combio.rhythm
import numpy as np
import pytest


def test_melody(tmp_path):
    mel = combio.melody.Melody.generate_random_melody(n_bars=2, key='G', octave=5)
    fig, ax = mel.plot_melody(suppress_display=True)
    assert fig, ax

    r = combio.rhythm.Rhythm.from_note_values([4, 4, 4, 4])
    m = combio.melody.Melody(r, 'CEGC', octave=3)
    assert list(m.note_values) == [4, 4, 4, 4]

    mel = combio.melody.Melody.generate_random_melody(n_bars=2, key='G', octave=5)
    samples, fs = mel.synthesize_and_return(n_channels=2, metronome=True)
    assert isinstance(samples, np.ndarray)
    assert fs

    mel.synthesize_and_write(tmp_path / 'test_melody', n_channels=2, metronome=True)
