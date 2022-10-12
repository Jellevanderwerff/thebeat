import thebeat.melody
import thebeat.rhythm
import numpy as np
import pytest


def test_melody(tmp_path):
    mel = thebeat.melody.Melody.generate_random_melody(n_bars=2, key='G', octave=5)
    fig, ax = mel.plot_melody(suppress_display=True)
    assert fig, ax

    r = thebeat.rhythm.Rhythm.from_note_values([4, 4, 4, 4])
    m = thebeat.melody.Melody(r, 'CEGC', octave=3)
    assert list(m.note_values) == [4, 4, 4, 4]

    mel = thebeat.melody.Melody.generate_random_melody(n_bars=2, key='G', octave=5)
    samples, fs = mel.synthesize_and_return(n_channels=2, metronome=True)
    assert isinstance(samples, np.ndarray)
    assert fs

    mel.synthesize_and_write(tmp_path / 'test_melody', n_channels=2, metronome=True)
