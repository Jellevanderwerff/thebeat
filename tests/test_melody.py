import thebeat.melody
import thebeat.rhythm
import numpy as np
import pytest


def test_melody(tmp_path):
    rng = np.random.default_rng(seed=123)
    r = thebeat.rhythm.Rhythm.from_note_values([4, 4, 4, 4])
    m = thebeat.melody.Melody(r, 'CEGC', octave=3)
    assert list(m.note_values) == [4, 4, 4, 4]

    mel = thebeat.melody.Melody.generate_random_melody(n_bars=2, key='G', octave=5, rng=rng)
    samples, fs = mel.synthesize_and_return(n_channels=2, metronome=True)
    assert isinstance(samples, np.ndarray)
    assert fs

    mel.synthesize_and_write(tmp_path / 'test_melody', n_channels=2, metronome=True)


@pytest.mark.mpl_image_compare
def test_melody_plot():
    rng = np.random.default_rng(seed=123)
    mel = thebeat.melody.Melody.generate_random_melody(n_bars=2, key='G', octave=4, rng=rng)
    fig, ax = mel.plot_melody(suppress_display=True)
    return fig
