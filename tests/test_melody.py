import combio.melody


def test_melody():
    mel = combio.melody.Melody.generate_random_melody(n_bars=2, key='G', octave=5)
    fig, ax = mel.plot_melody()
    assert fig, ax
