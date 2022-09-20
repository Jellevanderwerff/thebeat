import combio.linguistic


def test_stress():
    seq = combio.linguistic.generate_stress_timed_sequence(10)
    assert len(seq.onsets) == 10


def test_mora():
    seq = combio.linguistic.generate_moraic_sequence(10)
    assert seq.duration_s == 5.0
