import combio.linguistic


def test_stress():
    seq = combio.linguistic.generate_stress_timed_sequence(10, 10)
    assert len(seq.onsets) == 100


def test_mora():
    seq = combio.linguistic.generate_trimoraic_sequence(10)
    assert seq.duration_s == 7.5
