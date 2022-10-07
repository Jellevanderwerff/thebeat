import combio.linguistic


def test_stress():
    seq = combio.linguistic.generate_stress_timed_sequence(10)
    assert len(seq.onsets) == 10


def test_mora():
    seq = combio.linguistic.generate_moraic_sequence(10, foot_ioi=600)
    assert seq.duration == 6000
