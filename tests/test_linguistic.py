import combio.linguistic


def test_stress():
    seq = combio.linguistic.generate_stress_timed_sequence(10, 10)
    assert len(seq.onsets) == 101
