import pytest

import stimulus



def test_random_uniform():
	sequence = stimulus.Sequence.generate_random_uniform(n=5, a=200, b=600)
	assert len(sequence.iois) == 4
	assert len(sequence.onsets) == 5


def test_isochronous():
	sequence = stimulus.Sequence.generate_isochronous(n=10, ioi=500)
	assert len(sequence.iois) == 9
	assert len(sequence.onsets) == 10


def test_exception_demo():
	sequence = stimulus.Sequence.generate_isochronous(n=10, ioi=500)
	sequence.change_tempo(0.5)
	with pytest.raises(ValueError, match="Please provide a factor larger than 0."):
		sequence.change_tempo(-1)


def test_exception_demo_without():
	sequence = stimulus.Sequence.generate_isochronous(n=10, ioi=500)
	sequence.change_tempo(-42)

