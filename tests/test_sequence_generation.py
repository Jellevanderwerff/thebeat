

import stimulus



def test_random_uniform():
	sequence = stimulus.Sequence.generate_random_uniform(n=5, a=200, b=600)
	assert len(sequence.iois) == 4
	assert len(sequence.onsets) == 5


def test_isochronous():
	sequence = stimulus.Sequence.generate_isochronous(n=10, ioi=500)
	assert len(sequence.iois) == 9
	assert len(sequence.onsets) == 10

