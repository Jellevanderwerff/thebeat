from stimulus import *

"""
Now, sometimes we want a trial to be 'metrical' in the sense that after the final stimulus has been played
we get another IOI, so that we can combine it with other sequences. Here's an example.
"""

# First create a metrical sequence
seq1 = Sequence.generate_isochronous(n=4, ioi=500, metrical=True)

# Now if we print that sequence, we can already see that now there's an equal number of onsets
# to the number of inter-onset intervals. In non-metrical sequences, there's n-1 IOIs for n onsets.
print(seq1)

# An advantage of metrical sequences is that we can combine them with other metrical sequences, like so:
seq2 = Sequence.generate_isochronous(n=2, ioi=300, metrical=True)
seq_joined = seq1 + seq2
print(seq_joined)

# Now let's create a stimtrial, this time using stims that we get from a number of notes
stims = Stimuli.from_notes('CCCGGG')
trial = StimTrial(stims, seq_joined, name="joined metrical sequences")

# And play/plot etc.
trial.plot_waveform()
trial.play()
trial.write_wav()
