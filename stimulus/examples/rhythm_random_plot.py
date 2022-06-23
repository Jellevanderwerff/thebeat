from stimulus import *

"""
In this example we generate a random rhythm from a number of allowed note values.
Then, we create a stimuli, and make a RhythmTrial out of it, which we can play.
"""

# Generate random rhythm with 1/4th, 1/8th, or 1/16th notes
rhythm = Rhythm.generate_random_rhythm(allowed_note_values=[4, 8, 16], beat_ms=1000)

# Plot, either with or without staff
rhythm.plot_rhythm()
rhythm.plot_rhythm(print_staff=True)

# Make a stimulus
stim = Stimulus.generate(offramp=10, ramp='raised-cosine')

# We need multiple stimuli, so make a Stimuli object from a repeated Stimulus
# (number of repeats is the number of onsets in the rhythm)
stims = Stimuli.from_stim(stim, repeats=len(rhythm.onsets))

# Make a RhythmTrial and play!
trial = RhythmTrial(rhythm, stims)
trial.play()
