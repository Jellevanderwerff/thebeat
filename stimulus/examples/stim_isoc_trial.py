from stimulus import *

"""
In this example we create a simple trial that uses randomized A and B tones and plays those isochronously. 
"""

# First, let's create an isochronous sequence of 10 events, with an inter-onset interval of 500
seq = Sequence.generate_isochronous(n=10, ioi=500)

# We can always print an object to get some info about onsets etc.
print(seq)

# Now, let's create the A (440 Hz) and B (600 Hz) stimulus objects
stim_a = Stimulus.generate(freq=440, name="A")
stim_b = Stimulus.generate(freq=600, name="B")

# Make those into a Stimuli object, and randomize those
stims = Stimuli.from_stims([stim_a, stim_b], repeats=5)
stims.randomize()

# Now we can create a trial and plot/write/play it
trial = StimTrial(stims, seq, name="my first StimTrial")
print(trial)
trial.plot_waveform()
trial.write_wav('isoc_trial.wav')
trial.play()
