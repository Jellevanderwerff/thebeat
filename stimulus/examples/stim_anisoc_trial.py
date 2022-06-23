from stimulus import *

"""
In this example we create a simple trial that uses an imported sound and plays those at random onsets. 
"""

# First we create a sequence of 10 events that uses random inter-onset intervals sampled from a uniform distribution
seq = Sequence.generate_random_uniform(n=10, a=400, b=600)

# Then, let's use one stimulus sound that we get from a .wav file
stim = Stimulus.from_wav('click01.wav')
# Repeat it 10 times, notice how we now create a Stimuli (not Stimulus) object
stims = Stimuli.from_stim(stim, repeats=10)

# Now we can create a stimtrial, plot it, write it, and play it with a metronome sound
trial = StimTrial(stims, seq, name="random trial")
trial.plot_waveform()
trial.write_wav('anisoc_trial.wav')
trial.play(metronome=True)
