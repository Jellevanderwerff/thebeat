from stimulus import Stimulus, Sequence, StimulusSequence

# Example of a sequence
sequence = Sequence.generate_random_uniform(n=5, a=200, b=600)
print(sequence)
sequence.change_tempo(2)
print(sequence)

# Example of a stimulus
stimulus = Stimulus.from_wav('click01.wav')
stimulus.plot(title="Waveform for click01.wav")

# Example of a sound sequence with the same sound used throughout
stimulus_sequence = StimulusSequence(stimulus, sequence)
stimulus_sequence.plot("StimulusSequence with same sound throughout")
stimulus_sequence.write_wav('sequence_samesound.wav')

# Example of a sound sequence with different sounds for each event
# (we pass a list of Stimulus objects of equal length)
sequence = Sequence.generate_isochronous(n=5, ioi=500)

tone_heights = [500, 300, 600, 100, 300]
stimuli = [Stimulus.generate(freq=tone_height) for tone_height in tone_heights]

stimulus_sequence = StimulusSequence(stimuli, sequence)
stimulus_sequence.plot("StimulusSequence with different sounds")
stimulus_sequence.write_wav('stimulus_sequence.wav')

# All Sequence and Stimulus manipulation methods you can also use for StimulusSequence objects:
stimulus_sequence = StimulusSequence(Stimulus.generate(freq=440, onramp=10, offramp=10),
                                     Sequence.generate_isochronous(n=5, ioi=500))

stimulus_sequence.change_amplitude(factor=0.01)
stimulus_sequence.plot("Manipulation changed amplitude")

# Accelerando
sequence = Sequence.generate_isochronous(n=10, ioi=500)
stimulus = Stimulus.generate(freq=440, osc='sine', duration=10)
stimulus_sequence = StimulusSequence(stimulus, sequence)
stimulus_sequence.plot("Accelerando before")
stimulus_sequence.change_tempo_linearly(total_change=2)
stimulus_sequence.plot("Accelerando after")
# stimulus_sequence.play()

# Ritardando
stimulus_sequence = StimulusSequence(Stimulus.generate(freq=440, duration=50, onramp=10, offramp=10),
                                     Sequence.generate_isochronous(n=10, ioi=500))
stimulus_sequence.plot("Ritardando before")
stimulus_sequence.change_tempo_linearly(total_change=0.5)
stimulus_sequence.plot("Ritardando after")
# stimulus_sequence.play()