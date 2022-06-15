from stimulus import *

rhythm1 = random_rhythmic_sequence(1,
                                   allowed_note_values=[4, 8, 16],
                                   time_signature=(4, 4),
                                   quarternote_ms=500)

rhythm2 = random_rhythmic_sequence(1,
                                   allowed_note_values=[4, 8, 16],
                                   time_signature=(4, 4),
                                   quarternote_ms=500)


rhythm_joined = join_rhythms([rhythm1, rhythm2])


print(rhythm_joined.onsets)

stimseq = StimulusSequence(Stimulus.generate(), rhythm_joined)

stimseq.play()

