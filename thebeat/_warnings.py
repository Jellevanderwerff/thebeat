framerounding_stimseq = Warning("For one ore more of the used sounds, the exact start or end positions in frames (i.e. "
                                "samples) were rounded off to the neirest integer ceiling. This shouldn't be a problem."
                                "To get rid of this warning, try rounding off the onsets in the passed Sequence object "
                                "by calling Sequence.round_off_onsets() before passing the object to the StimSequence "
                                "constructor.")

framerounding_melody = Warning("For one ore more of the used sounds, the exact start or end positions in frames (i.e. "
                               "samples) were rounded off to the neirest integer ceiling. This shouldn't be a problem."
                               "To get rid of this warning, try using a sampling frequency of 48000 Hz, or a different"
                               "beat_ms.")

framerounding_soundsynthesis = Warning("During sound synthesis, the number of frames was rounded off."
                                       "This shouldn't be a problem."
                                       "To get rid of this warning, try using a combination of sound duration "
                                       "(in seconds) and sampling frequency that results in integer values.")

normalization = Warning("Sound was normalized to avoid distortion. If undesirable, change amplitude of the "
                        "sounds.")


phases_t_at_zero = Warning("The first onset of the test sequence was at t=0.\nThis would result in a phase "
                           "difference that is always 0, which is not very informative.\nAs such, the first onset "
                           "was discarded.\nIf you want the first onset at a different time than zero, "
                           "use the Sequence.from_onsets() method to create the Sequence object.")
