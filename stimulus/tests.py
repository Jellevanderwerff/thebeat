from stimulus import *
from mingus.core.value import determine
from mingus.containers import Bar, Track
from mingus.core.value import tuplet


stims = stims_from_notes('CXDEC', onramp=10, offramp=10)
seq = Rhythm.from_note_values([4, 8, 8, 4, 4], time_signature=(4, 4), quarternote_ms=500)

stim_seq = StimulusSequence(stims, seq)

stim_seq.plot_waveform()
stim_seq.play(metronome=True)

