from stimulus import *
import random


rhythm1 = Rhythm.from_iois([500, 500, 500, 500],
                          time_signature=(4, 4),
                          quarternote_ms=500)

rhythm2 = Rhythm.from_note_values([4, 8, 8, 4, 4], time_signature=(4, 4), quarternote_ms=500)

rhythm = rhythm1 + rhythm2

rhythm.plot()


