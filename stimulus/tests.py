from stimulus import *
import random
from mingus.containers import Note, Bar, Track
import numpy as np
from mingus.extra import lilypond
from skimage import io, img_as_float
import matplotlib.pyplot as plt


rhythm = Rhythm.from_note_values([4, 4, 4, 4], (4, 4), quarternote_ms=500)

rhythm.plot_rhythm()

