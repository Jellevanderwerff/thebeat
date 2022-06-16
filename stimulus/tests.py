from stimulus import *
import random
from mingus.containers import Note, Bar, Track
import numpy as np
from mingus.extra import lilypond
from skimage import io, img_as_float
import matplotlib.pyplot as plt


rhythm = random_rhythmic_sequence(1, [4, 8, 16], (4, 4), 500)

rhythm.plot_rhythm(print_staff=False)





