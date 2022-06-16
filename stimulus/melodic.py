from mingus.containers import Note
from mingus.core.scales import Major, NaturalMinor
import re
import os
from stimulus import Stimulus


def stims_from_notes(notes_str, event_duration=50, onramp=0, offramp=0):
    """
    Get stimulus objects on the basis of a provided string of notes.
    For instance: 'CDEC' returns a list of four Stimulus objects.
    Alternatively, one can use 'C4D4E4C4'. In place of
    silences one can use an 'X'.

    """
    notes = re.findall(r"[A-Z][0-9]?", notes_str)

    freqs = []

    for note in notes:
        if len(note) > 1:
            note, num = tuple(note)
            freqs.append(Note(note, int(num)).to_hertz())
        else:
            if note == 'X':
                freqs.append(None)
            else:
                freqs.append(Note(note).to_hertz())

    stims = []

    for freq in freqs:
        if freq is None:
            stims.append(Stimulus.rest(event_duration))
        else:
            stims.append(Stimulus.generate(freq=freq,
                                           duration=event_duration,
                                           onramp=onramp,
                                           offramp=offramp))

    return stims


def get_major_scale_freqs(starting_note):
    if len(starting_note) > 1:
        n, i = tuple(starting_note)
        i = int(i)
    else:
        n = starting_note
        i = 4

    notes = [Note(note, i) for note in Major(n).ascending()]

    return [note.to_hertz() for note in notes]


def get_minor_scale_freqs(starting_note):
    if len(starting_note) > 1:
        n, i = tuple(starting_note)
        i = int(i)
    else:
        n = starting_note
        i = 4

    notes = [Note(note, i) for note in NaturalMinor(n).ascending()]

    return [note.to_hertz() for note in notes]




