from mingus.containers import Note
from mingus.core.scales import Major, NaturalMinor
import re
import os


def notes_to_freqs(notes_str):
    """Converts a string of notes to frequencies. E.g. 'CCDDEFC', or 'C4D4
    """
    notes = re.findall(r"[A-Z][0-9]?", notes_str)

    freqs = []

    for note in notes:
        if len(note) > 1:
            note, num = tuple(note)
            freqs.append(Note(note, int(num)).to_hertz())
        else:
            freqs.append(Note(note).to_hertz())

    return freqs


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




