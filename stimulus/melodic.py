from mingus.containers import Note
from mingus.core.scales import Major, NaturalMinor


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




