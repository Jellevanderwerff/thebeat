from mingus.containers import Note
import re


def notes_to_freqs(notes_str):
    """Converts a string of notes to frequencies. E.g. 'CCDDEFC', or 'C4D4
    """
    notes = re.findall(r"[A-Z][0-9]?", notes_str)

    freqs = []

    for note in notes:
        if len(note) > 1:
            tupel = tuple([x for x in note])
            note, num = tupel
            freqs.append(Note(note, int(num)).to_hertz())
        else:
            freqs.append(Note(note).to_hertz())

    return freqs
