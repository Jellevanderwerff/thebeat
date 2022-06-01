from mingus.containers import Note, Bar
from mingus.core.scales import Major, NaturalMinor
from mingus.extra import lilypond
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


def plot_note_values(filename, note_values, time_signature):

    b = Bar(meter=time_signature)

    for note_value in note_values:
        b.place_notes('G-4', note_value)

    lp = lilypond.from_Bar(
        b) + '\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""\noddFooterMarkup = ""\nevenFooterMarkup = ""\n}'

    lilypond.save_string_and_execute_LilyPond(lp, filename, '-dbackend=eps -dresolution=600 --png -s')

    filenames = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi']
    filenames = [filename[:-4] + x for x in filenames]

    for file in filenames:
        os.remove(file)


