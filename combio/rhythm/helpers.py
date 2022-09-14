import os
import subprocess
import warnings

import numpy as np
import skimage
from matplotlib import pyplot as plt
from mingus.containers import Bar
from mingus.extra import lilypond


def all_possibilities(nums, target):
    """
    I stole this code
    """

    res = []
    nums.sort()

    def dfs(left, path):
        if not left:
            res.append(path)
        else:
            for val in nums:
                if val > left:
                    break
                dfs(left - val, path + [val])

    dfs(target, [])

    return res


def all_rhythmic_ratios(allowed_note_values, time_signature, target_n: int = None):
    common_denom = np.lcm(np.lcm.reduce(allowed_note_values), time_signature[1])

    allowed_numerators = common_denom // np.array(allowed_note_values)
    full_bar = time_signature[0] * (1 / time_signature[1])
    target = full_bar * common_denom

    possibilities = all_possibilities(allowed_numerators, target)

    out_list = [(np.array(result) / common_denom) for result in
                possibilities]

    if target_n is not None:
        out_list = [rhythm for rhythm in out_list if len(rhythm) == target_n]
        if len(out_list) == 0:
            raise ValueError("No random rhythms exist that adhere to these parameters. "
                             "Try providing different parameters.")

    return out_list


def get_lp_from_track(t, print_staff: bool):
    """
    Internal method for plotting a mingus Track object via lilypond.
    """

    remove_footers = """\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""
oddFooterMarkup = ""\nevenFooterMarkup = ""\n} """
    remove_staff = '{ \stopStaff \override Staff.Clef.color = #white'

    if print_staff is True:
        lp = '\\version "2.10.33"\n' + lilypond.from_Track(t) + remove_footers
    elif print_staff is False:
        lp = '\\version "2.10.33"\n' + remove_staff + lilypond.from_Track(t)[1:] + remove_footers
    else:
        raise ValueError("Wrong value specified for print_staff.")

    return lp


def get_lp_from_events(provided_events,
                       n_layers: int,
                       time_signature: tuple,
                       print_staff: bool = True,
                       lilypond_percussion_notes=None,
                       stem_directions=None):
    if any(event.samples is None for event in provided_events):
        warnings.warn("'Rests' are plotted as empty spaces, not as rests. Please check manually whether"
                      "the plot makes sense.")

    if lilypond_percussion_notes is None:
        lilypond_percussion_notes = ['bd', 'snare', 'hihat']

    if stem_directions is None:
        stem_directions = ['', '', '']
    else:
        stem_directions = ['\override Stem.direction = #' + stem_direction for stem_direction in stem_directions]

    if n_layers > 3:
        raise ValueError("Can maximally plot three layers.")

    if print_staff is True:
        print_staff_lp = ""
    else:
        print_staff_lp = "\\stopStaff \override Staff.Clef.color = #white "

    layers_list = []

    for layer in range(n_layers):
        bars = []
        events = [event for event in provided_events if event.layer == layer]

        bar = ''
        b = Bar(meter=time_signature)

        for event in events:
            note_value = event.note_value
            b.place_rest(note_value)  # This is only to keep track of the number of notes in a bar
            if event.samples is not None:
                note = lilypond_percussion_notes[layer] + str(note_value) + ' '
            else:
                note = 's' + str(note_value) + ' '
            bar += note
            if b.current_beat == b.length:
                bars.append("{ " + bar + "}\n")
                b = Bar(meter=time_signature)
                bar = ''

        layers_list.append(bars)

    voice_names = ['voiceOne', 'voiceTwo', 'voiceThree']
    layer_names = ['uno', 'dos', 'tres']

    string_firstbit = ''
    string_secondbit = '\\new DrumStaff << '

    for layer_i in range(len(layers_list)):
        bars = ' '.join(layers_list[layer_i])
        bars = print_staff_lp + bars
        layer_string = f"{layer_names[layer_i]} = \drummode {{ {stem_directions[layer_i]} {bars} }} \n"
        string_firstbit += layer_string
        staves_string = "\\new DrumVoice { \\%s \\%s }\n" % (voice_names[layer_i], layer_names[layer_i])
        string_secondbit += staves_string

    string_secondbit += ' >>'

    out_string = string_firstbit + string_secondbit

    remove_footers = """\n\paper {\nindent = 0\mm\nline-width = 110\mm\noddHeaderMarkup = ""\nevenHeaderMarkup = ""
oddFooterMarkup = ""\nevenFooterMarkup = ""\n} """

    lp = '\\version "2.10.33"\n' + out_string + remove_footers

    return lp


def plot_lp(lp, filepath, suppress_display):
    # This is the same each time:

    if filepath:
        location, filename = os.path.split(filepath)
        if location == '':
            location = '.'
    else:
        location = '.'
        filename = 'rhythm.png'

    # run subprocess
    if filename.endswith('.eps'):
        command = f'lilypond -dbackend=eps --silent -dresolution=600 --eps -o {filename[:-4]} {filename[:-4] + ".ly"}'
        to_be_removed = ['.ly']
    elif filename.endswith('.png'):
        command = f'lilypond -dbackend=eps --silent -dresolution=600 --png -o {filename[:-4]} {filename[:-4] + ".ly"}'
        to_be_removed = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi', '.ly']
    else:
        raise ValueError("Can only export .png or .eps files.")

    # write lilypond string to file
    with open(os.path.join(location, filename[:-4] + '.ly'), 'w') as file:
        file.write(lp)

    subprocess.Popen(command, shell=True, cwd=location).wait()

    image = skimage.img_as_float(skimage.io.imread(filename))

    # Select all pixels almost equal to white
    # (almost, because there are some edge effects in jpegs
    # so the boundaries may not be exactly white)
    white = np.array([1, 1, 1])
    mask = np.abs(image - white).sum(axis=2) < 0.05

    # Find the bounding box of those pixels
    coords = np.array(np.nonzero(~mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)

    out = image[top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]]

    # show plot
    if not filepath and not suppress_display:
        plt.imshow(out)
        plt.axis('off')
        plt.show()
    elif filename.endswith('.png') and not suppress_display:
        plt.imshow(out)
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight')
    else:
        pass

    # remove files
    if filepath:
        filenames = [filename[:-4] + x for x in to_be_removed]
    else:
        to_be_removed = ['-1.eps', '-systems.count', '-systems.tex', '-systems.texi', '.ly', '.png']
        filenames = ['rhythm' + x for x in to_be_removed]

    for file in filenames:
        os.remove(os.path.join(location, file))
