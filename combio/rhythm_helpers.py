



def _get_lp_from_events(provided_events,
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

