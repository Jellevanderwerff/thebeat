
class RhythmTrial:

    def __init__(self,
                 stimuli: Union[core.stimulus.Stimulus, Iterable[core.stimulus.Stimulus]],
                 rhythm: Rhythm,
                 name: str = None):

        # Check if correct objects were passed for stimuli
        if isinstance(stimuli, core.stimulus.Stimulus):
            stimuli = [stimuli] * len(rhythm)
        elif hasattr(stimuli, '__iter__'):
            pass
        else:
            raise ValueError("Please provided either a Stimulus object, "
                             "or an iterable of Stimulus objects as the first argument")

        # Check if correct objects was passed for rhythm
        if not isinstance(rhythm, Rhythm):
            raise ValueError("Please provide a Rhythm object as the second argument")

        # Initialize namedtuple
        self.Event = namedtuple('Event', 'layer onset ioi duration note_value samples')

        # Then add events as namedtuples to self.events
        events = []
        layer_id = [0] * len(stimuli)
        self.events = self._add_events(events, rhythm, stimuli, layer_id)

        # Save provided trial name
        self.name = name

        # Save rhythmic attributes
        self.time_signature = rhythm.time_signature
        self.beat_ms = rhythm.beat_ms
        self.n_bars = rhythm.n_bars
        self.note_values = rhythm.get_note_values
        self.bar_duration = np.sum(rhythm.iois) / rhythm.n_bars
        self.total_duration = np.sum(rhythm.iois)

        # Save stimulus attributes
        self.fs = stimuli[0].fs
        self.n_channels = stimuli[0].n_channels

        # Make initial sound
        self.samples = self._make_sound(self.events)

        # We start with one layer
        self.n_layers = 1

    def _make_sound(self, provided_events):
        # Generate an array of silence that has the length of all the onsets + one final stimulus.
        # In the case of a metrical sequence, we add the final ioi
        # The dtype is important, because that determines the values that the magnitudes can take.

        length = self.total_duration / 1000 * self.fs
        if int(length) != length:  # let's avoid rounding issues
            warnings.warn("Number of frames was rounded off to nearest integer ceiling. "
                          "This shouldn't be much of a problem.")

        array_length = int(np.ceil(length))

        if self.n_channels == 1:
            samples = np.zeros(array_length)
        else:
            samples = np.zeros((array_length, 2))

        for event in provided_events:
            if event.samples is not None:
                if self.n_channels == 1:
                    start_pos = int(event.onset / 1000 * self.fs)
                    end_pos = int(start_pos + event.samples.shape[0])
                    try:
                        samples[start_pos:end_pos] = samples[start_pos:end_pos] + event.samples
                    except ValueError:
                        raise ValueError("Could not make sound. Probably the final stimulus is longer than the "
                                         "final note value.")
                elif self.n_channels == 2:
                    start_pos = int(event.onset / 1000 * self.fs)
                    end_pos = int(start_pos + (event.duration / 1000 * self.fs))
                    try:
                        samples[start_pos:end_pos, :2] = samples[start_pos:end_pos, :2] + event.samples
                    except ValueError:
                        raise ValueError("Could not make sound. Probably the final stimulus is longer than the "
                                         "final note value.")

        if np.max(samples) > 1:
            warnings.warn("Sound was normalized to avoid distortion. If undesirable, change amplitude of the stims.")
            return core.helpers.normalize_audio(samples)
        else:
            return samples

    def add_layer(self, rhythm: Rhythm, stimuli: Union[core.stimulus.Stimulus, Iterable[core.stimulus.Stimulus]]):

        if self.n_layers > 3:
            raise ValueError("Can, for now, only handle 4 layers.")

        # Check if right objects were passed
        if not isinstance(rhythm, Rhythm):
            raise ValueError("Please provide a Rhythm object as the first argument.")

        if not all(isinstance(stim, core.stimulus.Stimulus) for stim in stimuli) or not isinstance(stimuli,
                                                                                                   core.stimulus.Stimulus):
            raise ValueError("Please provide either an iterable (e.g. a list) with Stimulus objects as the second "
                             "argument, or a single Stimulus object.")

        # multiply if Stimulus object was passed
        if isinstance(stimuli, core.stimulus.Stimulus):
            stimuli = [stimuli] * len(rhythm.onsets)

        # fs
        if not all(stimulus.fs == stimuli[0].fs for stimulus in stimuli):
            raise ValueError("The provided stimuli do not all have the same sampling frequency.")
        elif not stimuli[0].fs == self.fs:
            raise ValueError("The provided stimuli have a different sampling frequency than the stimuli "
                             "currently in this trial.")

        # n channels
        if not all(stimulus.n_channels for stimulus in stimuli):
            raise ValueError("The provided stimuli do not all have the same number of channels.")
        elif not stimuli[0].n_channels == self.n_channels:
            raise ValueError("The provided stimuli do not have the same number of channels as the stimuli "
                             "currently in this trial.")
        # beat ms
        if not rhythm.beat_ms == self.beat_ms:
            raise ValueError("The provided rhythm has a different beat_ms than the rhythm "
                             "currently in this trial.")

        # time signature
        if not rhythm.time_signature == self.time_signature:
            raise ValueError("The provided rhythm has a different time signature than the rhythm "
                             "currently in this trial.")

        # add layer to self.events
        layer_id = [self.n_layers] * len(stimuli)
        self.events = self._add_events(self.events, rhythm, stimuli, layer_id)

        # make sound and save to self.samples
        self.samples = self._make_sound(self.events)

        # add one to the number of layers
        self.n_layers += 1

    def play(self, loop=False, metronome=False, metronome_amplitude=1):
        core.helpers.play_samples(self.samples, self.fs, self.beat_ms, loop, metronome, metronome_amplitude)

    def plot_rhythm(self,
                    filepath=None,
                    print_staff=True,
                    lilypond_percussion_notes=None,
                    stem_directions=None,
                    suppress_display=False):
        """

        Parameters
        ----------
        filepath
        print_staff
        lilypond_percussion_notes:
            List of lilypond percussion notes for each layer. Defaults to ['bd', 'snare', 'hihat'].
            See possible options here: https://lilypond.org/doc/v2.23/Documentation/notation/percussion-notes
        stem_directions
        suppress_display

        Returns
        -------

        """

        lp = _get_lp_from_events(self.events,
                                 self.n_layers,
                                 self.time_signature,
                                 print_staff=print_staff,
                                 lilypond_percussion_notes=lilypond_percussion_notes,
                                 stem_directions=stem_directions)

        _plot_lp(lp, filepath=filepath, suppress_display=suppress_display)

        warnings.warn("Time signatures aren't implemented here yet! Forgot that...")

    def plot_waveform(self, title=None):
        if title:
            title = title
        else:
            if self.name:
                title = f"Waveform of {self.name}"
            else:
                title = "Waveform of RhythmTrial"

        core.helpers.plot_waveform(self.samples, self.fs, self.n_channels, title)

    def write_wav(self, out_path='.',
                  metronome=False,
                  metronome_amplitude=1):
        """
        Writes audio to disk.
        """

        core.helpers.write_wav(self.samples, self.fs, out_path, self.name, metronome, self.beat_ms, metronome_amplitude)


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

