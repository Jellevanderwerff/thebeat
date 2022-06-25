from stimulus import *
from mingus.extra import lilypond

rhythm = Rhythm.from_note_values([4] * 4)
stim = Stimulus.generate(freq=440, offramp=10)
stims = Stimuli.from_stim(stim, repeats=len(rhythm.onsets))
trial = RhythmTrial(rhythm, stims)
trial.add_layer(rhythm, stims)
rhythm2 = Rhythm.from_note_values([16] * 16)
stims2 = Stimuli.from_stim(stim, repeats=len(rhythm2.onsets))
trial.add_layer(rhythm2, stims2)

layers_list = []
layers_dict = {0: 'bd', 1: 'snare', 2: 'hihat'}

if trial.n_layers > 3:
    raise ValueError("Can only do three layers unfortunately.")

for layer in range(trial.n_layers):
    bars = []
    events = [event for event in trial.events if event.layer == layer]
    print(events)
    bar = ''
    b = Bar(meter=trial.time_sig)
    for event in events:
        note_value = int(event.note_value)
        print(note_value)
        b.place_rest(note_value)
        note = layers_dict[layer] + str(note_value) + ' '
        bar += note
        if b.current_beat == b.length:
            bars.append("{ " + bar + "}")
            b = Bar(meter=trial.time_sig)
            bar = ''

    layers_list.append(bars)

voice_names = ['voiceOne', 'voiceTwo', 'voiceThree']
layer_names = ['uno', 'dos', 'tres']

string_firstbit = ''
string_secondbit = '\\new DrumStaff << '

for layer_i in range(len(layers_list)):
    bars = ' '.join(layers_list[layer_i])
    layer_string = f"{layer_names[layer_i]} = \drummode {{ {bars} }} "
    string_firstbit += layer_string
    staves_string = "\\new DrumVoice { \\%s \\%s }" % (voice_names[layer_i], layer_names[layer_i])
    string_secondbit += staves_string


string_secondbit += ' >>'

out_string = string_firstbit + string_secondbit
