from stimulus import *
from mingus.extra import lilypond

rhythm = Rhythm.from_note_values([4] * 12)
stim = Stimulus.generate(freq=440, offramp=10)
stims = Stimuli.from_stim(stim, repeats=len(rhythm.onsets))
trial = RhythmTrial(rhythm, stims)
trial.add_layer(rhythm, stims)

layers_list = []
layers_dict = {0: 'bd', 1: 'snare'}

if trial.n_layers > 2:
    raise ValueError("Can only do two layers unfortunately.")

for layer in range(trial.n_layers):
    bars = []
    events = [event for event in trial.events if event.layer == layer]
    bar = ''
    b = Bar(meter=trial.time_sig)
    for event in events:
        note_value = int(event.note_value)
        b.place_rest(note_value)
        note = layers_dict[layer] + str(note_value) + ' '
        bar += note
        if b.current_beat == b.length:
            bars.append("{ " + bar + "}")
            b = Bar(meter=trial.time_sig)
            bar = ''

    layers_list.append(bars)


voice_names = ['voiceOne', 'voiceTwo']
layer_names = ['down', 'up']

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
print(out_string)