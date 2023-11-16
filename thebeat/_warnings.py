# Copyright (C) 2022-2023  Jelle van der Werff
#
# This file is part of thebeat.
#
# thebeat is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# thebeat is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with thebeat.  If not, see <https://www.gnu.org/licenses/>.

framerounding_soundseq = (
    "thebeat: For one or more of the used sounds, the exact start or end positions in frames (i.e. "
    "samples) were rounded off to the neirest integer ceiling. This shouldn't be a problem. "
    "To get rid of this warning, try rounding off the onsets in the passed Sequence object "
    "by calling Sequence.round_onsets() before passing the object to the SoundSequence "
    "constructor."
)

framerounding_melody = (
    "thebeat: For one or more of the used sounds, the exact start or end positions in frames (i.e. "
    "samples) were rounded off to the neirest integer ceiling. This shouldn't be a problem."
    "To get rid of this warning, try using a sampling frequency of 48000 Hz, or a different"
    "beat_ms."
)

framerounding_soundsynthesis = (
    "thebeat: During sound synthesis, the number of frames was rounded off. This shouldn't be a problem. "
    "To get rid of this warning, try using a combination of sound duration (in seconds) and sampling "
    "frequency that results in integer values."
)

missing_values = "thebeat: There were missing values in the passed data."

normalization = "thebeat: Sound was normalized to avoid distortion. If undesirable, change the amplitude of the sounds."

phases_t_at_zero = (
    "thebeat: The first onset of the test sequence was at t=0.\nThis would result in a phase difference "
    "that is always 0, which is not very informative.\nTherefore, the first phase difference was discarded.\n"
    "If you want the first onset at a different time than zero, use the Sequence.from_onsets() method to "
    "create the Sequence object."
)

duplicate_names_sequence_plot = (
    "thebeat: Two or more sequences have the same name.\n"
    "Numbers were added to the y-axis labels to distinguish between them.\n"
    "The numbers were added to the names in the order in which the sequences were provided."
)
