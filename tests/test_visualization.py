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

import numpy as np
import pytest

import thebeat.visualization
from thebeat.core import Sequence, SoundSequence, SoundStimulus
from thebeat.visualization import plot_multiple_sequences, recurrence_plot


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.custom_mpl_image_compare
def test_plot_multiple_sequences_0(rng):
    trials = []

    for x in range(10):
        seq = Sequence.generate_random_uniform(
            n_events=10, a=400, b=600, rng=rng
        )  # = 10 stimuli, 9 IOIs
        seq.round_onsets()
        stims = [SoundStimulus.generate() for _ in range(10)]  # = 10 stimuli
        trials.append(SoundSequence(stims, seq))

    fig, ax = plot_multiple_sequences(trials, style="ggplot", suppress_display=True)

    return fig


@pytest.custom_mpl_image_compare
def test_plot_multiple_sequences_1(rng):
    seqs = [Sequence.generate_random_normal(10, mu=500, sigma=25, rng=rng) for _ in range(10)]
    plot_multiple_sequences(seqs, suppress_display=True)

    seq1 = Sequence.generate_random_normal(
        n_events=5, mu=500, sigma=25, end_with_interval=True, rng=rng
    )
    seq2 = Sequence.generate_random_normal(
        n_events=5, mu=500, sigma=25, end_with_interval=True, rng=rng
    )
    fig, ax = plot_multiple_sequences(
        [seq1, seq2], figsize=(10, 5), suppress_display=True, colors=["red", "blue"]
    )

    return fig


@pytest.custom_mpl_image_compare
def test_plot_multiple_sequences_2(rng):
    seq1 = Sequence.generate_random_normal(
        n_events=5, mu=500, sigma=25, end_with_interval=True, rng=rng
    )
    seq2 = Sequence.generate_random_normal(
        n_events=5, mu=500, sigma=25, end_with_interval=True, rng=rng
    )
    fig, ax = plot_multiple_sequences(
        [seq1, seq2], figsize=(10, 5), suppress_display=True, colors=[(1, 0, 0), (0, 0, 1)]
    )
    return fig


@pytest.custom_mpl_image_compare
def test_plot_multiple_sequences_3(rng):
    seq1 = Sequence.generate_random_normal(n_events=5, mu=500, sigma=25, rng=rng)
    seq1.round_onsets()
    seq2 = Sequence.generate_random_normal(n_events=5, mu=500, sigma=25, rng=rng)
    seq2.round_onsets()
    sound_seq = SoundSequence(SoundStimulus.generate(), seq2)
    fig, ax = plot_multiple_sequences(
        [seq1, sound_seq], figsize=(10, 5), suppress_display=True, colors=[(1, 0, 0), (0, 0, 1)]
    )
    return fig


@pytest.custom_mpl_image_compare
def test_recurrence_plot_threshold(rng):
    seq = (
        Sequence.generate_random_normal(
            n_events=10, mu=500, sigma=20, end_with_interval=True, rng=rng
        )
        * 5
    )
    fig, ax = recurrence_plot(seq, 0.03, suppress_display=True)

    return fig


@pytest.custom_mpl_image_compare
def test_recurrence_plot_nothreshold(rng):
    seq = (
        Sequence.generate_random_normal(
            n_events=10, mu=500, sigma=20, end_with_interval=True, rng=rng
        )
        * 5
    )
    fig, ax = recurrence_plot(seq, suppress_display=True)

    return fig


@pytest.custom_mpl_image_compare
def test_plot_phase_differences(rng):
    test_seqs = [
        Sequence.generate_random_normal(n_events=10, mu=500, sigma=100, rng=rng) for _ in range(100)
    ]
    ref_seq = Sequence.generate_random_normal(n_events=10, mu=500, sigma=100, rng=rng)

    fig, ax = thebeat.visualization.plot_phase_differences(
        test_seqs, ref_seq, suppress_display=True, title="My first phase difference plot"
    )

    return fig


@pytest.custom_mpl_image_compare
def test_interval_ratios_plot_density(rng):
    seqs = [
        Sequence.generate_random_normal(n_events=10, mu=500, sigma=100, rng=rng) for _ in range(100)
    ]

    fig, ax = thebeat.visualization.plot_interval_ratios_density(
        seqs, suppress_display=True, title="My first density plot", resolution=0.1
    )
    return fig


@pytest.custom_mpl_image_compare
def test_interval_ratios_plot_histogram(rng):
    seqs = [
        Sequence.generate_random_normal(n_events=10, mu=500, sigma=100, rng=rng) for _ in range(100)
    ]

    fig, ax = thebeat.visualization.plot_interval_ratios_histogram(
        seqs, suppress_display=True, title="My first density plot"
    )
    return fig
