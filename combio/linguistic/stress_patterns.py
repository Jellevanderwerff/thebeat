import numpy as np
from combio.core import Sequence, Stimulus, StimSequence


def generate_stress_timed_sequence(n_phrases: int,
                                   n_iois_per_phrase: int,
                                   syllable_ioi: int = 500,
                                   noise_sd: float = 0.0,
                                   rng=None) -> Sequence:
    """Description here"""
    if rng is None:
        rng = np.random.default_rng()

    start_of_pattern = syllable_ioi

    ioi_types = (np.arange(start=1, stop=16) / 8) * syllable_ioi
    # number of iois
    n_iois = int(n_iois_per_phrase * n_phrases)

    iois = np.array([])

    c = 0

    while c < n_phrases:
        iois_pattern = rng.choice(ioi_types, n_iois_per_phrase - 1)
        # add a final ioi so that cluster duration = nIOI * ioiDur
        final_ioi = (n_iois_per_phrase * syllable_ioi) - np.sum(iois_pattern)
        # if there is not enough room for a final ioi, repeat (q'n'd..)
        if final_ioi >= 0.25:
            iois_pattern = np.concatenate([iois_pattern, [final_ioi]])
            iois = np.concatenate([iois, iois_pattern])
            c += 1
        else:
            continue

    ioi_sums = np.cumsum(iois)  # from now on it's an array again
    pattern_shifted = ioi_sums[:-1] + start_of_pattern
    pattern = np.concatenate([[start_of_pattern], pattern_shifted])

    # Add Gaussian noise
    noises = rng.normal(0, scale=noise_sd, size=n_iois - 1)
    pattern[1:] = pattern[1:] + noises

    # get iois from noisy pattern
    iois[:-1] = np.diff(pattern)
    iois[-1] = float(iois[-1] + rng.normal(0, scale=noise_sd, size=1))

    return Sequence(iois)


def generate_trimoraic_sequence(n_feet: int,
                                mora_ioi: int = 250,
                                noise_sd: float = 0.0,
                                rng=None) -> Sequence:

    """
    Parameters
    sdfg
    mora_ioi : This is the tempo in milliseconds, based on one mora. Sequences have three morae per cluster.
    noise_sd
    rng

    Returns
    -------

    """

    if rng is None:
        rng = np.random.default_rng()

    start_of_pattern = 2

    # built around clusters with either one or two iois with same total duration (3 * syllable_ioi)
    ioi_types = (np.arange(start=1, stop=8) / 4)

    iois = np.array([], dtype=np.float32)

    i = 0

    while i < n_feet:
        if rng.choice([1, 2], 1) == 1 or i == (n_feet - 1):
            iois = np.append(iois, 3)
        else:
            ioi_one = rng.choice(ioi_types, 1)
            ioi_two = 3 - ioi_one
            iois = np.concatenate([iois, ioi_one, ioi_two])

        i += 1

    ioi_sums = np.cumsum(iois)
    pattern_shifted = ioi_sums[:-1] + start_of_pattern
    pattern = np.concatenate([[start_of_pattern], pattern_shifted])

    # Add Gaussian noise
    noises = rng.normal(0, scale=noise_sd, size=len(iois) - 1)
    iois[1:] = iois[1:] + noises

    # Get iois from noisy pattern
    iois[:-1] = np.diff(pattern)
    iois[-1] = float(iois[-1] + rng.normal(0, scale=noise_sd, size=1))

    # Calculate with proper tempo
    iois = iois * mora_ioi

    return Sequence(iois)
