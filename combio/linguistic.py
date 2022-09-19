import numpy as np
from combio.core.sequence import Sequence


def generate_moraic_sequence(n_feet: int,
                             foot_ioi: int = 500,
                             rng=None) -> Sequence:
    """
    This function generates a Sequence object with inter-onset intervals (IOIs) mimicing the rhythmic
    structure of mora-timed languages. The feet contain clusters of either one or two IOIs with the same total duration.
    The total duration is specified in foot_ioi.

    The phrases all have the same duration, but are made up of different combinations of IOIs.

    Parameters
    ----------
    foot_ioi
    n_feet : int
        The number of feet in the sequence.
    foot_ioi : int, optional
        The duration in milliseconds of the foot.
    rng : numpy.random.Generator, optional
        A NumPy Generator object, if none is supplied will default to numpy.random.default_rng()

    Returns
    -------
    Sequence
        A newly created Sequence object.

    Examples
    --------
    >>> rng = np.random.default_rng(seed=123)
    >>> seq = generate_moraic_sequence(n_feet=3, foot_ioi=500, rng=rng)
    >>> print(seq.iois)
    [500.      208.33333 291.66666 500.     ]


    Notes
    -----
    Both the methodology and the code are based on/taken from [1]_.

    References
    ----------
    .. [1] Ravignani, A., & Norton, P. (2017). Measuring rhythmic complexity:
       a primer to quantify and compare temporal structure in speech, movement, and animal vocalizations.
       Journal of Language Evolution, 2(1), 4-19.
       https://doi.org/10.1093/jole/lzx002


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

    # Get iois from pattern
    iois[:-1] = np.diff(pattern)

    # Calculate with proper tempo
    iois = iois * (foot_ioi / 3)

    return Sequence(iois)


def generate_stress_timed_sequence(n_events_per_phrase: int,
                                   syllable_ioi: int = 500,
                                   n_phrases: int = 1,
                                   rng=None) -> Sequence:
    """
    This function generates a Sequence object with inter-onset intervals (IOIs) mimicing the rhythmic
    structure of stress-timed languages. One can provide the length of a 'phrase' (or 'cluster') as n_events_per_phrase.
    In one sequence (e.g. 'sentence'), one would have n_phrases of n_events_per_phrase.

    The phrases all have the same duration, but are made up of different combinations of IOIs.

    Parameters
    ----------
    n_events_per_phrase : int
        The number of events in a single 'phrase'.
    syllable_ioi : int, optional
        The duration of the reference IOI in milliseconds. The default is 500.
    n_phrases : int, optional
        The number of phrases in the sequence. The default is 1.
    rng : numpy.random.Generator, optional
        A NumPy Generator object, if none is supplied will default to numpy.random.default_rng()

    Returns
    -------
    Sequence
        A newly created Sequence object.

    Examples
    --------
    >>> rng = np.random.default_rng(seed=123)
    >>> seq = generate_stress_timed_sequence(n_events_per_phrase=4, n_phrases=3, rng=rng)
    >>> print(seq.iois)
    [ 62.5 687.5 562.5 687.5  62.5 875.  250.  812.5 250.  187.5 375. ]


    Notes
    -----
    Both the methodology and the code are based on/taken from [2]_.

    References
    ----------
    .. [2] Ravignani, A., & Norton, P. (2017). Measuring rhythmic complexity:
       a primer to quantify and compare temporal structure in speech, movement, and animal vocalizations.
       Journal of Language Evolution, 2(1), 4-19.
       https://doi.org/10.1093/jole/lzx002


    """
    if rng is None:
        rng = np.random.default_rng()

    start_of_pattern = syllable_ioi

    ioi_types = (np.arange(start=1, stop=16) / 8) * syllable_ioi

    iois = np.array([])

    c = 0

    while c < n_phrases:
        iois_pattern = rng.choice(ioi_types, n_events_per_phrase - 1)
        # add a final ioi so that cluster duration = nIOI * ioiDur
        final_ioi = (n_events_per_phrase * syllable_ioi) - np.sum(iois_pattern)
        # if there is not enough room for a final ioi, repeat (q'n'd..)
        if final_ioi >= 0.25:
            iois_pattern = np.concatenate([iois_pattern, [final_ioi]])
            iois = np.concatenate([iois, iois_pattern])
            c += 1
        else:
            continue

    ioi_sums = np.cumsum(iois)
    pattern_shifted = ioi_sums[:-1] + start_of_pattern
    pattern = np.concatenate([[start_of_pattern], pattern_shifted])

    return Sequence.from_onsets(pattern)

