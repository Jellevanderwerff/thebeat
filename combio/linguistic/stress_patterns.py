import numpy as np
from combio.core import Sequence, Stimulus, StimTrial


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


def generate_trimoraic_sequence(n_phrases: int,
                                n_iois_per_phrase: int,
                                mora_ioi: int = 250,
                                noise_sd: float = 0.0,
                                rng=None) -> Sequence:

    """
    Parameters
    ----------
    n_phrases
    n_iois_per_phrase
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
    print(ioi_types)
    n_iois = int(n_phrases * n_iois_per_phrase)

    iois = np.array([], dtype=np.float32)

    for n in range(n_phrases):
        i = 0

        while i < n_iois_per_phrase:
            if rng.choice([1, 2], 1) == 1 or i == (n_iois_per_phrase - 1):
                iois = np.append(iois, 3)
                i += 1
            else:
                ioi_one = rng.choice(ioi_types, 1)
                ioi_two = 3 - ioi_one
                iois = np.concatenate([iois, ioi_one, ioi_two])
                i += 2

    ioi_sums = np.cumsum(iois)  # from now on it's an array again
    pattern_shifted = ioi_sums[:-1] + start_of_pattern
    pattern = np.concatenate([[start_of_pattern], pattern_shifted])

    # Add Gaussian noise
    #noises = rng.normal(0, scale=noise_sd, size=n_iois - 1)
    #iois[1:] = iois[1:] + noises

    # Get iois from noisy pattern
    iois[:-1] = np.diff(pattern)
    iois[-1] = float(iois[-1] + rng.normal(0, scale=noise_sd, size=1))

    # Calculate with proper tempo
    iois = iois * mora_ioi

    return Sequence(iois)


seq = generate_trimoraic_sequence(10, 10)
stim = Stimulus.generate()
trial = StimTrial(stim, seq)

"""
    case 'mora'
        % construct pattern based on mora-timed languages. consists of
        % clusters that contain either one or two iois and have the same
        % total duration (3s).
        ioiTypes    = [0.25 0.5 0.75 1 1.25 1.5 1.75];
        nTypes      = length(ioiTypes);
        % loop over iois and create clusters of one or two iois
        iois        = [];
        i = 0;
        while (i < n)
            if (randi(2) == 1 || i == n-1)
                iois    = [iois 3];
                i       = i + 1;
            else
                ioiOne  = ioiTypes(randi(nTypes,1));
                ioiTwo  = 3 - ioiOne;
                iois    = [iois ioiOne ioiTwo];
                i       = i + 2;
            end
        end
        ioiSums = cumsum(iois);
        pattern = [startOfPattern ioiSums(1:end-1)+startOfPattern];
        % introduce gaussean noise
        pattern(2:end)  = pattern(2:end) + stdev * randn(1,n-1);
        % get iois from noisy pattern
        iois(1:end-1)   = pattern(2:end) - pattern(1:end-1);
        iois(end)       = iois(end) + stdev * randn(1,1);
end
"""