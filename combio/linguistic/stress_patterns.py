import numpy as np
from combio.core import Sequence, Stimulus, StimTrial


def generate_stress_timed_sequence(n, repetitions, ioi=2, sd=10, rng=None):
    # todo change ioi to ms
    """Description here"""
    if rng is None:
        rng = np.random.default_rng()

    if not n % repetitions == 0:
        raise ValueError("Number of events ('n') must be divisible by number of repetitions of rhythmic pattern.")

    start_of_pattern = ioi

    # generate list of ioi types (0.25,0.5,0.75,1,...)
    ioi_types = list(np.arange(start=1, stop=16) / 4)
    n_types = len(ioi_types)
    # number of iois in each cluster
    n_iois = int(n / repetitions)

    iois = []
    c = 0

    while c < repetitions:
        iois_pattern = []

        # create nIOI-1 semi-random iois
        for i in range(n_iois):
            chosen_ioi = int(rng.choice(n_types, 1))
            iois_pattern.append(ioi_types[chosen_ioi])

        # add a final ioi so that cluster duration = nIOI * ioiDur
        final_ioi = (n_iois * 2) - sum(iois_pattern)

        # if there is not enough room for a final ioi, repeat (q'n'd..)
        if final_ioi >= 0.25:
            iois_pattern = list(iois_pattern + final_ioi)
            iois = iois + iois_pattern
            c += 1

    ioi_sums = np.cumsum(iois)  # from now on it's an array again
    pattern_shifted = ioi_sums[:-1] + start_of_pattern
    pattern = np.concatenate([[start_of_pattern], pattern_shifted], axis=0)

    noises = rng.normal(0, scale=sd, size=n-1)
    pattern[1:] = pattern[1:] + noises

    # get iois from noisy pattern
    iois[:-1] = np.diff(pattern)
    iois[-1] = float(iois[-1] + rng.normal(0, scale=sd, size=1))

    iois = np.array(iois) * 1000
    print(iois)

    return Sequence(iois)


seq = generate_stress_timed_sequence(100, 10, 0.2, 0.01)

#todo Alright, this is now what the script did, now only improve: e.g. flexible ioi durations,
# make everything into milliseconds etc.



"""
case 'stress'
        % construct pattern based on stress-timed languages. consists of
        % nRep clusters of IOIs that have the same total duration, but
        % different sets of within-cluster IOIs.
        if (mod(n,nRep))
            disp(['Error: number of events (' n ') must be divisible' ...
                  'by number of repetitions of the rhythmic pattern (' ...
                  nRep ') without remainder.']);
            return;
        end
        % generate list of ioi types (0.25,0.5,0.75,1,...)
        ioiTypes    = unique((1:15)/4);
        nTypes      = length(ioiTypes);
        % number of iois in each cluster
        nIOI    = n / nRep;
        % loop over patterns
        iois    = [];
        c = 0;
        while (c < nRep)
            % create nIOI-1 semi-random iois
            ioisPat = [];
            for i = 1:nIOI-1
                ioisPat     = [ioisPat ioiTypes(randi(nTypes,1))];
            end
            % add a final ioi so that cluster duration = nIOI * ioiDur
            finalIOI    = nIOI*2 - sum(ioisPat);
            % if there is not enough room for a final ioi, repeat (q'n'd..)
            if (finalIOI >= 0.25)
                ioisPat     = [ioisPat finalIOI];
                iois        = [iois ioisPat];
                c           = c + 1;
            end
        end
        ioiSums = cumsum(iois);
        pattern = [startOfPattern ioiSums(1:end-1)+startOfPattern];
        % introduce gaussean noise
        pattern(2:end)  = pattern(2:end) + stdev * randn(1,n-1);
        % get iois from noisy pattern
        iois(1:end-1)   = pattern(2:end) - pattern(1:end-1);
        iois(end)       = iois(end) + stdev * randn(1,1);
"""