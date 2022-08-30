import numpy as np
from fractions import Fraction
from typing import Iterable


class BaseSequence:
    """Base Sequence class that holds only IOIs and calculates onset values. """

    def __init__(self,
                 iois: Iterable,
                 metrical: bool = False):

        self.iois = iois

        # Save metrical attribute
        self.metrical = metrical

    @property
    def iois(self):
        # Return a copy of self._iois
        return np.array(self._iois, dtype=np.float32).copy()

    @iois.setter
    def iois(self, values):

        # Make input
        iois = np.array(values, dtype=np.float32)

        if np.any(iois <= 0):
            raise ValueError("Inter-onset intervals (IOIs) cannot be zero or negative.")

        self._iois = iois

    @property
    def onsets(self):
        """Get the event onsets. This is the cumulative sum of Sequence.iois, with 0 additionally prepended.
        """

        if self.metrical is True:
            return np.cumsum(np.append(0, self.iois[:-1]), dtype=np.float32)
        else:
            return np.cumsum(np.append(0, self.iois), dtype=np.float32)

    @onsets.setter
    def onsets(self, values):

        # Onsets may be in the wrong order, so sort first
        np.sort(values)

        # Check whether first onset is 0 (requirement of this package)
        if values[0] != 0:
            raise ValueError("First onset is not 0")

        if np.any(values[:-1] == values[1:]):
            raise ValueError("Cannot have two onsets that occur simultaneously.")

        # Set the IOIs
        self.iois = np.diff(values)


class Sequence(BaseSequence):
    """
    Sequence class that holds a sequence of inter-onset intervals (IOIs) and stimulus onsets.
    Additionally has class methods that can be used for generating a new sequence.

    Attributes
    ----------

    iois : Numpy 1-D array
        A list of the sequence's inter-onset intervals.

    Class methods
    -------------

    generate_random_normal(n, mu, sigma, rng=None)
        Generate a random sequence using the normal distribution.
    generate_random_uniform(n, a, b, rng=None)
        Generate a random sequence using a uniform distribution.
    generate_random_poisson(n, lam, rng=None)
        Generate a random sequence using a Poisson distribution.
    generate_random_exponential(n, lam, rng=None)
        Generate a random sequence using an exponential distribution.
    generate_isochronous(n, ioi)
        Generate an isochronous sequence using an exponential distribution.

    Methods
    -------


    """

    def __init__(self, iois, metrical=False):

        # Call super init method
        BaseSequence.__init__(self, iois=iois, metrical=metrical)

    def __str__(self):
        if self.metrical:
            return f"Object of type Sequence (metrical version):\n{len(self.onsets)} events\nIOIs: {self.iois}\nOnsets: {self.onsets}\n"
        else:
            return f"Object of type Sequence (non-metrical version):\n{len(self.onsets)} events\nIOIs: {self.iois}\nOnsets: {self.onsets}\n"

    def __add__(self, other):
        return _join_sequences([self, other])

    def __len__(self):
        return len(self.onsets)

    @classmethod
    def generate_random_normal(cls, n: int, mu: int, sigma: int, rng=None, metrical=False):
        """

        Class method that generates a sequence of random inter-onset intervals based on the normal distribution.
        Note that there will be n-1 IOIs in a sequence.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        mu : int
            The mean of the normal distribution.
        sigma : int
            The standard deviation of the normal distribution.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)
        metrical : boolean
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
            grid)

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        if metrical:
            n_iois = n
        elif not metrical:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.normal(loc=mu, scale=sigma, size=n_iois))

        return cls(round_iois, metrical=metrical)

    @classmethod
    def generate_random_uniform(cls, n: int, a: int, b: int, rng=None, metrical=False):
        """

        Class method that generates a sequence of random inter-onset intervals based on a uniform distribution.
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        a : int
            The left bound of the uniform distribution.
        b : int
            The right bound of the normal distribution.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)
        metrical : boolean
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
            grid)

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        if metrical:
            n_iois = n
        elif not metrical:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.uniform(low=a, high=b, size=n_iois))

        return cls(round_iois, metrical=metrical)

    @classmethod
    def generate_random_poisson(cls, n: int, lam: int, rng=None, metrical=False):
        """

        Class method that generates a sequence of random inter-onset intervals based on a Poisson distribution.
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        lam : int
            The desired value for lambda.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)
        metrical : boolean
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
                grid)

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        if metrical:
            n_iois = n
        elif not metrical:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.poisson(lam=lam, size=n_iois))

        return cls(round_iois, metrical=metrical)

    @classmethod
    def generate_random_exponential(cls, n: int, lam: int, rng=None, metrical=False):
        """

        Class method that generates a sequence of random inter-onset intervals based on an exponential distribution.
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        lam : int
           The desired value for lambda.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)
        metrical : boolean
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
            grid)

        Returns
        -------
        Returns an object of class Sequence.

        """
        if rng is None:
            rng = np.random.default_rng()

        if metrical:
            n_iois = n
        elif not metrical:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.exponential(scale=lam, size=n_iois))

        return cls(round_iois, metrical=metrical)

    @classmethod
    def generate_isochronous(cls, n: int, ioi: int, metrical=False):
        """

        Class method that generates a sequence of isochronous inter-onset intervals.
        Note that there will be n-1 IOIs in a sequence. IOIs are rounded off to integers.


        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        ioi : int
            The inter-onset interval to be used between all events.
        metrical : boolean
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
            grid)

        Returns
        -------
        Returns an object of class Sequence.

        """

        if metrical:
            n_iois = n
        elif not metrical:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        return cls(np.round([ioi] * n_iois), metrical=metrical)

    @classmethod
    def from_integer_ratios(cls, numerators, value_of_one_in_ms, metrical=False):
        numerators = np.array(numerators)
        return cls(numerators * value_of_one_in_ms, metrical=metrical)

    @classmethod
    def from_onsets(cls, onsets, metrical=False):
        iois = np.diff(onsets)

        return cls(iois, metrical=metrical)

    # Manipulation methods
    def change_tempo(self, factor):
        """
        Change the tempo of the sequence.
        A factor of 1 or bigger increases the tempo (resulting in smaller IOIs).
        A factor between 0 and 1 decreases the tempo (resulting in larger IOIs).
        """
        if factor > 0:
            self.iois /= factor
        else:
            raise ValueError("Please provide a factor larger than 0.")

    def change_tempo_linearly(self, total_change):
        """
        This function can be used for creating a ritardando or accelerando effect.
        You provide the total change over the entire sequence.
        So, total change of 2 results in a final IOI that is
        twice as short as the first IOI.
        """
        self.iois /= np.linspace(1, total_change, self.iois.size)

    @property
    def duration_ms(self):
        return np.sum(self.iois)

    @property
    def duration_s(self):
        return np.sum(self.iois) / 1000

    @property
    def integer_ratios(self):
        """
        This function calculates how to describe a sequence of IOIs in integer ratio numerators from
        the total duration of the sequence, by finding the least common multiplier.

        Example:
        A sequence of IOIs [250, 500, 1000, 250] has a total duration of 2000 ms.
        This can be described using the least common multiplier as 1/8, 2/8, 4/8, 1/8,
        so this function returns the numerators [1, 2, 4, 1].

        For an example of this method being used, see:
        Jacoby, N., & McDermott, J. H. (2017). Integer Ratio Priors on Musical Rhythm
            Revealed Cross-culturally by Iterated Reproduction.
            Current Biology, 27(3), 359–370. https://doi.org/10.1016/j.cub.2016.12.031


        """

        fractions = [Fraction(int(ioi), int(self.duration_ms)) for ioi in self.iois]
        lcm = np.lcm.reduce([fr.denominator for fr in fractions])

        vals = [int(fr.numerator * lcm / fr.denominator) for fr in fractions]

        return vals

    @property
    def interval_ratios_from_dyads(self):
        """
        This function returns sequential interval ratios,
        calculated as ratio_k = ioi_k / (ioi_k + ioi_{k+1})

        Note that for n IOIs this function returns n-1 ratios.

        It follows the methodology from:
        Roeske, T. C., Tchernichovski, O., Poeppel, D., & Jacoby, N. (2020).
            Categorical Rhythms Are Shared between Songbirds and Humans. Current Biology, 30(18),
            3544-3555.e6. https://doi.org/10.1016/j.cub.2020.06.072

        """

        return np.array([self.iois[k] / (self.iois[k] + self.iois[k + 1]) for k in range(len(self.iois) - 1)])


def _join_sequences(iterator):
    """
    This function can join metrical Sequence objects.
    """

    # Check whether iterable was passed
    if not hasattr(iterator, '__iter__'):
        raise ValueError("Please pass this function a list or other iterable object.")

    # Check whether all the objects are of the same type
    if not all(isinstance(x, Sequence) for x in iterator):
        raise ValueError("This function can only join multiple Sequence objects.")

    # Sequence objects need to be metrical:
    if not all(x.metrical for x in iterator):
        raise ValueError("Only metrical Sequence objects can be joined. This is intentional.")

    iois = [sequence.iois for sequence in iterator]
    iois = np.concatenate(iois)

    return Sequence(iois, metrical=True)
