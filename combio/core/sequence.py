import numpy as np
from fractions import Fraction
from typing import Iterable, Union


class BaseSequence:
    """
    This is the most basic of classes that the Sequence class inherits from, as well as the Rhythm class.
    It cannot do many things, apart from holding a number of inter-onset intervals (IOIs).

    The BaseSequence class dictates that a sequence can either be metrical or not.
    The default is non-metrical, meaning that if there are n onset values (i.e. t values),
    there are n-1 IOIs. This is what people will need in most cases.
    Metrical sequences have an additional final IOI (so they end with a gap of silence).
    This is what you will need in cases with e.g. rhythmical/musical sequences.

    The BaseSequence class protects against impossible values for the IOIs, as well as for the
    event onsets (t values).

    Finally, remember that the first event onset is always at t = 0!

    Attributes
    ----------
    iois : NumPy 1-D array
        Contains the inter-onset intervals (IOIs). This is the bread and butter of the BaseSequence class.
        Non-metrical sequences have n IOIs and n+1 onsets. Metrical sequences have an equal number of IOIs
        and onsets.
    onsets : NumPy 1-D array
        Property that contains the onsets (t values) in milliseconds. The first onset is additionally added and
        is always zero.

    Parameters
    ----------
    iois : iterable
        An iterable of inter-onset intervals (IOIs). For instance: [500, 500, 400, 200]
    metrical : bool, optional
        Indicates whether sequence has an extra final inter-onset interval; this is useful for musical/rhythmical
        sequences.

    """

    def __init__(self,
                 iois: Iterable,
                 metrical: bool = False):

        self.iois = iois

        # Save metrical attribute
        self.metrical = metrical

    @property
    def iois(self):
        """IOI getter. Returns a copy of the IOIs attribute object, to protect against
        misuse."""
        # Return a copy of self._iois
        return np.array(self._iois, dtype=np.float32).copy()

    @iois.setter
    def iois(self, values):
        """IOI setter. Checks against negative IOIs."""

        # We always want a NumPy array:
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
        """Set the event onsets. First onset must be zero, and there cannot be two onsets that occur simultaneously.
        """

        # Onsets may be in the wrong order, so sort first
        np.sort(values)

        # Check whether first onset is 0 (requirement of this package)
        if values[0] != 0:
            raise ValueError("First onset is not 0")

        if np.any(values[:-1] == values[1:]):
            raise ValueError("Cannot have two onsets that occur simultaneously.")

        # Set the IOIs
        if self.metrical is True:
            raise ValueError("Cannot change onsets of metrical sequences. This is because we need to know the final "
                             "IOI for metrical sequences. Either reconstruct the sequence, or change the IOIs.")

        self.iois = np.diff(values)


class Sequence(BaseSequence):
    """
    The Sequence class is the most important class in this package. It is used as the basis
    for many functions, and can be passed to many functions.
    Sequences rely most importantly on inter-onset intervals (IOIs; the times between the onset of an event,
    and the onset of the next event). IOIs are also what we use to construct sequences
    (rather than event onsets, or t values).

    The most basic way of constructing a Sequence object is by passing it a list (or other iterable) of IOIs.
    However, the different class methods (e.g. Sequence.generate_isochronous()) may also be used.

    This class additionally contains functions and attributes to, for instance, get the event onsets values, to
    change the tempo, or to add Gaussian noise.


    Attributes
    ----------

    iois : NumPy 1-D array
        Contains the inter-onset intervals (IOIs). This is the bread and butter of the Sequence class.
        Non-metrical sequences have n IOIs and n+1 onsets. Metrical sequences have an equal number of IOIs
        and onsets.

    Examples
    --------
    >>> iois = [500, 400, 600]
    >>> seq = Sequence(iois)
    >>> print(seq.onsets)
    [   0.  500.  900. 1500.]

    >>> seq = Sequence.generate_isochronous(n=10, ioi=500)
    >>> print(len(seq.iois))
    9
    >>> print(len(seq.onsets))
    10
    """

    def __init__(self, iois: Iterable, metrical: bool = False):
        """Initialization of Sequence class."""

        # Call super init method
        BaseSequence.__init__(self, iois=iois, metrical=metrical)

    def __str__(self):
        if self.metrical:
            return f"Object of type Sequence (metrical)\n{len(self.onsets)} events\nIOIs: {self.iois}\nOnsets: {self.onsets}\n"
        else:
            return f"Object of type Sequence (non-metrical)\n{len(self.onsets)} events\nIOIs: {self.iois}\nOnsets: {self.onsets}\n"

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
            The right bound of the uniform distribution.
        rng : numpy.random.Generator, optional
            A Generator object, e.g. np.default_rng(seed=12345)
        metrical : boolean, optional
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
        metrical : bool, optional
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
        metrical : bool
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
    def from_onsets(cls, onsets):
        iois = np.diff(onsets)

        return cls(iois, metrical=False)

    # Manipulation methods

    def add_noise_gaussian(self, noise_sd: Union[int, float], rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.iois = self.iois + rng.normal(loc=0, scale=noise_sd, size=self.iois.size)

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
    def duration_ms(self) -> np.float32:
        return np.float32(np.sum(self.iois))

    @property
    def duration_s(self) -> np.float32:
        return np.float32(np.sum(self.iois) / 1000)

    @property
    def integer_ratios(self):
        """
        This function calculates how to describe a sequence of IOIs in integer ratio numerators from
        the total duration of the sequence, by finding the least common multiplier.

        Example:
        A sequence of IOIs [250, 500, 1000, 250] has a total duration of 2000 ms.
        This can be described using the least common multiplier as 1/8, 2/8, 4/8, 1/8,
        so this function returns the numerators [1, 2, 4, 1].

        References
        ----------
        For an example of this method being used, see:

        Jacoby, N., & McDermott, J. H. (2017). Integer Ratio Priors on Musical Rhythm
        Revealed Cross-culturally by Iterated Reproduction.
        Current Biology, 27(3), 359–370. https://doi.org/10.1016/j.cub.2016.12.031


        """

        fractions = [Fraction(int(ioi), int(self.duration_ms)) for ioi in self.iois]
        lcm = np.lcm.reduce([fr.denominator for fr in fractions])

        vals = [int(fr.numerator * lcm / fr.denominator) for fr in fractions]

        return np.array(vals)

    @property
    def interval_ratios_from_dyads(self):
        """
        This function returns sequential interval ratios,
        calculated as ratio_k = ioi_k / (ioi_k + ioi_{k+1})

        Note that for n IOIs this function returns n-1 ratios.

        References
        ----------

        The method follows the methodology from:

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
