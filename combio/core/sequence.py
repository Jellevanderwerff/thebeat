from __future__ import annotations  # this is to make sure we can type hint the return value in a class method
from fractions import Fraction
from typing import Iterable, Union
import numpy as np


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

        # Save attributes
        self.iois = iois
        self.metrical = metrical

    @property
    def iois(self) -> np.ndarray:
        """The inter-onset intervals (IOIs) of the Sequence object. These are the intervals in milliseconds
        between the onset of an event, and the onset of the next event. This is the most important
        attribute of the Sequence class and is used throughout.

        This getter returns a copy of the IOIs instead of
        the actual variable.
        """

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
    def onsets(self) -> np.ndarray:
        """ Returns the event onsets (t values) in milliseconds on the basis of the sequence objects'
        inter-onset intervals (IOIs). An additional first onset (t=0) is additionally prepended.
        """

        if self.metrical is True:
            return np.cumsum(np.append(0, self.iois[:-1]), dtype=np.float32)
        else:
            return np.cumsum(np.append(0, self.iois), dtype=np.float32)

    @onsets.setter
    def onsets(self, values):
        """Setter for the event onsets. First onset must be zero, and there cannot be two onsets that occur
        simultaneously.
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
    for many functions as it contains timing information in the form of inter-onset intervals (IOIs; the times between
    the onset of an event, and the onset of the next event) and event onsets (i.e. `t` values).
    IOIs are what we use to construct `Sequence` objects.

    The most basic way of constructing a `Sequence` object is by passing it a list (or other iterable) of IOIs.
    However, the different class methods (e.g. `Sequence.generate_isochronous()`) may also be used.

    This class additionally contains methods and attributes to, for instance, get the event onset values, to
    change the tempo, add Gaussian noise, or to plot the `Sequence` object using matplotlib.


    Attributes
    ----------

    iois : NumPy 1-D array
        Contains the inter-onset intervals (IOIs). This is the bread and butter of the Sequence class.
        Non-metrical sequences have n IOIs and n+1 onsets. Metrical sequences have an equal number of IOIs
        and onsets.
    name : str
        If desired, one can give a Sequence object a name. This is for instance used when printing the sequence,
        or when plotting the sequence. It can always be retrieved and changed via this attribute `Sequence.name`.

    Examples
    --------
    >>> iois = [500, 400, 600]
    >>> seq = Sequence(iois)
    >>> print(seq.onsets)
    [   0.  500.  900. 1500.]

    >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
    >>> print(len(seq.iois))
    4
    >>> print(len(seq.onsets))
    5
    """

    def __init__(self,
                 iois: Iterable,
                 metrical: bool = False,
                 name: str = None):
        """Initialization of Sequence class on the basis of inter-onset intervals (IOIs).
        When metrical is 'True', the sequence contains an equal number of IOIs and event onsets.
        If 'False' (the default), the sequence contains n event onsets, and n-1 IOIs."""

        # Call super init method
        BaseSequence.__init__(self, iois=iois, metrical=metrical)

        # Additionally save the provided name (may be None)
        self.name = name

    def __str__(self):

        if self.metrical:
            return f"Object of type Sequence (metrical)\nSequence name: {self.name}\n{len(self.onsets)} events\nIOIs: {self.iois}\nOnsets: {self.onsets}\n"
        else:
            return f"Object of type Sequence (non-metrical)\nSequence name: {self.name}\n{len(self.onsets)} events\nIOIs: {self.iois}\nOnsets: {self.onsets}\n"

    def __add__(self, other):
        return _join_sequences([self, other])

    def __len__(self):
        return len(self.onsets)

    @classmethod
    def from_integer_ratios(cls,
                            numerators: Iterable,
                            value_of_one_in_ms: int,
                            metrical: bool = False,
                            name: str = None) -> Sequence:
        """

        This class method can be used to construct a new Sequence object on the basis of 'integer ratios'.

        Parameters
        ----------
        numerators : iterable
            Contains the numerators of the integer ratios. For instance: [1, 2, 4]
        value_of_one_in_ms : int
            This represents the duration of the '1' numerator in milliseconds. If the numerators do not contain a
            1, multiples of this value are used. For instance, a sequence of [2, 4] with value_of_one_in_ms=500
            would be a Sequence with IOIs: [1000 2000].
        metrical : bool, optional
            Indicates whether a metrical or non-metrical sequence should be generated (see documentation for Sequence).
            Defaults to 'False'.
        name : str, optional
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute (`Sequence.name`).

        Returns
        -------
        Sequence
            A newly constructed Sequence object.

        Examples
        --------
        >>> seq = Sequence.from_integer_ratios(numerators=[1, 2, 4], value_of_one_in_ms=500)
        >>> print(seq.iois)
        [ 500. 1000. 2000.]
        """

        numerators = np.array(numerators)
        return cls(numerators * value_of_one_in_ms, metrical=metrical, name=name)

    @classmethod
    def from_onsets(cls, onsets: Iterable,
                    name: str = None) -> Sequence:
        """
        Class method that can be used to generate a new Sequence object on the basis of event onsets.

        Parameters
        ----------
        onsets: iterable
            An iterable of event onsets which must start from 0, e.g.: [0, 500, 1000]
        name : str, optional
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute (`Sequence.name`).

        Returns
        -------
        Sequence
            A newly constructed Sequence object.

        Examples
        --------
        >>> seq = Sequence.from_onsets([0, 500, 1000])
        >>> print(seq.iois)
        [500. 500.]
        """
        iois = np.diff(onsets)

        return cls(iois, metrical=False, name=name)

    @classmethod
    def generate_random_normal(cls,
                               n: int,
                               mu: int,
                               sigma: int,
                               rng=None,
                               metrical=False,
                               name: str = None) -> Sequence:
        """
        Class method that generates a Sequence object with random inter-onset intervals (IOIs) based on the normal
        distribution.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        mu : int
            The mean of the normal distribution.
        sigma : int
            The standard deviation of the normal distribution.
        rng : numpy.random.Generator, optional
            A Generator object. If not supplied NumPy's random.default_rng() is used.
        metrical : bool, optional
            Indicates whether a metrical or non-metrical sequence should be generated (see documentation for Sequence).
            Defaults to 'False'.
        name : str, optional
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute (`Sequence.name`).

        Returns
        -------
        Sequence
            A newly created Sequence object.

        Examples
        --------
        >>> generator = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_random_normal(n=5, mu=500, sigma=50, rng=generator)
        >>> print(seq.iois)
        [451. 482. 564. 510.]

        >>> seq = Sequence.generate_random_normal(n=5, mu=500, sigma=50, metrical=True)
        >>> len(seq.onsets) == len(seq.iois)
        True
        """
        if rng is None:
            rng = np.random.default_rng()

        if metrical is True:
            n_iois = n
        elif metrical is False:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.normal(loc=mu, scale=sigma, size=n_iois))

        return cls(round_iois, metrical=metrical, name=name)

    @classmethod
    def generate_random_uniform(cls,
                                n: int,
                                a: int,
                                b: int,
                                rng=None,
                                metrical: bool = False,
                                name: str = None) -> Sequence:
        """
        Class method that generates a sequence of random inter-onset intervals based on a uniform distribution.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        a : int
            The left bound of the uniform distribution.
        b : int
            The right bound of the uniform distribution.
        rng : numpy.random.Generator, optional
            A Generator object. If not supplied NumPy's random.default_rng() is used.
        metrical : bool, optional
            Indicates whether a metrical or non-metrical sequence should be generated (see documentation for Sequence).
            Defaults to 'False'.
        name : str, optional
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute (`Sequence.name`).

        Returns
        -------
        Sequence
            A newly created Sequence object.

        Examples
        --------
        >>> generator = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_random_uniform(n=5, a=400, b=600, rng=generator)
        >>> print(seq.iois)
        [536. 411. 444. 437.]

        >>> seq = Sequence.generate_random_uniform(n=5, a=400, b=600, metrical=True)
        >>> len(seq.onsets) == len(seq.iois)
        True
        """

        if rng is None:
            rng = np.random.default_rng()

        if metrical is True:
            n_iois = n
        elif metrical is False:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.uniform(low=a, high=b, size=n_iois))

        return cls(round_iois, metrical=metrical)

    @classmethod
    def generate_random_poisson(cls,
                                n: int,
                                lam: int,
                                rng=None,
                                metrical: bool = False,
                                name: str = None) -> Sequence:

        """
        Class method that generates a sequence of random inter-onset intervals based on a Poisson distribution.

        Parameters
        ----------
        n : int
            The desired number of events in the sequence.
        lam : int
            The desired value for lambda.
        rng : numpy.random.Generator, optional
            A Generator object, if none is supplied NumPy's random.default_rng() is used.s
        metrical : bool, optional
            Indicates whether there's an additional final IOI (for use in rhythmic sequences that adhere to a metrical
            grid)
        name : str, optional
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute (`Sequence.name`).

        Returns
        -------
        Sequence
            A newly created Sequence object

        Examples
        --------
        >>> generator = np.random.default_rng(123)
        >>> seq = Sequence.generate_random_poisson(n=5, lam=500, rng=generator)
        >>> print(seq.iois)
        [512. 480. 476. 539.]

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

        return cls(round_iois, metrical=metrical, name=name)

    @classmethod
    def generate_random_exponential(cls,
                                    n: int,
                                    lam: int,
                                    rng=None,
                                    metrical: bool = False,
                                    name: str = None) -> Sequence:
        """

        Class method that generates a sequence of random inter-onset intervals based on an exponential distribution.

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
        name : str, optional
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute (`Sequence.name`).

        Returns
        -------
        Returns an object of class Sequence.

        Examples
        --------
        >>> generator = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_random_exponential(n=5, lam=500, rng=generator)
        >>> print(seq.iois)
        [298.  59. 126. 154.]

        """
        if rng is None:
            rng = np.random.default_rng()

        if metrical is True:
            n_iois = n
        elif metrical is False:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        round_iois = np.round(rng.exponential(scale=lam, size=n_iois))

        return cls(round_iois, metrical=metrical)

    @classmethod
    def generate_isochronous(cls,
                             n: int,
                             ioi: int,
                             metrical=False,
                             name: str = None) -> Sequence:
        """
        Class method that generates a sequence of isochronous (i.e. equidistant) inter-onset intervals.
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
        name : str, optional
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute (`Sequence.name`).

        Returns
        -------
        Sequence
            A newly created Sequence object.

        Examples
        --------
        >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
        >>> print(seq.iois)
        [500. 500. 500. 500.]
        >>> print(len(seq.onsets))
        5
        >>> print(len(seq.iois))
        4

        >>> seq = Sequence.generate_isochronous(n=5, ioi=500, metrical=True)
        >>> print(len(seq.onsets))
        5
        >>> print(len(seq.iois))
        5

        """

        if metrical is True:
            n_iois = n
        elif metrical is False:
            n_iois = n - 1
        else:
            raise ValueError("Illegal value passed to 'metrical' argument. Can only be True or False.")

        return cls(np.round([ioi] * n_iois), metrical=metrical, name=name)

    # Manipulation methods
    def add_noise_gaussian(self,
                           noise_sd: Union[int, float],
                           rng=None) -> None:
        """
        This function can be used to add some Gaussian noise to the inter-onset intervals (IOIs)
        of the Sequence object. It uses a normal distribution with mean 0, and a standard deviation
        of 'noise_sd'.

        Parameters
        ----------
        noise_sd : int or float
            The standard deviation of the normal distribution used for adding in noise.
        rng : numpy.random.Generator, optional
            A Numpy Generator object. If none is supplied, Numpy's random.default_rng() is used.

        Examples
        --------
        >>> gen = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
        >>> print(seq.iois)
        [500. 500. 500. 500.]
        >>> seq.add_noise_gaussian(noise_sd=50, rng=gen)
        >>> print(seq.iois)
        [450.54395 481.61066 564.39624 509.69873]
        >>> print(seq.onsets)
        [   0.       450.54395  932.1546  1496.5508  2006.2495 ]
        """
        if rng is None:
            rng = np.random.default_rng()
        self.iois = self.iois + rng.normal(loc=0, scale=noise_sd, size=self.iois.size)

    def change_tempo(self,
                     factor: Union[int, float]) -> None:
        """
        Change the tempo of the Sequence object, where a factor of 1 or bigger increases the tempo (but results in
        smaller inter-onset intervals). A factor between 0 and 1 decreases the tempo (but results in larger
        inter-onset intervals).

        Parameters
        ----------
        factor : int or float
            Tempo change factor. E.g. '2' means twice as fast. 0.5 means twice as slow.

        Examples
        --------
        >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
        >>> print(seq.onsets)
        [   0.  500. 1000. 1500. 2000.]
        >>> seq.change_tempo(2)
        >>> print(seq.onsets)
        [   0.  250.  500.  750. 1000.]
        """

        if factor > 0:
            self.iois /= factor
        else:
            raise ValueError("Please provide a factor larger than 0.")

    def change_tempo_linearly(self,
                              total_change: Union[int, float]) -> None:
        """
        This function can be used for creating a ritardando or accelerando effect in the inter-onset intervals (IOIs).
        It divides the IOIs by a vector linearly spaced between 1 and total_change.

        Parameters
        ----------
        total_change : int or float
            Total tempo change at the end of the Sequence compared to the beginning.
            So, a total change of 2 (accelerando) results in a final IOI that is twice as short as the first IOI.
            A total change of 0.5 (ritardando) results in a final IOI that is twice as long as the first IOI.

        Examples
        --------
        >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
        >>> print(seq.iois)
        [500. 500. 500. 500.]
        >>> seq.change_tempo_linearly(total_change=2)
        >>> print(seq.iois)
        [500. 375. 300. 250.]

        """
        self.iois /= np.linspace(1, total_change, self.iois.size)

    @property
    def duration_ms(self) -> np.float32:
        """Get the total duration of the Sequence object in milliseconds.
        """
        return np.float32(np.sum(self.iois))

    @property
    def duration_s(self) -> np.float32:
        """Get the total duration of the Sequence object in seconds.
        """
        return np.float32(np.sum(self.iois) / 1000)

    @property
    def integer_ratios(self):
        """
        This property calculates how to describe a sequence of IOIs in integer ratio numerators from
        the total duration of the sequence by finding the least common multiplier.

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

        Examples
        --------
        >>> seq = Sequence([250, 500, 1000, 250])
        >>> print(seq.integer_ratios)
        [1 2 4 1]

        """
        fractions = [Fraction(int(ioi), int(self.duration_ms)) for ioi in self.iois]
        lcm = np.lcm.reduce([fr.denominator for fr in fractions])

        vals = [int(fr.numerator * lcm / fr.denominator) for fr in fractions]

        return np.array(vals)

    @property
    def interval_ratios_from_dyads(self):
        """
        This property returns sequential interval ratios,
        calculated as ratio_k = ioi_k / (ioi_k + ioi_{k+1})

        Note that for n IOIs this function returns n-1 ratios.

        References
        ----------

        The method follows the methodology from:

        Roeske, T. C., Tchernichovski, O., Poeppel, D., & Jacoby, N. (2020).
        Categorical Rhythms Are Shared between Songbirds and Humans. Current Biology, 30(18),
        3544-3555.e6. https://doi.org/10.1016/j.cub.2020.06.072

        Examples
        --------
        >>> seq = Sequence.from_integer_ratios([2, 2, 1, 1], value_of_one_in_ms=500)
        >>> print(seq.iois)
        [1000. 1000.  500.  500.]
        >>> print(seq.interval_ratios_from_dyads)
        [0.5       0.6666667 0.5      ]

        """

        return np.array([self.iois[k] / (self.iois[k] + self.iois[k + 1]) for k in range(len(self.iois) - 1)])


def _join_sequences(iterator):
    """
    This helper function joins metrical Sequence objects (it is used in the __add__ of Sequence).
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
