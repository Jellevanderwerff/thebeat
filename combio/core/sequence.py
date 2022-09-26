from __future__ import annotations  # this is to make sure we can type hint the return value in a class method
from fractions import Fraction
from typing import Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import combio._helpers


class BaseSequence:
    """
    This is the most basic of classes that the Sequence class inherits from, as well as the Rhythm class.
    It cannot do many things, apart from holding a number of inter-onset intervals (IOIs).

    The BaseSequence class dictates that a sequence can either be metrical or not.
    The default is non-metrical, meaning that if there are `n` onset values (i.e. `t` values),
    there are `n`-1 IOIs. This is what people will need in most cases.
    Metrical sequences have an additional final IOI (so they end with a gap of silence).
    This is what you will need in cases with e.g. rhythmical/musical sequences.

    The BaseSequence class protects against impossible values for the IOIs, as well as for the
    event onsets (`t` values).

    Finally, remember that the first event onset is always at :math:`t = 0`!

    Attributes
    ----------
    iois : NumPy 1-D array
        Contains the inter-onset intervals (IOIs). This is the bread and butter of the BaseSequence class.
        Non-metrical sequences have n IOIs and n+1 onsets. Metrical sequences have an equal number of IOIs
        and onsets.
    metrical : bool
        If False, sequence has an n-1 inter-onset intervals (IOIs) for n event onsets. If True,
        sequence has an equal number of IOIs and event onsets.
    """

    def __init__(self,
                 iois: Union[list, np.ndarray],
                 metrical: Optional[bool] = False):
        """Initialization of BaseSequence class."""

        # Save attributes
        self.iois = iois
        self.metrical = metrical

    @property
    def iois(self) -> np.ndarray:
        """The inter-onset intervals (IOIs) of the Sequence object. These are the intervals in milliseconds
        between the onset of an event, and the onset of the next event. This is the most important
        attribute of the Sequence class and is used throughout.

        This getter returns a copy of the IOIs instead of the actual variable.
        """

        return np.array(self._iois, dtype=np.float64, copy=True)

    @iois.setter
    def iois(self, values):
        """IOI setter. Checks against negative IOIs."""

        # We always want a NumPy array
        iois = np.array(values, dtype=np.float64, copy=True)

        if np.any(iois <= 0):
            raise ValueError("Inter-onset intervals (IOIs) cannot be zero or negative.")

        self._iois = iois

    @property
    def onsets(self) -> np.ndarray:
        """ Returns the event onsets (t values) in milliseconds on the basis of the sequence objects'
        inter-onset intervals (IOIs). An additional first onset is additionally prepended at :math:`t = 0`.
        """

        if self.metrical is True:
            return np.cumsum(np.append(0, self.iois[:-1]))
        else:
            return np.cumsum(np.append(0, self.iois))

    @onsets.setter
    def onsets(self, values):
        """Setter for the event onsets. First onset must be zero, onsets must be in order,
        and there cannot be two simultaneous onsets that occur simultaneously.
        """

        # Check whether first onset is 0 (requirement of this package)
        if values[0] != 0:
            raise ValueError("First onset is not 0")

        if np.any(values[:-1] >= values[1:]):
            raise ValueError("Onsets are not ordered strictly monotonically.")

        # Set the IOIs
        if self.metrical is True:
            raise ValueError("Cannot change onsets of metrical sequences. This is because we need to know the final "
                             "IOI for metrical sequences. Either reconstruct the sequence, or change the IOIs.")

        self._iois = np.array(np.diff(values), dtype=np.float64)


class Sequence(BaseSequence):
    """
    The Sequence class is the most important class in this package. It is used as the basis
    for many functions as it contains timing information in the form of inter-onset intervals (IOIs; the times between
    the onset of an event, and the onset of the next event) and event onsets (i.e. `t` values).
    IOIs are what we use to construct :py:class:`Sequence` objects.

    The most basic way of constructing a :py:class:`Sequence` object is by passing it a list (or other iterable) of
    IOIs. However, the different class methods (e.g. :py:meth:`Sequence.generate_isochronous`) may also be used.

    This class additionally contains methods and attributes to, for instance, get the event onset values, to
    change the tempo, add Gaussian noise, or to plot the :py:class:`Sequence` object using matplotlib.


    Attributes
    ----------
    iois : :class:`numpy.ndarray`
        One-dimensional array containing the inter-onset intervals (IOIs). This is the bread and butter of the
        :py:class:`Sequence` class.
        Non-metrical sequences have `n` IOIs and `n`+1 onsets. Metrical sequences have an equal number of IOIs
        and onsets.
    metrical : bool
        If ``False``, sequence has an `n`-1 inter-onset intervals (IOIs) for n event onsets. If ``True``,
        sequence has an equal number of IOIs and event onsets.
    name : str
        If desired, one can give a Sequence object a name. This is for instance used when printing the sequence,
        or when plotting the sequence. It can always be retrieved and changed via this attribute.

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
                 iois: Union[list, np.ndarray],
                 metrical: bool = False,
                 name: Optional[str] = None):
        """Construct a Sequence class on the basis of inter-onset intervals (IOIs).
        When metrical is ``True``, the sequence contains an equal number of IOIs and event onsets.
        If ``False`` (the default), the sequence contains `n` event onsets, and `n`-1 IOIs.

        Parameters
        ----------
        iois
            An iterable of inter-onset intervals (IOIs). For instance: ``[500, 500, 400, 200]``
        metrical
            Indicates whether sequence has an extra final inter-onset interval; this is useful for musical/rhythmical
            sequences.
        name
            Optionally, you can give the Sequence object a name. This is used when printing, plotting, or writing
            the Sequence object.

        Examples
        --------
        >>> iois = [500, 400, 600, 400]
        >>> seq = Sequence(iois)
        >>> print(seq.onsets)
        [   0.  500.  900. 1500. 1900.]

        """

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

        # Check whether all the objects are of the same type
        if not isinstance(other, Sequence):
            raise ValueError("Right-hand side object is not a Sequence")

        # Sequence objects need to be metrical:
        if not self.metrical:
            raise ValueError("The left-hand side Sequence must be metrical. Otherwise, we miss an inter-onset interval"
                             "(IOI) in between the joined sequences. Try creating a Sequence with the metrical=True "
                             "flag, this means there's an equal number of IOIs and onsets.")

        iois = np.concatenate([self.iois, other.iois])

        if other.metrical:
            return Sequence(iois, metrical=True)
        else:
            return Sequence(iois)

    def __len__(self):
        return len(self.onsets)

    @classmethod
    def from_integer_ratios(cls,
                            numerators: Union[np.array, list],
                            value_of_one_in_ms: int,
                            metrical: bool = False,
                            name: Optional[str] = None) -> Sequence:
        """

        This class method can be used to construct a new :py:class:`Sequence` object on the basis of integer ratios.

        Parameters
        ----------
        numerators
            Contains the numerators of the integer ratios. For instance: ``[1, 2, 4]``
        value_of_one_in_ms
            This represents the duration of the 1, multiples of this value are used.
            For instance, a sequence of `[2, 4]` using `value_of_one_in_ms=500` would be a :py:class:`Sequence` with
            IOIs: ``[1000 2000]``.
        metrical
            Indicates whether a metrical or non-metrical sequence should be generated
            (see :py:attr:`Sequence.metrical`). Defaults to ``False``.
        name
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute.

        Returns
        -------
        Sequence
            A newly constructed :py:class:`Sequence` object.

        Examples
        --------
        >>> seq = Sequence.from_integer_ratios(numerators=[1, 2, 4], value_of_one_in_ms=500)
        >>> print(seq.iois)
        [ 500. 1000. 2000.]
        """

        numerators = np.array(numerators)
        return cls(numerators * value_of_one_in_ms, metrical=metrical, name=name)

    @classmethod
    def from_onsets(cls,
                    onsets: Union[np.ndarray, list],
                    name: Optional[str] = None) -> Sequence:
        """
        Class method that can be used to generate a new :py:class:`Sequence` object on the basis of event onsets.

        Parameters
        ----------
        onsets
            An iterable of event onsets which must start from 0, e.g.: ``[0, 500, 1000]``.
        name
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute (`Sequence.name`).

        Returns
        -------
        Sequence
            A newly constructed :py:class:`Sequence` object.

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
                               rng: Optional[np.random.Generator] = None,
                               metrical: bool = False,
                               name: Optional[str] = None) -> Sequence:
        """
        Class method that generates a py:class:`Sequence` object with random inter-onset intervals (IOIs) based on the
        normal distribution.

        Parameters
        ----------
        n
            The desired number of events in the sequence.
        mu
            The mean of the normal distribution.
        sigma
            The standard deviation of the normal distribution.
        rng
            A :class:`numpy.random.Generator` object. If not supplied :func:`numpy.random.default_rng` is
            used.
        metrical
            Indicates whether a metrical or non-metrical sequence should be generated
            (see :py:attr:`Sequence.metrical`).
        name
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute.

        Returns
        -------
        Sequence
            A newly created :py:class:`Sequence` object.

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
                                rng: Optional[np.random.Generator] = None,
                                metrical: bool = False,
                                name: Optional[str] = None) -> Sequence:
        """
        Class method that generates a sequence of random inter-onset intervals based on a uniform distribution.

        Parameters
        ----------
        n
            The desired number of events in the sequence.
        a
            The left bound of the uniform distribution.
        b
            The right bound of the uniform distribution.
        rng
            A :class:`numpy.random.Generator` object. If not supplied :func:`numpy.random.default_rng` is
            used.
        metrical
            Indicates whether a metrical or non-metrical sequence should be generated
            (see :py:attr:`Sequence.metrical`).
        name
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute.

        Returns
        -------
        Sequence
            A newly created py:class:`Sequence` object.

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

        return cls(round_iois, metrical=metrical, name=name)

    @classmethod
    def generate_random_poisson(cls,
                                n: int,
                                lam: int,
                                rng: Optional[np.random.Generator] = None,
                                metrical: bool = False,
                                name: Optional[str] = None) -> Sequence:

        """
        Class method that generates a sequence of random inter-onset intervals based on a Poisson distribution.

        Parameters
        ----------
        n
            The desired number of events in the sequence.
        lam
            The desired value for lambda.
        rng
            A :class:`numpy.random.Generator` object. If not supplied :func:`numpy.random.default_rng` is
            used.
        metrical
            Indicates whether a metrical or non-metrical sequence should be generated
            (see :py:attr:`Sequence.metrical`).
        name
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute.

        Returns
        -------
        Sequence
            A newly created :py:class:`Sequence` object.

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
                                    rng: Optional[np.random.Generator] = None,
                                    metrical: bool = False,
                                    name: Optional[str] = None) -> Sequence:
        """

        Class method that generates a sequence of random inter-onset intervals based on an exponential distribution.

        Parameters
        ----------
        n
            The desired number of events in the sequence.
        lam
           The desired value for lambda.
        rng
            A :class:`numpy.random.Generator` object. If not supplied NumPy's :func:`numpy.random.default_rng` is
            used.
        metrical
            Indicates whether a metrical or non-metrical sequence should be generated
            (see :py:attr:`Sequence.metrical`).
        name
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via this attribute.

        Returns
        -------
        Sequence
            A newly created :py:class:`Sequence` object.

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

        return cls(round_iois, metrical=metrical, name=name)

    @classmethod
    def generate_isochronous(cls,
                             n: int,
                             ioi: int,
                             metrical: bool = False,
                             name: Optional[str] = None) -> Sequence:
        """
        Class method that generates a sequence of isochronous (i.e. equidistant) inter-onset intervals.
        Note that there will be `n`-1 IOIs in a sequence. IOIs are rounded off to integers.

        Parameters
        ----------
        n
            The desired number of events in the sequence.
        ioi
            The inter-onset interval to be used between all events.
        metrical
            Indicates whether a metrical or non-metrical sequence should be generated
            (see :py:attr:`Sequence.metrical`).
        name
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
                           rng: Optional[np.random.Generator] = None) -> None:
        """
        This method can be used to add some Gaussian noise to the inter-onset intervals (IOIs)
        of the Sequence object. It uses a normal distribution with mean 0, and a standard deviation
        of ``noise_sd``.

        Parameters
        ----------
        noise_sd
            The standard deviation of the normal distribution used for adding in noise.
        rng
            A Numpy Generator object. If none is supplied, :func:`numpy.random.default_rng` is used.

        Examples
        --------
        >>> gen = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
        >>> print(seq.iois)
        [500. 500. 500. 500.]
        >>> seq.add_noise_gaussian(noise_sd=50, rng=gen)
        >>> print(seq.iois)
        [450.54393248 481.61066743 564.39626306 509.69872096]
        >>> print(seq.onsets)
        [   0.          450.54393248  932.15459991 1496.55086297 2006.24958393]
        """
        if rng is None:
            rng = np.random.default_rng()
        self.iois = self.iois + rng.normal(loc=0, scale=noise_sd, size=len(self.iois))

    def change_tempo(self,
                     factor: Union[int, float]) -> None:
        """
        Change the tempo of the Sequence object, where a factor of 1 or bigger increases the tempo (but results in
        smaller inter-onset intervals). A factor between 0 and 1 decreases the tempo (but results in larger
        inter-onset intervals).

        Parameters
        ----------
        factor
            Tempo change factor. E.g. 2 means twice as fast. 0.5 means twice as slow.

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
        This method can be used for creating a ritardando or accelerando effect in the inter-onset intervals (IOIs).
        It divides the IOIs by a vector linearly spaced between 1 and total_change.

        Parameters
        ----------
        total_change
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
        self.iois /= np.linspace(1, total_change, len(self.iois))

    # Visualization
    def plot(self,
             style: str = 'seaborn',
             title: Optional[str] = None,
             linewidth: int = 50,
             figsize: Optional[tuple] = None,
             suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the Sequence object as an event plot on the basis of the event onsets.

        In principle, the `x` axis shows milliseconds. However, if the total sequence duration is over 10 seconds,
        this changes to seconds.

        Parameters
        ----------
        style
            Matplotlib style to use for the plot. Defaults to 'seaborn'.
            See `matplotlib style sheets reference <Style sheets reference>`_.
        title
            If desired, one can provide a title for the plot. This takes precedence over using the
            StimSequence name as the title of the plot (if the object has one).
        linewidth
            The desired width of the bars (events) in milliseconds. Defaults to 50 milliseconds.
        figsize
            The desired figure size in inches as a tuple: ``(width, height)``.
        suppress_display
            If ``True``, the plot is only returned, and not displayed via :func:`matplotlib.pyplot.show`.

        Returns
        -------
        fig
            A matplotlib Figure object
        ax
            A matplotlib Axes object

        Examples
        --------
        >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
        >>> seq.plot()  # doctest: +SKIP
        """

        # If a title was provided that has preference. If none is provided,
        # use the Sequence object's name. Otherwise use ``None``.
        if title:
            title = title
        elif self.name:
            title = self.name

        # Linewidths
        linewidths = np.repeat(linewidth, len(self.onsets))

        fig, ax = combio._helpers.plot_sequence_single(onsets=self.onsets, style=style, title=title,
                                                       linewidths=linewidths, figsize=figsize,
                                                       suppress_display=suppress_display)

        return fig, ax

    @property
    def duration_ms(self) -> np.float64:
        """Get the total duration of the :py:class:`Sequence` object in milliseconds.
        """
        return np.float64(np.sum(self.iois))

    @property
    def duration_s(self) -> np.float64:
        """Get the total duration of the :py:class:`Sequence` object in seconds.
        """
        return np.float64(np.sum(self.iois) / 1000)

    @property
    def integer_ratios(self) -> np.array:
        """
        This property calculates how to describe a sequence of IOIs in integer ratio numerators from
        the total duration of the sequence by finding the least common multiplier.

        Example:
        A sequence of IOIs ``[250, 500, 1000, 250]`` has a total duration of 2000 ms.
        This can be described using the least common multiplier as
        :math:`\frac{1}{8}, \frac{2}{8}, \frac{4}{8}, \frac{1}{8}`,
        so this method returns the numerators ``[1, 2, 4, 1]``.

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
    def interval_ratios_from_dyads(self) -> np.array:
        r"""
        This property returns sequential interval ratios, calculated as:

        .. math::

            \textrm{ratio}_k = \frac{\textrm{IOI}_k}{\textrm{IOI}_k + \textrm{IOI}_{k+1}}


        Note that for `n` IOIs this property returns `n`-1 ratios.

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
        [0.5        0.66666667 0.5       ]

        """

        return np.array([self.iois[k] / (self.iois[k] + self.iois[k + 1]) for k in range(len(self.iois) - 1)])
