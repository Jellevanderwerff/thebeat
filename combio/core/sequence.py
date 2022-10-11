from __future__ import annotations
from fractions import Fraction
from typing import Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import combio._helpers


class BaseSequence:
    """This is the most basic of classes that the Sequence class inherits from, as well as the Rhythm class.
    It cannot do many things, apart from holding a number of inter-onset intervals (IOIs).

    The BaseSequence class dictates that a sequence can either be metrical or not.
    The default is non-metrical, meaning that if there are *n* onset values (i.e. *t* values),
    there are *n*-1 IOIs. This is what people will need in most cases.
    Metrical sequences have an additional final IOI (so they end with a gap of silence).
    This is what you will need in cases with e.g. rhythmical/musical sequences.

    The BaseSequence class protects against impossible values for the IOIs, as well as for the
    event onsets (*t* values).

    Attributes
    ----------
    iois : NumPy 1-D array
        Contains the inter-onset intervals (IOIs). This is the bread and butter of the BaseSequence class.
        Non-metrical sequences have n IOIs and n+1 onsets. Metrical sequences have an equal number of IOIs
        and onsets.
    metrical : bool
        If False, sequence has an n-1 inter-onset intervals (IOIs) for n event onsets. If True,
        sequence has an equal number of IOIs and event onsets.
    name : str
        If desired, one can give the (Base)Sequence object a name. This is for instance used when printing the sequence,
        or when plotting the sequence. It can always be retrieved and changed via this attribute.

    """

    def __init__(self,
                 iois: npt.ArrayLike[float],
                 first_onset: float = 0.0,
                 metrical: Optional[bool] = False,
                 name: Optional[str] = None):
        """Initialization of BaseSequence class."""

        if metrical is True and first_onset != 0:
            raise ValueError("First onset must be 0 for metrical sequences.")

        # Save attributes
        self.iois = iois
        self._first_onset = first_onset
        self.metrical = metrical
        # Additionally save the provided name (may be None)
        self.name = name

    @property
    def iois(self) -> np.ndarray:
        """The inter-onset intervals (IOIs) of the Sequence object. These are the intervals in milliseconds
        between the onset of an event, and the onset of the next event. This is the most important
        attribute of the Sequence class and is used throughout.

        This getter returns a copy of the IOIs instead of the actual attribute.
        """

        return np.array(self._iois, dtype=np.float64, copy=True)

    @iois.setter
    def iois(self, values: npt.ArrayLike[float]):
        """IOI setter. Checks against negative IOIs."""

        # We always want a NumPy array
        iois = np.array(values, dtype=np.float64, copy=True)

        if np.any(iois <= 0):
            raise ValueError("Inter-onset intervals (IOIs) cannot be zero or negative.")

        self._iois = iois

    @property
    def onsets(self) -> np.ndarray:
        """Returns the event onsets (t values) on the basis of the sequence objects'
        inter-onset intervals (IOIs).
        """

        if self.metrical is True:
            return np.cumsum(np.append(self._first_onset, self.iois[:-1]))
        else:
            return np.cumsum(np.append(self._first_onset, self.iois))

    @onsets.setter
    def onsets(self, values):
        """Setter for the event onsets. Onsets must be in order, and there cannot be two
        simultaneous onsets that occur simultaneously.
        """

        # Set the IOIs
        if self.metrical is True:
            raise ValueError("Cannot change onsets of metrical sequences. This is because we need to know the final "
                             "IOI for metrical sequences. Either reconstruct the sequence, or change the IOIs.")

        values = np.array(values, dtype=np.float64)
        if np.any(values[:-1] >= values[1:]):
            raise ValueError("Onsets are not ordered strictly monotonically.")

        self._iois = np.array(np.diff(values), dtype=np.float64)
        self._first_onset = float(values[0])

    @property
    def mean_ioi(self) -> np.float64:
        """The average inter-onset interval (IOI)."""
        return np.float64(np.mean(self.iois))

    @property
    def duration(self) -> np.float64:
        """Property that returns the summed total of the inter-onset intervals."""
        return np.float64(np.sum(self.iois))


class Sequence(BaseSequence):
    """
    The Sequence class is the most important class in this package. It is used as the basis
    for many functions as it contains timing information in the form of inter-onset intervals (IOIs; the times between
    the onset of an event, and the onset of the next event) and event onsets (i.e. *t* values).
    IOIs are what we use to construct :py:class:`Sequence` objects.

    The most basic way of constructing a :py:class:`Sequence` object is by passing it a list or array of
    IOIs. However, the different class methods (e.g. :py:meth:`Sequence.generate_isochronous`) may also be used.

    This class additionally contains methods and attributes to, for instance,
    change the tempo, add Gaussian noise, or to plot the :py:class:`Sequence` object using matplotlib.

    For more info, check out the :py:meth:`~combio.core.Sequence.__init__` method, and the different methods below.
    """

    def __init__(self,
                 iois: npt.ArrayLike[float],
                 first_onset: float = 0.0,
                 metrical: bool = False,
                 name: Optional[str] = None):
        """Construct a Sequence class on the basis of inter-onset intervals (IOIs).
        When metrical is ``True``, the sequence contains an equal number of IOIs and event onsets.
        If ``False`` (the default), the sequence contains *n* event onsets, and *n*-1 IOIs.

        Parameters
        ----------
        iois
            An iterable of inter-onset intervals (IOIs). For instance: ``[500, 500, 400, 200]``.
        metrical
            Indicates whether sequence has an extra final inter-onset interval; this is useful for musical/rhythmical
            sequences.
        name
            Optionally, you can give the Sequence object a name. This is used when printing, plotting, or writing
            the Sequence object. It can always be retrieved and changed via :py:attr:`Sequence.name`.

        Examples
        --------
        >>> iois = [500, 400, 600, 400]
        >>> seq = Sequence(iois)
        >>> print(seq.onsets)
        [   0.  500.  900. 1500. 1900.]

        """

        # Call super init method
        super().__init__(iois=iois, first_onset=first_onset, metrical=metrical, name=name)

    def __str__(self):
        name = self.name if self.name else "Not provided"
        metricality = "(metrical)" if self.metrical else "(non-metrical)"

        return (f"Object of type Sequence {metricality}\n"
                f"{len(self.onsets)} events\n"
                f"IOIs: {self.iois}\n"
                f"Onsets: {self.onsets}\n"
                f"Sequence name: {name}")

    def __add__(self, other):
        # Check whether the objects are of the same type
        if not isinstance(other, Sequence):
            raise ValueError("Right-hand side object is not a Sequence object.")

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

    @classmethod
    def from_integer_ratios(cls,
                            numerators: npt.ArrayLike[float],
                            value_of_one: float,
                            metrical: bool = False,
                            name: Optional[str] = None) -> Sequence:
        """This class method can be used to construct a new :py:class:`Sequence` object on the basis of integer ratios.
        See :py:attr:`Sequence.integer_ratios` for explanation.

        Parameters
        ----------
        numerators
            Contains the numerators of the integer ratios. For instance: ``[1, 2, 4]``
        value_of_one
            This represents the duration of the 1, multiples of this value are used.
            For instance, a sequence of ``[2, 4]`` using ``value_of_one=500`` would be a :py:class:`Sequence` with
            IOIs: ``[1000 2000]``.
        metrical
            Indicates whether a metrical or non-metrical sequence should be generated
            (see :py:attr:`Sequence.metrical`). Defaults to ``False``.
        name
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via :py:attr:`Sequence.name`.


        Examples
        --------
        >>> seq = Sequence.from_integer_ratios(numerators=[1, 2, 4], value_of_one=500)
        >>> print(seq.iois)
        [ 500. 1000. 2000.]
        """

        numerators = np.array(numerators)
        return cls(numerators * value_of_one, metrical=metrical, name=name)

    @classmethod
    def from_onsets(cls,
                    onsets: Union[np.ndarray[float], list[float]],
                    name: Optional[str] = None) -> Sequence:
        """Class method that can be used to generate a new :py:class:`Sequence` object on the basis of event onsets.

        Parameters
        ----------
        onsets
            An array or list containg event onsets, for instance: ``[0, 500, 1000]``. The onsets do not have to
            start with zero.
        name
            If desired, one can give a sequence a name. This is for instance used when printing the sequence,
            or when plotting the sequence. It can always be retrieved and changed via :py:attr:`Sequence.name`.


        Examples
        --------
        >>> seq = Sequence.from_onsets([0, 500, 1000])
        >>> print(seq.iois)
        [500. 500.]
        """
        iois = np.diff(onsets)

        return cls(iois, first_onset=onsets[0], metrical=False, name=name)

    @classmethod
    def generate_random_normal(cls,
                               n: int,
                               mu: float,
                               sigma: float,
                               rng: Optional[np.random.Generator] = None,
                               metrical: bool = False,
                               name: Optional[str] = None) -> Sequence:
        """Class method that generates a py:class:`Sequence` object with random inter-onset intervals (IOIs) based on the
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
            or when plotting the sequence. It can always be retrieved and changed via :py:attr:`Sequence.name`.


        Examples
        --------
        >>> generator = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_random_normal(n=5, mu=500, sigma=50, rng=generator)
        >>> print(seq.iois)
        [450.54393248 481.61066743 564.39626306 509.69872096]

        >>> seq = Sequence.generate_random_normal(n=5, mu=500, sigma=50, metrical=True)
        >>> len(seq.onsets) == len(seq.iois)
        True
        """
        if rng is None:
            rng = np.random.default_rng()

        # Number of IOIs depends on metricality
        n_iois = n if metrical else n - 1

        return cls(rng.normal(loc=mu, scale=sigma, size=n_iois), metrical=metrical, name=name)

    @classmethod
    def generate_random_uniform(cls,
                                n: int,
                                a: float,
                                b: float,
                                rng: Optional[np.random.Generator] = None,
                                metrical: bool = False,
                                name: Optional[str] = None) -> Sequence:
        """Class method that generates a sequence of random inter-onset intervals based on a uniform distribution.

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
            or when plotting the sequence. It can always be retrieved and changed via :py:attr:`Sequence.name`.


        Examples
        --------
        >>> generator = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_random_uniform(n=5, a=400, b=600, rng=generator)
        >>> print(seq.iois)
        [536.47037265 410.76420376 444.07197455 436.87436214]

        >>> seq = Sequence.generate_random_uniform(n=5, a=400, b=600, metrical=True)
        >>> len(seq.onsets) == len(seq.iois)
        True
        """

        if rng is None:
            rng = np.random.default_rng()

        # Number of IOIs depends on metricality
        n_iois = n if metrical else n - 1

        return cls(rng.uniform(low=a, high=b, size=n_iois), metrical=metrical, name=name)

    @classmethod
    def generate_random_poisson(cls,
                                n: int,
                                lam: float,
                                rng: Optional[np.random.Generator] = None,
                                metrical: bool = False,
                                name: Optional[str] = None) -> Sequence:

        """Class method that generates a sequence of random inter-onset intervals based on a Poisson distribution.

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
            or when plotting the sequence. It can always be retrieved and changed via :py:attr:`Sequence.name`.


        Examples
        --------
        >>> generator = np.random.default_rng(123)
        >>> seq = Sequence.generate_random_poisson(n=5, lam=500, rng=generator)
        >>> print(seq.iois)
        [512. 480. 476. 539.]

        """
        if rng is None:
            rng = np.random.default_rng()

        # Number of IOIs depends on metricality
        n_iois = n if metrical else n - 1

        return cls(rng.poisson(lam=lam, size=n_iois), metrical=metrical, name=name)

    @classmethod
    def generate_random_exponential(cls,
                                    n: int,
                                    lam: float,
                                    rng: Optional[np.random.Generator] = None,
                                    metrical: bool = False,
                                    name: Optional[str] = None) -> Sequence:
        """Class method that generates a sequence of random inter-onset intervals based on an exponential distribution.

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
            or when plotting the sequence. It can always be retrieved and changed via :py:attr:`Sequence.name`.


        Examples
        --------
        >>> generator = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_random_exponential(n=5, lam=500, rng=generator)
        >>> print(seq.iois)
        [298.48624756  58.51553052 125.89734975 153.98272273]

        """
        if rng is None:
            rng = np.random.default_rng()

        n_iois = n if metrical else n - 1

        return cls(rng.exponential(scale=lam, size=n_iois), metrical=metrical, name=name)

    @classmethod
    def generate_isochronous(cls,
                             n: int,
                             ioi: float,
                             metrical: bool = False,
                             name: Optional[str] = None) -> Sequence:
        """
        Class method that generates a sequence of isochronous (i.e. equidistant) inter-onset intervals.
        Note that there will be *n*-1 IOIs in a sequence. IOIs are rounded off to integers.

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
            or when plotting the sequence. It can always be retrieved and changed via :py:attr:`Sequence.name`.


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

        n_iois = n if metrical else n - 1

        return cls([ioi] * n_iois, metrical=metrical, name=name)

    # Manipulation methods
    def add_noise_gaussian(self,
                           noise_sd: float,
                           rng: Optional[np.random.Generator] = None):
        """This method can be used to add some Gaussian noise to the inter-onset intervals (IOIs)
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

        self.iois += rng.normal(loc=0, scale=noise_sd, size=len(self.iois))

    def change_tempo(self,
                     factor: float):
        """Change the tempo of the :py:`Sequence` object, where a factor of 1 or bigger increases the tempo 
        (but results in smaller inter-onset intervals). A factor between 0 and 1 decreases the tempo (but results in 
        larger inter-onset intervals).

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
                              total_change: float):
        """Create a ritardando or accelerando effect in the inter-onset intervals (IOIs).
        It divides the IOIs by a vector linearly spaced between 1 and ``total_change``.

        Parameters
        ----------
        total_change
            Total tempo change at the end of the `Sequence` compared to the beginning.
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

        self.iois /= np.linspace(start=1, stop=total_change, num=len(self.iois))

    def round_onsets(self,
                     decimals: int = 0):
        """Use this function to round off the :py:`Sequence` object's onsets (i.e. *t* values). This can, for instance,
        be useful to get rid of warnings that are the result of frame rounding. See e.g. :py:`StimSequence.__init__`.

        Parameters
        ----------
        decimals
            The number of decimals desired.

        """

        self.onsets = np.round(self.onsets, decimals=decimals)

    # Visualization
    def plot_sequence(self,
                      style: str = 'seaborn',
                      title: Optional[str] = None,
                      x_axis_label: str = "Time",
                      linewidth: Optional[Union[npt.ArrayLike[float], float]] = None,
                      figsize: Optional[tuple] = None,
                      suppress_display: bool = False) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the :py:class:`Sequence` object as an event plot on the basis of the event onsets.

        Parameters
        ----------
        style
            Matplotlib style to use for the plot. Defaults to 'seaborn'.
            See `matplotlib style sheets reference
            <https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html>`_.
        title
            If desired, one can provide a title for the plot. This takes precedence over using the
            StimSequence name as the title of the plot (if the object has one).
        x_axis_label
            A label for the x axis.
        linewidth
            The desired width of the bars (events). Defaults to 1/10th of the smallest inter-onset interval (IOI).
        figsize
            A tuple containing the desired output size of the plot in inches, e.g. ``(4, 1)``.
            This refers to the ``figsize`` parameter in :func:`matplotlib.pyplot.figure`.
        suppress_display
            If ``True``, the plot is only returned, and not displayed via :func:`matplotlib.pyplot.show`.

        Examples
        --------
        >>> seq = Sequence.generate_isochronous(n=5, ioi=500)
        >>> seq.plot_sequence()  # doctest: +SKIP
        """

        # For the title, use the Sequence name if it has one. Otherwise use the title parameter,
        # which may be None.
        title = self.name if self.name is not None else title

        # Linewidths
        if linewidth is None:
            linewidths = np.repeat(np.min(self.iois) / 10, len(self.onsets))
        elif isinstance(linewidth, float):
            linewidths = np.repeat(linewidth, len(self.onsets))
        else:
            linewidths = np.array(linewidth)

        # Plot the sequence
        fig, ax = combio._helpers.plot_sequence_single(onsets=self.onsets, style=style, title=title,
                                                       x_axis_label=x_axis_label,
                                                       linewidths=linewidths, figsize=figsize,
                                                       suppress_display=suppress_display)

        return fig, ax

    @property
    def integer_ratios(self) -> np.ndarray:
        r"""Calculate how to describe a sequence of IOIs in integer ratio numerators from
        the total duration of the sequence by finding the least common multiplier.

        Example
        -------

        A sequence of IOIs ``[250, 500, 1000, 250]`` has a total duration of 2000 ms.
        This can be described using the least common multiplier as
        :math:`\frac{1}{8}, \frac{2}{8}, \frac{4}{8}, \frac{1}{8}`,
        so this method returns the numerators ``[1, 2, 4, 1]``.

        Notes
        -----
        The method for calculating the integer ratios is based on :footcite:t:`jacobyIntegerRatioPriors2017`.

        Examples
        --------
        >>> seq = Sequence([250, 500, 1000, 250])
        >>> print(seq.integer_ratios)
        [1 2 4 1]

        """

        fractions = [Fraction(int(ioi), int(self.duration)) for ioi in self.iois]
        lcm = np.lcm.reduce([fr.denominator for fr in fractions])

        vals = [int(fr.numerator * lcm / fr.denominator) for fr in fractions]

        return np.array(vals)

    @property
    def interval_ratios_from_dyads(self) -> np.ndarray:
        r"""Return sequential interval ratios, calculated as:
        :math:`\textrm{ratio}_k = \frac{\textrm{IOI}_k}{\textrm{IOI}_k + \textrm{IOI}_{k+1}}`.

        Note that for *n* IOIs this property returns *n*-1 ratios.

        Notes
        -----
        The used method is based on the methodology from :footcite:t:`roeskeCategoricalRhythmsAre2020`.

        Examples
        --------
        >>> seq = Sequence.from_integer_ratios([2, 2, 1, 1], value_of_one=500)
        >>> print(seq.iois)
        [1000. 1000.  500.  500.]
        >>> print(seq.interval_ratios_from_dyads)
        [0.5        0.66666667 0.5       ]

        """

        iois = self.iois
        return iois[:-1] / (iois[1:] + iois[:-1])
