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

from __future__ import annotations

import copy
import numbers
import pathlib
import re
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

import thebeat.helpers


class BaseSequence:
    """This is the most basic of classes that the :py:class:`~thebeat.core.Sequence`,
    :py:class:`~thebeat.music.Rhythm`, and :py:class:`~thebeat.core.SoundSequence` classes inherit
    from. It cannot do many things, apart from holding a number of inter-onset intervals (IOIs).

    The BaseSequence class dictates that a sequence can either end with an interval or not.
    The default is to end with an event, meaning that if there are *n* onset values
    (i.e. *t* values), there are *n*-1 IOIs. This is what people will need in most cases.
    Sequences that end with an interval have an IOI at the end (so they end with a gap of silence).
    This is what you will need in cases with e.g. rhythmical/musical sequences.

    The BaseSequence class protects against impossible values for the IOIs, as well as for the
    event onsets (*t* values).

    Attributes
    ----------
    iois : NumPy 1-D array
        Contains the inter-onset intervals (IOIs). This is the bread and butter of the BaseSequence
        class. Sequences that end with an event have *n* onsets and *n*-1 IOIs. Sequences that end
        with an interval have an equal number of IOIs and onsets.
    end_with_interval : bool
        If ``False``, sequence has *n*-1 inter-onset intervals (IOIs) for *n* event onsets.
        If ``True``, sequence has an equal number of IOIs and event onsets.
    name : str
        If desired, one can give the object a name. This is for instance used when printing the
        sequence, or when plotting the sequence. It can always be retrieved and changed via this
        attribute.

    """

    def __init__(
        self,
        iois: npt.ArrayLike[float],
        first_onset: float = 0.0,
        end_with_interval: bool | None = False,
        name: str | None = None,
    ):
        """Initialization of BaseSequence class."""

        # Save attributes
        self.iois = iois
        self._first_onset = first_onset
        self.end_with_interval = end_with_interval
        # Additionally save the provided name (can be None)
        self.name = name

    def copy(self, deep: bool = True):
        """Returns a copy of itself. See :py:func:`copy.copy` for more information.

        Parameters
        ----------
        deep
            If ``True``, a deep copy is returned. If ``False``, a shallow copy is returned.

        """
        if deep is True:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)

    @property
    def iois(self) -> np.ndarray:
        """The inter-onset intervals (IOIs) of the Sequence object. These are the intervals in
        milliseconds between the onset of an event, and the onset of the next event.
        This is the most important attribute of the Sequence class and is used throughout.

        This getter returns a copy of the IOIs instead of the actual attribute.
        """

        return np.array(self._iois, dtype=np.float64, copy=True)

    @iois.setter
    def iois(self, values: npt.ArrayLike[float]):
        """IOI setter. Checks against negative (or zero) IOIs."""

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

        if self.end_with_interval is True:
            return np.cumsum(np.append(self._first_onset, self.iois[:-1]))
        else:
            return np.cumsum(np.append(self._first_onset, self.iois))

    @onsets.setter
    def onsets(self, values):
        """Setter for the event onsets. Onsets must be in order, and there cannot be two
        simultaneous onsets that occur simultaneously.
        """

        # Set the IOIs
        if self.end_with_interval is True:
            raise ValueError(
                "Cannot change onsets of sequences that end with an interval. This is because we need to "
                "know the final IOI for such sequences. Either reconstruct the sequence, or change the "
                "IOIs."
            )

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
    Arguably, the :py:class:`~thebeat.core.Sequence` class is the most important class in this
    package. It is used as the basis for many functions as it contains timing information in the
    form of inter-onset intervals (IOIs; the times between the onset of an event, and the onset of
    the next event) and event onsets (i.e. *t* values).
    IOIs are what we use to construct :py:class:`Sequence` objects.

    The most basic way of constructing a :py:class:`Sequence` object is by passing it a list or
    array of IOIs (see under :py:meth:`~thebeat.core.Sequence.__init__`).
    However, the different class methods (e.g. :py:meth:`Sequence.generate_isochronous`) may
    also be used.

    For the :py:class:`Sequence` class it does not matter  whether the provided IOIs are in seconds
    or milliseconds. However, it does matter when passing the :py:class:`Sequence` object to e.g.
    a :py:class:`SoundSequence` object (see :py:meth:`SoundSequence.__init__`).

    This class additionally contains methods and attributes to, for instance, change the tempo,
    add Gaussian noise, or to plot the :py:class:`Sequence` object using matplotlib.

    For more info, check out the :py:meth:`~thebeat.core.Sequence.__init__` method, and the
    different methods below.
    """

    def __init__(
        self,
        iois: npt.ArrayLike[float],
        first_onset: float = 0.0,
        end_with_interval: bool = False,
        name: str | None = None,
    ):
        """Construct a Sequence class on the basis of inter-onset intervals (IOIs).
        When ``end_with_interval`` is ``False`` (the default), the sequence contains *n* event
        onsets, but *n*-1 IOIs. If ``True``, the sequence contains an equal number of event onsets
        and IOIs.

        Parameters
        ----------
        iois
            An iterable of inter-onset intervals (IOIs). For instance: ``[500, 500, 400, 200]``.
        end_with_interval
            Indicates whether sequence has an extra final inter-onset interval; this is useful for
            musical/rhythmical sequences.
        name
            Optionally, you can give the Sequence object a name. This is used when printing,
            plotting, or writing the Sequence object. It can always be retrieved and changed via
            :py:attr:`BaseSequence.name`.

        Examples
        --------
        >>> iois = [500, 400, 600, 400]
        >>> seq = Sequence(iois)
        >>> print(seq.onsets)
        [   0.  500.  900. 1500. 1900.]

        """

        # Call super init method
        super().__init__(
            iois=iois,
            first_onset=first_onset,
            end_with_interval=end_with_interval,
            name=name,
        )

    def __add__(self, other):
        if isinstance(other, Sequence):
            if not self.end_with_interval:
                raise ValueError(
                    "When concatenating sequences the sequence on the left-hand side must end with an interval."
                )
            return Sequence(
                iois=np.concatenate([self.iois, other.iois]),
                first_onset=self._first_onset,
                end_with_interval=other.end_with_interval,
            )
        elif (
            isinstance(other, (int, float, np.integer, np.floating)) and not self.end_with_interval
        ):
            return Sequence(
                iois=np.append(self.iois, other),
                first_onset=self._first_onset,
                end_with_interval=True,
                name=self.name,
            )
        elif isinstance(other, (int, float, np.integer, np.floating)) and self.end_with_interval:
            if not other > 0:
                raise ValueError("Cannot add a negative- or zero-length silence to a sequence.")
            iois = self.iois
            iois[-1] += other
            return Sequence(
                iois=iois,
                first_onset=self._first_onset,
                end_with_interval=True,
                name=self.name,
            )
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float, np.integer, np.floating)):
            return Sequence(
                iois=self.iois,
                first_onset=self._first_onset + other,
                end_with_interval=self.end_with_interval,
                name=self.name,
            )
        return NotImplemented

    def __mul__(self, other: int):
        return self._repeat(times=other)

    def __str__(self):
        name = self.name if self.name else "Not provided"
        end_with_intervality = (
            "(ends with interval)" if self.end_with_interval else "(ends with event)"
        )

        return (
            f"Object of type Sequence {end_with_intervality}\n"
            f"{len(self.onsets)} events\n"
            f"IOIs: {self.iois}\n"
            f"Onsets: {self.onsets}\n"
            f"Sequence name: {name}\n"
        )

    def __repr__(self):
        if self.name:
            f"Sequence(name={self.name}, iois={np.array2string(self.iois, threshold=8, precision=2)})"

        return f"Sequence(iois={np.array2string(self.iois, threshold=8, precision=2)})"

    @classmethod
    def from_binary_string(cls, pattern: str, grid_ioi: float, **kwargs) -> Sequence:
        """

        This class method can be used to construct a new :py:class:`Sequence` object on the basis of
        a binary string. The binary string is interpreted as a sequence of 0s and 1s, where 0
        represents silence and 1 represents an event onset. The grid_ioi represents the duration of
        one digit.

        Parameters
        ----------
        string
            The binary string. For instance: ``'10101100'``.
        grid_ioi
            The duration of one digit in the binary string. For instance: ``250``.
        **kwargs
            Additional keyword arguments are passed to the :py:class:`Sequence` constructor (see
            :py:meth:`thebeat.core.Sequence.__init__`).

        Examples
        --------
        >>> seq = Sequence.from_binary_string('10101100', 250)
        >>> print(seq.iois)
        [500. 500. 250. 750.]
        """

        split = re.findall(r"10*", pattern)
        iois = np.array([len(s) for s in split]) * grid_ioi

        return cls(iois=iois, **kwargs)

    @classmethod
    def from_integer_ratios(
        cls, numerators: npt.ArrayLike[float], value_of_one: float, **kwargs
    ) -> Sequence:
        """

        This class method can be used to construct a new :py:class:`Sequence` object on the basis of
        integer ratios.
        See :py:attr:`Sequence.integer_ratios` for explanation.

        Parameters
        ----------
        numerators
            The numerators of the integer ratios. For instance: ``[1, 2, 4]``
        value_of_one
            This represents the duration of the 1, multiples of this value are used.
            For instance, a sequence of ``[2, 4]`` using ``value_of_one=500`` would be a
            :py:class:`Sequence` with IOIs: ``[1000 2000]``.
        **kwargs
            Additional keyword arguments are passed to the :py:class:`Sequence` constructor (see
            :py:meth:`thebeat.core.Sequence.__init__`.


        Examples
        --------
        >>> seq = Sequence.from_integer_ratios(numerators=[1, 2, 4], value_of_one=500)
        >>> print(seq.iois)
        [ 500. 1000. 2000.]
        """

        numerators = np.array(numerators)
        return cls(numerators * value_of_one, **kwargs)

    @classmethod
    def from_onsets(cls, onsets: np.ndarray[float] | list[float], **kwargs) -> Sequence:
        """
        Class method that can be used to generate a new :py:class:`Sequence` object on the basis of
        event onsets. Here, the onsets do not have to start with zero.

        Parameters
        ----------
        onsets
            An array or list containg event onsets, for instance: ``[0, 500, 1000]``.
            The onsets do not have to start with zero.
        **kwargs
            Additional keyword arguments are passed to the :py:class:`Sequence` constructor
            (excluding ``first_onset`` and ``end_with_interval``, which are set by this method).


        Examples
        --------
        >>> seq = Sequence.from_onsets([0, 500, 1000])
        >>> print(seq.iois)
        [500. 500.]
        """

        iois = np.diff(onsets)

        return cls(iois, first_onset=onsets[0], end_with_interval=False, **kwargs)

    @classmethod
    def from_txt(
        cls,
        filepath: str | pathlib.Path,
        type: str = "iois",
        end_with_interval: bool = False,
        **kwargs,
    ) -> Sequence:
        """
        Class method that can be used to generate a new :py:class:`Sequence` object from a text
        file.

        The text file is assumed to contain one IOI/onset per line.

        Parameters
        ----------
        filepath
            The path to the text file.
        type
            The type of the sequence. Can be either ``iois`` or ``onsets``.
        end_with_interval
            Indicates whether the sequence should end with an event (``False``) or an interval (``True``).
        **kwargs
            Additional keyword arguments are passed to the :py:class:`Sequence` constructor.

        """
        with open(filepath) as f:
            data = f.readlines()

        if type == "iois":
            return cls(
                iois=np.array(data, dtype=np.float64),
                end_with_interval=end_with_interval,
                **kwargs,
            )
        elif type == "onsets":
            return cls.from_onsets(onsets=np.array(data, dtype=np.float64), **kwargs)
        else:
            raise ValueError("type can only be 'iois' or 'onsets'.")

    @classmethod
    def generate_isochronous(
        cls, n_events: int, ioi: float, end_with_interval: bool = False, **kwargs
    ) -> Sequence:
        """
        Class method that generates a sequence of isochronous (i.e. equidistant) inter-onset
        intervals. Note that there will be *n*-1 IOIs in a sequence. IOIs are rounded off to
        integers.

        Parameters
        ----------
        n_events
            The desired number of events in the sequence.
        ioi
            The inter-onset interval to be used between all events.
        end_with_interval
            Indicates whether the sequence should end with an event (``False``) or an interval
            (``True``).
        **kwargs
            Additional keyword arguments are passed to the :py:class:`Sequence` constructor.


        Examples
        --------
        >>> seq = Sequence.generate_isochronous(n_events=5,ioi=500)
        >>> print(seq.iois)
        [500. 500. 500. 500.]
        >>> print(len(seq.onsets))
        5
        >>> print(len(seq.iois))
        4

        >>> seq = Sequence.generate_isochronous(n_events=5,ioi=500,end_with_interval=True)
        >>> print(len(seq.onsets))
        5
        >>> print(len(seq.iois))
        5

        """

        # Number of IOIs depends on end_with_interval argument
        n_iois = n_events if end_with_interval else n_events - 1

        return cls([ioi] * n_iois, end_with_interval=end_with_interval, **kwargs)

    @classmethod
    def generate_random_normal(
        cls,
        n_events: int,
        mu: float,
        sigma: float,
        rng: np.random.Generator | None = None,
        end_with_interval: bool = False,
        **kwargs,
    ) -> Sequence:
        """
        Class method that generates a :py:class:`Sequence` object with random inter-onset intervals
        (IOIs) based on the normal distribution.

        Parameters
        ----------
        n_events
            The desired number of events in the sequence.
        mu
            The mean of the normal distribution.
        sigma
            The standard deviation of the normal distribution.
        rng
            A :class:`numpy.random.Generator` object. If not supplied
            :func:`numpy.random.default_rng` is used.
        end_with_interval
            Indicates whether the sequence should end with an event (``False``) or an interval
            (``True``).
        **kwargs
            Additional keyword arguments are passed to the :py:class:`Sequence` constructor.

        Examples
        --------
        >>> generator = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_random_normal(n_events=5,mu=500,sigma=50,rng=generator)
        >>> print(seq.iois)
        [450.54393248 481.61066743 564.39626306 509.69872096]

        >>> seq = Sequence.generate_random_normal(n_events=5,mu=500,sigma=50,end_with_interval=True)
        >>> len(seq.onsets) == len(seq.iois)
        True
        """
        if rng is None:
            rng = np.random.default_rng()

        # Number of IOIs depends on end_with_intervality
        n_iois = n_events if end_with_interval else n_events - 1

        return cls(
            rng.normal(loc=mu, scale=sigma, size=n_iois),
            end_with_interval=end_with_interval,
            **kwargs,
        )

    @classmethod
    def generate_random_uniform(
        cls,
        n_events: int,
        a: float,
        b: float,
        rng: np.random.Generator | None = None,
        end_with_interval: bool = False,
        **kwargs,
    ) -> Sequence:
        """
        Class method that generates a :py:class:`Sequence` object with random inter-onset intervals
        (IOIs) based on a uniform distribution.

        Parameters
        ----------
        n_events
            The desired number of events in the sequence.
        a
            The left bound of the uniform distribution.
        b
            The right bound of the uniform distribution.
        rng
            A :class:`numpy.random.Generator` object. If not supplied
            :func:`numpy.random.default_rng` is used.
        end_with_interval
            Indicates whether the sequence should end with an event (``False``) or an interval
            (``True``).
        **kwargs
            Additional keyword arguments are passed to the :py:class:`Sequence` constructor.


        Examples
        --------
        >>> generator = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_random_uniform(n_events=5,a=400,b=600,rng=generator)
        >>> print(seq.iois)
        [536.47037265 410.76420376 444.07197455 436.87436214]

        >>> seq = Sequence.generate_random_uniform(n_events=5,a=400,b=600,end_with_interval=True)
        >>> len(seq.onsets) == len(seq.iois)
        True
        """

        if rng is None:
            rng = np.random.default_rng()

        # Number of IOIs depends on end_with_interval argument
        n_iois = n_events if end_with_interval else n_events - 1

        iois = rng.uniform(low=a, high=b, size=n_iois)
        return cls(iois, end_with_interval=end_with_interval, **kwargs)

    @classmethod
    def generate_random_poisson(
        cls,
        n_events: int,
        lam: float,
        rng: np.random.Generator | None = None,
        end_with_interval: bool = False,
        **kwargs,
    ) -> Sequence:
        """
        Class method that generates a :py:class:`Sequence` object with random inter-onset intervals
        (IOIs) based on a Poisson distribution.

        Parameters
        ----------
        n_events
            The desired number of events in the sequence.
        lam
            The desired value for lambda.
        rng
            A :class:`numpy.random.Generator` object. If not supplied
            :func:`numpy.random.default_rng` is used.
        end_with_interval
            Indicates whether the sequence should end with an event (``False``) or an interval
            (``True``).
        **kwargs
            Additional keyword arguments are passed to the :py:class:`Sequence` constructor.


        Examples
        --------
        >>> generator = np.random.default_rng(123)
        >>> seq = Sequence.generate_random_poisson(n_events=5,lam=500,rng=generator)
        >>> print(seq.iois)
        [512. 480. 476. 539.]

        """
        if rng is None:
            rng = np.random.default_rng()

        # Number of IOIs depends on end_with_interval argument
        n_iois = n_events if end_with_interval else n_events - 1

        return cls(
            rng.poisson(lam=lam, size=n_iois),
            end_with_interval=end_with_interval,
            **kwargs,
        )

    @classmethod
    def generate_random_exponential(
        cls,
        n_events: int,
        lam: float,
        rng: np.random.Generator | None = None,
        end_with_interval: bool = False,
        **kwargs,
    ) -> Sequence:
        """Class method that generates a :py:class:`Sequence` object with random inter-onset
        intervals (IOIs) based on an exponential distribution.

        Parameters
        ----------
        n_events
            The desired number of events in the sequence.
        lam
           The desired value for lambda.
        rng
            A :class:`numpy.random.Generator` object. If not supplied NumPy's
            :func:`numpy.random.default_rng` is used.
        end_with_interval
            Indicates whether the sequence should end with an event (``False``) or an interval
            (``True``).
        **kwargs
            Additional keyword arguments are passed to the :py:class:`Sequence` constructor.


        Examples
        --------
        >>> generator = np.random.default_rng(seed=123)
        >>> seq = Sequence.generate_random_exponential(n_events=5,lam=500,rng=generator)
        >>> print(seq.iois)
        [298.48624756  58.51553052 125.89734975 153.98272273]

        """
        if rng is None:
            rng = np.random.default_rng()

        n_iois = n_events if end_with_interval else n_events - 1

        return cls(
            rng.exponential(scale=lam, size=n_iois),
            end_with_interval=end_with_interval,
            **kwargs,
        )

    def merge(self, other: thebeat.core.Sequence | list[thebeat.core.Sequence]):
        """
        Merge this :py:class:`Sequence` object with one or multiple other :py:class:`Sequence`
        objects.

        Returns a new :py:class:`Sequence` object.


        Parameters
        ----------
        other
            A :py:class:`Sequence` object, or a list of :py:class:`Sequence` objects.

        Returns
        -------
        object
            A :py:class:`Sequence` object.

        """
        if isinstance(other, thebeat.Sequence):
            return thebeat.utils.merge_sequences([self, other])

        return thebeat.utils.merge_sequences([self, *other])

    # Manipulation methods
    def add_noise_gaussian(self, noise_sd: float, rng: np.random.Generator | None = None):
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
        >>> seq = Sequence.generate_isochronous(n_events=5,ioi=500)
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

    def change_tempo(self, factor: float) -> None:
        """Change the tempo of the `Sequence` object, where a factor of 1 or bigger increases the
        tempo (resulting in smaller inter-onset intervals). A factor between 0 and 1 decreases the
        tempo (resulting in larger inter-onset intervals).

        Parameters
        ----------
        factor
            Tempo change factor. E.g. 2 means twice as fast. 0.5 means twice as slow.

        Examples
        --------
        >>> seq = Sequence.generate_isochronous(n_events=5, ioi=500)
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

    def change_tempo_linearly(self, total_change: float):
        """Create a ritardando or accelerando effect in the inter-onset intervals (IOIs).
        It divides the IOIs by a vector linearly spaced between 1 and ``total_change``.

        Parameters
        ----------
        total_change
            Total tempo change at the end of the :py:class:`Sequence` compared to the beginning.
            So, a total change of 2 (accelerando) results in a final IOI that is twice as short as
            the first IOI. A total change of 0.5 (ritardando) results in a final IOI that is twice
            as long as the first IOI.

        Examples
        --------
        >>> seq = Sequence.generate_isochronous(n_events=5,ioi=500)
        >>> print(seq.iois)
        [500. 500. 500. 500.]
        >>> seq.change_tempo_linearly(total_change=2)
        >>> print(seq.iois)
        [500. 375. 300. 250.]
        """

        self.iois /= np.linspace(start=1, stop=total_change, num=len(self.iois))

    def round_onsets(self, decimals: int = 0):
        """Use this function to round off the :py:class:`Sequence` object's onsets
        (i.e. *t* values). This can, for instance, be useful to get rid of warnings that are the
        result of frame rounding. See e.g. :py:class:`SoundSequence`.

        Note that this function does not return anything. The onsets of the sequence object from
        which this method is called are rounded.

        Parameters
        ----------
        decimals
            The number of decimals desired.

        """

        self.onsets = np.round(self.onsets, decimals=decimals)

    def quantize_iois(self, to: float, inplace: bool = False):
        """Quantize the Sequence object's IOIs to be multiples of ``to``.


        Parameters
        ----------
        to
            The value to be quantized to. E.g. a value of ``100`` means that the IOIs will be
            quantized to the nearest multiple of 100.
        inplace
            Whether to change the IOIs of the Sequence object itself (True), or to return a new Sequence object (False).

        Examples
        --------
        >>> seq = Sequence(iois=[235, 510, 420, 99])
        >>> print(seq.iois)
        [235. 510. 420.  99.]
        >>> seq.quantize_iois(to=100, inplace=True)
        >>> print(seq.iois)
        [200. 500. 400. 100.]

        """

        round_iois = np.round(self.iois / to) * to

        if inplace is True:
            self.iois = round_iois
        else:
            return thebeat.Sequence(
                iois=round_iois,
                first_onset=self._first_onset,
                end_with_interval=self.end_with_interval,
                name=self.name,
            )

    # Visualization
    def plot_sequence(
        self, linewidth: npt.ArrayLike[float] | float | None = None, **kwargs
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the :py:class:`Sequence` object as an event plot on the basis of the event onsets.

        The lines' left boundaries are placed at the event onsets.

        Parameters
        ----------
        linewidth
            The desired width of the bars (events). Defaults to 1/10th of the smallest inter-onset
            interval (IOI). Can be a single value that will be used for each onset, or a list or
            array of values (i.e with a value for each respective onsets).
        **kwargs
            Additional parameters (e.g. 'title', 'dpi' etc.) are passed to
            :py:func:`thebeat.helpers.plot_single_sequence`.

        Examples
        --------
        >>> seq = Sequence.generate_isochronous(n_events=5,ioi=500)
        >>> seq.plot_sequence()  # doctest: +SKIP

        In this example, we plot onto an existing :class:`~matplotlib.pyplot.Axes` object.

        >>> import matplotlib.pyplot as plt
        >>> seq = Sequence([500, 200, 1000])
        >>> fig, axs = plt.subplots(nrows=1, ncols=2)
        >>> seq.plot_sequence(ax=axs[0])  # doctest: +SKIP

        """

        # For the title, use the Sequence name if it has one. Otherwise use the title parameter,
        # which may be None.
        if self.name and kwargs.get("title") is None:
            kwargs.get("title", self.name)

        # Linewidths
        if linewidth is None:
            linewidths = np.repeat(np.min(self.iois) / 10, len(self.onsets))
        elif isinstance(linewidth, numbers.Real):
            linewidths = np.repeat(linewidth, len(self.onsets))
        else:
            linewidths = np.array(linewidth)

        # If the sequence is end_with_interval we also want to plot the final ioi
        final_ioi = self.iois[-1] if self.end_with_interval else None

        # Plot the sequence
        fig, ax = thebeat.helpers.plot_single_sequence(
            onsets=self.onsets,
            end_with_interval=self.end_with_interval,
            final_ioi=final_ioi,
            linewidths=linewidths,
            **kwargs,
        )

        return fig, ax

    @property
    def integer_ratios(self) -> np.ndarray:
        r"""Calculate how to describe a sequence of IOIs in integer ratio numerators from
        the total duration of the sequence by finding the least common multiplier.

        Example
        -------
        A sequence of IOIs ``[250, 500, 1000, 250]`` has a total duration of 2000.
        This can be described using the least common multiplier as
        :math:`\frac{1}{8}, \frac{2}{8}, \frac{4}{8}, \frac{1}{8}`,
        so this method returns the numerators ``[1, 2, 4, 1]``.

        Notes
        -----
        The method for calculating the integer ratios is based on
        :cite:t:`jacobyIntegerRatioPriors2017`.

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
        The used method is based on the methodology from :cite:t:`roeskeCategoricalRhythmsAre2020`.

        Examples
        --------
        >>> seq = Sequence.from_integer_ratios([2, 2, 1, 1], value_of_one=500)
        >>> print(seq.iois)
        [1000. 1000.  500.  500.]
        >>> print(seq.interval_ratios_from_dyads)
        [0.5        0.66666667 0.5       ]

        """

        iois = self.iois
        return thebeat.utils.get_interval_ratios_from_dyads(iois)

    def _repeat(self, times: int) -> Sequence:
        """
        Repeat the inter-onset intervals (IOIs) ``times`` times. Returns a new Sequence instance.
        Only works for Sequences that end with an interval! Otherwise, we do not know what the IOI
        is between the offset of the final event of the original sequence, and the onset of the
        first sound in the repeated sequence.

        Parameters
        ----------
        times
            The number of times the inter-onset intervals should be repeated.

        """
        if not isinstance(times, int):
            raise TypeError("You can only multiply Sequence objects by integers.")

        if not self.end_with_interval or not self.onsets[0] == 0:
            raise ValueError(
                "You can only repeat Sequences that end with an interval that additionally have first_onset == 0.0. "
                "Try adding the end_with_interval=True flag when creating this object."
            )

        new_iois = np.tile(self.iois, reps=times)

        return Sequence(iois=new_iois, first_onset=0.0, end_with_interval=True, name=self.name)
