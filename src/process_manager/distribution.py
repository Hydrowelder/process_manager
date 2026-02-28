from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Literal, NewType, Self

import numpy as np
import scipy.stats as stats
from numpydantic import NDArray
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    PrivateAttr,
    model_validator,
)

from process_manager.base_collections import BaseDict, BaseList
from process_manager.named_value import NamedValue, NamedValueDict, ValueName

if TYPE_CHECKING:
    from scipy.stats.distributions import rv_continuous, rv_discrete, rv_frozen
else:
    rv_continuous = Any
    rv_discrete = Any
    rv_frozen = Any

logger = logging.getLogger(__name__)

__all__ = [
    "BernoulliDistribution",
    "CategoricalDistribution",
    "Dist",
    "DistName",
    "DistType",
    "DistributionDict",
    "DistributionList",
    "ExponentialDistribution",
    "LogNormalDistribution",
    "NormalDistribution",
    "PoissonDistribution",
    "TriangularDistribution",
    "TruncatedNormalDistribution",
    "UniformDistribution",
]

DistName = NewType("DistName", str)
"""Alias of string. Used to type hint a distribution."""

NOMINAL_TRIAL_NUM = 0
"""Trial number definition where nominal case will be used."""


class DistType(StrEnum):
    NORMAL = "normal"
    """Normal distribution."""

    UNIFORM = "uniform"
    """Uniform distribution."""

    CATEGORICAL = "categorical"
    """Categorical distribution."""

    TRIANGULAR = "triangular"
    """Triangular distribution."""

    TRUNCATED_NORMAL = "truncated_normal"
    """Truncated Normal distribution."""

    LOG_NORMAL = "log_normal"
    """Log Normal distribution."""

    POISSON = "poisson"
    """Poisson distribution."""

    EXPONENTIAL = "exponential"
    """Exponential distribution."""

    BERNOULLI = "bernoulli"
    """Bernoulli distribution."""


class Undefined:
    """Undefined value."""

    def __bool__(self):
        return False

    def __repr__(self):
        return "UNDEFINED"


UNDEFINED = Undefined()
"""Sentinel to differentiate between None and unset."""


def validate_undefined(v: Any) -> Any:
    if v == "__UNDEFINED__":
        return UNDEFINED
    return v


SerializableUndefined = Annotated[
    Undefined,
    BeforeValidator(validate_undefined),
    PlainSerializer(lambda _: "__UNDEFINED__", return_type=str),
]


class Distribution[T](BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: DistName
    """Name of the distribution."""

    seed: int | None = None
    """Seed of the distribution.

    Leave as None or omit to use a random seed. If the seed is not None, the specified seed will be salted with the name and trial_num attributes to add randomness. This allows you to use the same seed for multiple distributions while also being able to simply serialize and deserialize the distribution. See the `validate_seed` validator method for how the seed is hashed.

    Note:
        | Salt        | Functional Purpose       | Description                                                                                                                                                                           |
        |:------------|:-------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
        | `seed`      | **Campaign Entropy**     | Controls the broad "global" cycle. Change this to generate a completely new set of results for the same analysis (e.g., using a date-based seed for a daily batch).                   |
        | `name`      | **Parameter Decoupling** | Ensures unique draws for different parameters. Without this, two distributions with the same config (e.g., an `x` and `y` center of gravity) would produce identical, coupled values. |
        | `trial_num` | **Iteration Variance**   | Provides a unique draw for each Monte Carlo trial. This ensures that every individual simulation trial receives a unique value from the dispersion.                                   |

    Clear as mud?
    """

    nominal: T | None | SerializableUndefined = Field(default=UNDEFINED)
    """Value the distribution should take if the trial_number attribute is equal to 0."""

    _trial_num: int = PrivateAttr(default=NOMINAL_TRIAL_NUM)
    """Run number for sampling from the distribution. This is used to salt the seed (if specified)."""

    _rng: np.random.Generator = PrivateAttr()
    """Random number generator."""

    def with_seed(self, seed: int | None) -> Self:
        self.seed = seed
        return self

    def with_trial_num(self, trial_num: int) -> Self:
        self.trial_num = trial_num
        return self

    def refresh_seed(self) -> None:
        if self.seed is not None:
            # combine name and run number to salt
            name_to_salt = f"{self.name}_{self._trial_num}"

            # generate a repeatable salt for the seed, name, and trial_number
            salt = int(hashlib.md5(name_to_salt.encode()).hexdigest(), 16)
            local_seed = (self.seed + salt) % (2**32)

            self._rng = np.random.default_rng(seed=local_seed)
        else:
            # use pure random value
            self._rng = np.random.default_rng()

    @model_validator(mode="after")
    def validate_seed(self) -> Self:
        self.refresh_seed()
        return self

    @property
    def trial_num(self) -> int:
        """Run number for sampling from the distribution. This is used to salt the seed (if specified)."""
        return self._trial_num

    @trial_num.setter
    def trial_num(self, new: int) -> None:
        self._trial_num = new
        self.refresh_seed()

    @property
    def rng(self) -> np.random.Generator:
        """Provides a localized random number generator."""
        return self._rng

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Flag indicating if the distribution is continuous (True) or discrete (False)."""
        msg = f"This method has not been implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    @abstractmethod
    def draw(self, size: int = 1) -> NDArray[Any, T]:
        """Perform a random draw"""
        msg = f"This method has not been implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    def sample(self, size: int = 1) -> NDArray[Any, T]:
        """The core sampling logic for the distribution."""
        if self._trial_num == NOMINAL_TRIAL_NUM and self.nominal is not UNDEFINED:
            return np.full(size, self.nominal)
        else:
            return self.draw(size=size)

    @abstractmethod
    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Probability Density Function."""
        msg = f"This method has not been implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    @abstractmethod
    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Cumulative Distribution Function."""
        msg = f"This method has not been implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    def update_dicts(
        self,
        dist_dict: DistributionDict,
        named_value_dict: NamedValueDict,
        size: int = 1,
        force: bool = False,
        warn: bool = True,
    ) -> NamedValue[NDArray[Any, T]]:
        """Samples from the distribution and registers the result."""
        if self.name in named_value_dict and not force:
            return named_value_dict[self.name]
        elif self.name in named_value_dict and force and warn:
            logger.warning(
                f"NamedValue {self.name} already exists in named_value_list. Overwriting!"
            )

        samples = self.sample(size=size)
        nv = NamedValue(name=ValueName(self.name), stored_value=samples)

        dist_dict.update(self)  # pyright: ignore[reportArgumentType]
        if force:
            named_value_dict.force_update(nv, warn=warn)
        else:
            named_value_dict.update(nv)
        return nv

    @abstractmethod
    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        """
        Percent Point Function (Inverse of CDF). Used to find the value at a specific quantile (e.g., 0.95).

        Args:
            q: Probability (0.0 to 1.0).

        """
        msg = f"Method 'ppf' not implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

    @property
    def has_nominal(self) -> bool:
        return self.nominal is not UNDEFINED

    @property
    def is_nominal(self) -> bool:
        return self.trial_num == NOMINAL_TRIAL_NUM and self.has_nominal


class NormalDistribution(Distribution[float]):
    """
    Represent a standard Gaussian "Bell Curve" distribution.

    The Normal distribution is the most common choice for modeling natural variation in physical properties where the value is likely to be near a mean with symmetric falloff.

    Examples:
        ```python
        # Modeling the mass of a link with 10% standard deviation
        dist = NormalDistribution(name="link_mass", mu=1.5, sigma=0.15)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Normal_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/process_manager/refs/heads/main/docs/assets/distributions/normal.png" width="600" />

    """

    mu: float
    """Mean value of distribution."""

    sigma: float
    """Standard deviation of distribution. Must be positive."""

    _scipy: rv_continuous = PrivateAttr()

    dist_type: Literal[DistType.NORMAL] = DistType.NORMAL

    @model_validator(mode="after")
    def validate_sigma(self) -> Self:
        if self.sigma <= 0:
            msg = f"Distribution {self.name} has a negative standard deviation. It must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def draw(self, size: int = 1):
        return self.rng.normal(loc=self.mu, scale=self.sigma, size=size)

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.norm(loc=self.mu, scale=self.sigma)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    @property
    def is_continuous(self) -> Literal[True]:
        return True

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x=x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x=x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class UniformDistribution(Distribution[float]):
    """
    Represent a distribution where every value in a range is equally likely.

    Use this when you have strict upper and lower bounds but no knowledge of which values are more probable within that window.

    Examples:
        ```python
        # Randomizing a starting joint position between -0.5 and 0.5 radians
        dist = UniformDistribution(name="init_pos", low=-0.5, high=0.5)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.uniform.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/process_manager/refs/heads/main/docs/assets/distributions/uniform.png" width="600" />

    """

    low: float
    """Minimum value of distribution."""

    high: float
    """Maximum value of distribution."""

    _scipy: rv_continuous = PrivateAttr()

    dist_type: Literal[DistType.UNIFORM] = DistType.UNIFORM

    @model_validator(mode="after")
    def validate_low_high(self) -> Self:
        if self.high <= self.low:
            msg = f"Distribution {self.name}: high must be greater than low"
            logger.error(msg)
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.uniform(loc=self.low, scale=self.scale)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    @property
    def scale(self) -> float:
        return self.high - self.low

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.uniform(low=self.low, high=self.high, size=size)

    @property
    def is_continuous(self) -> Literal[True]:
        return True

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x=x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x=x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class CategoricalDistribution[T](Distribution[T]):
    """
    Represent a discrete distribution over a fixed set of named choices.

    Ideal for switching between discrete simulation states, such as different
    materials, object types, or floor textures.

    Examples:
        ```python
        class Material(StrEnum):
            STEEL = "steel"
            WOOD = "wood"

        dist = CategoricalDistribution[Material](
            name="floor_material",
            choices={Material.STEEL: 0.8, Material.WOOD: 0.2}
        )
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_discrete.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Categorical_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/process_manager/refs/heads/main/docs/assets/distributions/categorical.png" width="600" />

    """

    choices: dict[T, float]
    """Choices for the categorical distribution. Tuples have the format (category, probability). This guarantees each category has an associated probability."""

    _scipy: rv_discrete = PrivateAttr()

    dist_type: Literal[DistType.CATEGORICAL] = DistType.CATEGORICAL

    @property
    def categories(self) -> list[T]:
        return list(self.choices.keys())

    @property
    def probabilities(self) -> list[float]:
        return list(self.choices.values())

    @model_validator(mode="after")
    def validate_probabilities(self) -> Self:
        s = sum(self.probabilities)
        if not np.isclose(s, 1, atol=1e-8):
            msg = f"Distribution {self.name} sum of probabilities ({s:.2f}) do not sum to 1"
            logger.error(msg)
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        indices = np.arange(len(self.choices))
        self._scipy = stats.rv_discrete(
            name=self.name, values=(indices, self.probabilities), seed=self.rng
        )
        return self

    def draw(self, size: int = 1) -> NDArray[Any, T]:
        return self.rng.choice(a=self.categories, size=size, p=self.probabilities)  # type: ignore

    @property
    def is_continuous(self) -> Literal[False]:
        return False

    def pdf(self, x: Any) -> float | NDArray[Any, float]:
        """Categorical distributions have no PDF. Did you mean to use pmf?"""
        logger.warning(
            "Discrete distributions use pmf, not pdf. Using pmf method instead."
        )
        return self.pmf(x=x)

    def pmf(self, x: T) -> float:
        """
        Probability Mass Function. Returns the probability of a specific category 'x'.
        """
        try:
            return self.choices[x]
        except KeyError:
            return 0.0

    def cdf(self, x: Any) -> float:
        """
        Cumulative Distribution Function.

        Note:
            This follows the order of the 'choices' list.

        """
        try:
            idx = self.categories.index(x)
            return self._scipy.cdf(idx)
        except ValueError:
            return 0.0

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class TriangularDistribution(Distribution[float]):
    """
    Represent a continuous distribution with a triangular shape.

    Often used as a "simpler" version of the Normal distribution when you know the minimum, maximum, and most likely value (mode).

    Examples:
        ```python
        # Damping where 1.0 is the spec, but it might range from 0.8 to 1.5
        dist = TriangularDistribution(name="joint_damping", low=0.8, high=1.5, mode=1.0)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.triang.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Triangular_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/process_manager/refs/heads/main/docs/assets/distributions/triangular.png" width="600" />

    """

    low: float
    """Minimum value of distribution."""

    mode: float
    """Peak value of distribution."""

    high: float
    """Maximum value of distribution."""

    _scipy: rv_continuous = PrivateAttr()

    dist_type: Literal[DistType.TRIANGULAR] = DistType.TRIANGULAR

    @model_validator(mode="after")
    def validate_logic(self) -> Self:
        if not (self.low <= self.mode <= self.high):
            msg = f"{self.name}: Must satisfy low <= mode <= high"
            logger.error(msg)
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        # Scipy mapping: loc=low, scale=high-low, c=(mode-low)/scale
        rescale = self.high - self.low
        c = (self.mode - self.low) / rescale if rescale != 0 else 0
        self._scipy = stats.triang(c=c, loc=self.low, scale=rescale)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.triangular(
            left=self.low, mode=self.mode, right=self.high, size=size
        )

    @property
    def is_continuous(self) -> Literal[True]:
        return True

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class TruncatedNormalDistribution(Distribution[float]):
    """
    Represent a Normal distribution restricted to a specific interval.

    Useful for physical parameters that follow a bell curve but have hard physical limits (e.g., a mass that cannot be negative).

    Examples:
        ```python
        # Friction coefficient with a lower bound of 0 to prevent anti-friction
        dist = TruncatedNormalDistribution(name="friction", mu=0.5, sigma=0.1, low=0.0)
        ```

    References:
        1. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html)
        2. [Wikipedia](https://en.wikipedia.org/wiki/Truncated_normal_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/process_manager/refs/heads/main/docs/assets/distributions/truncated_normal.png" width="600" />

    """

    mu: float
    """Mean value of distribution."""

    sigma: float
    """Standard deviation of distribution."""

    low: float = float("-inf")
    """Lower bound of distribution."""

    high: float = float("inf")
    """Upper bound of distribution."""

    _scipy: rv_continuous = PrivateAttr()

    dist_type: Literal[DistType.TRUNCATED_NORMAL] = DistType.TRUNCATED_NORMAL

    @model_validator(mode="after")
    def validate_and_setup(self) -> Self:
        # a and b are the number of standard deviations away from the mean
        a = (self.low - self.mu) / self.sigma
        b = (self.high - self.mu) / self.sigma
        self._scipy = stats.truncnorm(a=a, b=b, loc=self.mu, scale=self.sigma)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def draw(self, size: int = 1) -> np.ndarray:
        # numpy doesn't have a truncnorm generator
        return self._scipy.rvs(size=size, random_state=self.rng)

    @property
    def is_continuous(self) -> Literal[True]:
        return True

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class LogNormalDistribution(Distribution[float]):
    """
    Represent a distribution whose logarithm is normally distributed.

    Best for variables that are naturally positive and can have "long-tail" outliers, such as contact forces or durations.

    Examples:
        ```python
        # Modeling a scale factor that is usually 1.0 but can occasionally be 5.0
        dist = LogNormalDistribution(name="stiffness_scale", s=0.5, scale=1.0)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Log-normal_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/process_manager/refs/heads/main/docs/assets/distributions/log_normal.png" width="600" />

    """

    s: float
    """The shape parameter (sigma of the log)"""

    scale: float = 1.0
    """exp(mu)"""

    _scipy: rv_continuous = PrivateAttr()

    dist_type: Literal[DistType.LOG_NORMAL] = DistType.LOG_NORMAL

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.lognorm(s=self.s, scale=self.scale)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.lognormal(mean=np.log(self.scale), sigma=self.s, size=size)

    @property
    def is_continuous(self) -> Literal[True]:
        return True

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class PoissonDistribution(Distribution[int]):
    """
    Represent the probability of a number of events occurring in a fixed interval.

    Useful for counting discrete occurrences, like how many distractor objects should appear in a scene.

    Examples:
        ```python
        # Sampling the number of random noise impulses to apply during a trial
        dist = PoissonDistribution(name="impulse_count", lam=3.5)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Poisson_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/process_manager/refs/heads/main/docs/assets/distributions/poisson.png" width="600" />

    """

    lam: float
    """Lambda: Average rate of occurrences"""

    _scipy: rv_discrete = PrivateAttr()

    dist_type: Literal[DistType.POISSON] = DistType.POISSON

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.poisson(mu=self.lam)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.poisson(lam=self.lam, size=size)

    @property
    def is_continuous(self) -> Literal[False]:
        return False

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        logger.warning(
            "Discrete distributions use pmf, not pdf. Using pmf method instead."
        )
        return self.pmf(k=x)

    def pmf(self, k: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pmf(k)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class ExponentialDistribution(Distribution[float]):
    """
    Represent the time between events in a Poisson process.

    In robotics, this is often used to model time-to-failure or the duration
    between random perturbations.

    Examples:
        ```python
        # Time in seconds between sensor noise "glitches"
        dist = ExponentialDistribution(name="glitch_interval", lam=0.5)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Exponential_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/process_manager/refs/heads/main/docs/assets/distributions/exponential.png" width="600" />

    """

    lam: float
    """Rate parameter (lambda)."""

    _scipy: rv_continuous = PrivateAttr()

    dist_type: Literal[DistType.EXPONENTIAL] = DistType.EXPONENTIAL

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.expon(scale=1 / self.lam)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.exponential(scale=1 / self.lam, size=size)

    @property
    def is_continuous(self) -> Literal[True]:
        return True

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class BernoulliDistribution(Distribution[bool]):
    """
    Represent a single binary trial (Success/Failure).

    The simplest distribution, used for "on/off" flags, such as enabling
    or disabling a specific sensor or randomized model feature.

    Examples:
        ```python
        # 50% chance to enable gravity compensation for this trial
        dist = BernoulliDistribution(name="use_grav_comp", p=0.5)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Bernoulli_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/process_manager/refs/heads/main/docs/assets/distributions/bernoulli.png" width="600" />

    """

    p: float
    """Probability of success (0.0 to 1.0)."""

    _scipy: rv_discrete = PrivateAttr()

    dist_type: Literal[DistType.BERNOULLI] = DistType.BERNOULLI

    @model_validator(mode="after")
    def validate_probability(self) -> Self:
        if not (0 <= self.p <= 1):
            msg = f"Probability p must be between 0 and 1. Got {self.p}"
            logger.error(msg)
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.bernoulli(self.p)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def draw(self, size: int = 1) -> np.ndarray:
        # np.random doesn't have a 'bernoulli', so we use binomial with n=1
        return self.rng.binomial(n=1, p=self.p, size=size)

    @property
    def is_continuous(self) -> Literal[False]:
        return False

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        logger.warning(
            "Discrete distributions use pmf, not pdf. Using pmf method instead."
        )
        return self.pdf(x=x)

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


Dist = Annotated[
    NormalDistribution
    | UniformDistribution
    | CategoricalDistribution
    | TriangularDistribution
    | TruncatedNormalDistribution
    | LogNormalDistribution
    | PoissonDistribution
    | ExponentialDistribution
    | BernoulliDistribution,
    Field(discriminator="dist_type"),
]


class DistributionDict(BaseDict[Dist]):
    """Dictionary specifically for sampled results."""

    @property
    def distribution_list(self) -> DistributionList:
        """Converts the DistributionDict to a DistributionList."""
        return DistributionList(list(self.values()))

    def set_trial_nums(self, trial_num: int) -> None:
        for dist in self.values():
            if dist.trial_num != trial_num:
                dist.trial_num = trial_num


class DistributionList(BaseList[Dist]):
    """List specifically for distributions."""

    @property
    def to_distribution_dict(self) -> DistributionDict:
        """Converts the DistributionList to a DistributionDict."""
        d = DistributionDict()
        d.update_many(self.root)
        return d

    def set_trial_nums(self, trial_num: int) -> None:
        for dist in self:
            if dist.trial_num != trial_num:
                dist.trial_num = trial_num


if __name__ == "__main__":
    from enum import StrEnum

    # 1. Define distributions (Serialization ready!)
    normal_dist = NormalDistribution(name=DistName("height"), mu=170, sigma=10, seed=42)
    uniform_dist = UniformDistribution(
        name=DistName("weight"), low=60, high=90, seed=42
    )

    class Blood(StrEnum):
        O_P = "O+"
        O_N = "O-"
        A_P = "A+"
        A_N = "A-"
        B_P = "B+"
        B_N = "B-"
        AB_P = "AB+"
        AB_N = "AB-"

    cat_dist = CategoricalDistribution[Blood](
        name=DistName("blood_type"),
        choices={
            Blood.O_P: 0.36,
            Blood.O_N: 0.14,
            Blood.A_P: 0.28,
            Blood.A_N: 0.08,
            Blood.B_P: 0.08,
            Blood.B_N: 0.03,
            Blood.AB_P: 0.02,
            Blood.AB_N: 0.01,
        },
        seed=42,
        nominal=Blood.O_P,
    )
    print(cat_dist.choices)

    identical_normal_dist = NormalDistribution(
        name=DistName("height_copy"), mu=170, sigma=10, seed=42
    )

    # 2. Create the registry
    dist_dict = DistributionDict()
    named_value_dict = NamedValueDict()

    # 3. Sample and Register
    # These return NamedValue[np.ndarray] objects
    height = normal_dist.update_dicts(
        dist_dict=dist_dict, named_value_dict=named_value_dict, size=5
    )
    weight = uniform_dist.update_dicts(
        dist_dict=dist_dict, named_value_dict=named_value_dict, size=5
    )
    blood_type = cat_dist.update_dicts(
        dist_dict=dist_dict, named_value_dict=named_value_dict, size=5
    )

    # 4. Access values via the NamedValue reference OR the hash
    print(f"Heights: {height.value}")
    print(f"Weights: {named_value_dict.get_value('weight')}")
    print(f"Blood Types: {blood_type.value}")

    # 5. Serialization Check
    # This captures the parameters of the simulation
    print(normal_dist.model_dump_json(indent=2))

    # 6. Check that child_seeds works
    identical_height = identical_normal_dist.update_dicts(
        dist_dict=dist_dict, named_value_dict=named_value_dict, size=5
    )

    print(f"{normal_dist.pdf(x=np.array([np.linspace(150, 190, 5)]))=}")
    print(f"{normal_dist.cdf(x=np.array([np.linspace(150, 190, 5)]))=}")

    print(f"{uniform_dist.pdf(x=np.array([np.linspace(50, 100, 5)]))=}")
    print(f"{uniform_dist.cdf(x=np.array([np.linspace(50, 100, 5)]))=}")

    print(f"{cat_dist.pmf(x=Blood.O_P)=}")

    breakpoint()
