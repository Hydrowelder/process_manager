from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import scipy.stats as stats
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    model_validator,
)

from process_manager.data_handlers.named_value import NamedValue
from process_manager.data_handlers.named_value_collections import NamedValueDict

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
    "ExponentialDistribution",
    "LogNormalDistribution",
    "NormalDistribution",
    "PoissonDistribution",
    "TriangularDistribution",
    "TruncatedNormalDistribution",
    "UniformDistribution",
]


class Distribution(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """Name of the distribution."""

    seed: int | None = None
    """Seed of the distribution.

    Leave as None or omit to use a random seed. If the seed is not None, the specified seed will be salted with the name attribute to add randomness. This allows you to use the same seed for multiple distributions while also being able to simply serialize and deserialize the distribution. See the `validate_seed` validator method for how the seed is hashed.
    """

    _rng: np.random.Generator = PrivateAttr()

    @model_validator(mode="after")
    def validate_seed(self) -> Self:
        if self.seed is not None:
            salt = int(hashlib.md5(self.name.encode()).hexdigest(), 16)
            local_seed = (self.seed + salt) % (2**32)
            self._rng = np.random.default_rng(seed=local_seed)
        else:
            self._rng = np.random.default_rng()

        return self

    @property
    def rng(self) -> np.random.Generator:
        """Provides a localized random number generator."""
        return self._rng

    @abstractmethod
    def sample(self, size: int = 1) -> np.ndarray:
        """The core sampling logic for the distribution."""
        msg = f"This method has not been implemented for {self.__class__.__name__}"
        logger.error(msg)
        raise NotImplementedError(msg)

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

    def register_to_dict(
        self, dict: NamedValueDict, size: int = 1
    ) -> NamedValue[np.ndarray]:
        """Samples from the distribution and registers the result."""
        samples = self.sample(size=size)
        nv = NamedValue[np.ndarray](name=self.name, stored_value=samples)
        dict.update(nv)
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


class NormalDistribution(Distribution):
    mu: float
    """Mean value of distribution."""

    sigma: float
    """Standard deviation of distribution. Must be positive."""

    _scipy: rv_continuous = PrivateAttr()

    @model_validator(mode="after")
    def validate_sigma(self) -> Self:
        if self.sigma <= 0:
            msg = f"Distribution {self.name} has a negative standard deviation. It must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def sample(self, size: int = 1) -> np.ndarray:
        return self.rng.normal(loc=self.mu, scale=self.sigma, size=size)

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.norm(loc=self.mu, scale=self.sigma)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x=x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x=x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class UniformDistribution(Distribution):
    low: float
    """Minimum value of distribution."""

    high: float
    """Maximum value of distribution."""

    _scipy: rv_continuous = PrivateAttr()

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

    def sample(self, size: int = 1) -> np.ndarray:
        return self.rng.uniform(low=self.low, high=self.high, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x=x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x=x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class CategoricalDistribution(Distribution):
    choices: Sequence[tuple[Any, float]]
    """Choices for the categorical distribution. Tuples have the format (category, probability). This guarantees each category has an associated probability."""

    _scipy: rv_discrete = PrivateAttr()

    @property
    def categories(self) -> list[Any]:
        return [t[0] for t in self.choices]

    @property
    def probabilities(self) -> list[float]:
        return [t[1] for t in self.choices]

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

    def sample(self, size: int = 1) -> np.ndarray:
        return self.rng.choice(a=self.categories, size=size, p=self.probabilities)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        """Categorical distributions have no PDF. Did you mean to use pmf?"""
        logger.warning(
            "Discrete distributions use pmf, not pdf. Using pmf method instead."
        )
        return self.pmf(x=x)

    def pmf(self, x: Any) -> float:
        """
        Probability Mass Function. Returns the probability of a specific category 'x'.
        """
        try:
            idx = self.categories.index(x)
            return self._scipy.pmf(idx)
        except ValueError:
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


class TriangularDistribution(Distribution):
    low: float
    """Minimum value of distribution."""

    mode: float
    """Peak value of distribution."""

    high: float
    """Maximum value of distribution."""

    _scipy: rv_continuous = PrivateAttr()

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

    def sample(self, size: int = 1) -> np.ndarray:
        return self.rng.triangular(
            left=self.low, mode=self.mode, right=self.high, size=size
        )

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class TruncatedNormalDistribution(Distribution):
    mu: float
    """Mean value of distribution."""

    sigma: float
    """Standard deviation of distribution."""

    lower: float = float("-inf")
    """Lower bound of distribution."""

    upper: float = float("inf")
    """Upper bound of distribution."""

    _scipy: rv_continuous = PrivateAttr()

    @model_validator(mode="after")
    def validate_and_setup(self) -> Self:
        # a and b are the number of standard deviations away from the mean
        a = (self.lower - self.mu) / self.sigma
        b = (self.upper - self.mu) / self.sigma
        self._scipy = stats.truncnorm(a=a, b=b, loc=self.mu, scale=self.sigma)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def sample(self, size: int = 1) -> np.ndarray:
        # numpy doesn't have a truncnorm generator
        return self._scipy.rvs(size=size, random_state=self.rng)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class LogNormalDistribution(Distribution):
    s: float
    """The shape parameter (sigma of the log)"""

    scale: float = 1.0
    """exp(mu)"""

    _scipy: rv_continuous = PrivateAttr()

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.lognorm(s=self.s, scale=self.scale)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def sample(self, size: int = 1) -> np.ndarray:
        return self.rng.lognormal(mean=np.log(self.scale), sigma=self.s, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class PoissonDistribution(Distribution):
    lam: float
    """Lambda: Average rate of occurrences"""

    _scipy: rv_discrete = PrivateAttr()

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.poisson(mu=self.lam)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def sample(self, size: int = 1) -> np.ndarray:
        return self.rng.poisson(lam=self.lam, size=size)

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


class ExponentialDistribution(Distribution):
    lam: float
    """Rate parameter (lambda)."""

    _scipy: rv_continuous = PrivateAttr()

    @model_validator(mode="after")
    def validate_scipy(self) -> Self:
        self._scipy = stats.expon(scale=1 / self.lam)  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def sample(self, size: int = 1) -> np.ndarray:
        return self.rng.exponential(scale=1 / self.lam, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


class BernoulliDistribution(Distribution):
    p: float
    """Probability of success (0.0 to 1.0)."""

    _scipy: rv_discrete = PrivateAttr()

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

    def sample(self, size: int = 1) -> np.ndarray:
        # np.random doesn't have a 'bernoulli', so we use binomial with n=1
        return self.rng.binomial(n=1, p=self.p, size=size)

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)


if __name__ == "__main__":
    # 1. Define distributions (Serialization ready!)
    normal_dist = NormalDistribution(name="hight", mu=170, sigma=10, seed=42)
    uniform_dist = UniformDistribution(name="weight", low=60, high=90, seed=42)

    cat_dist = CategoricalDistribution(
        name="blood_type",
        choices=[
            ("O+", 0.36),
            ("O-", 0.14),
            ("A+", 0.28),
            ("A-", 0.08),
            ("B+", 0.08),
            ("B-", 0.03),
            ("AB+", 0.02),
            ("AB-", 0.01),
        ],
        seed=42,
    )
    print(cat_dist.choices)

    identical_normal_dist = NormalDistribution(
        name="height_copy", mu=170, sigma=10, seed=42
    )

    # 2. Create the registry
    named_value_dict = NamedValueDict()

    # 3. Sample and Register
    # These return NamedValue[np.ndarray] objects
    height = normal_dist.register_to_dict(named_value_dict, size=5)
    weight = uniform_dist.register_to_dict(named_value_dict, size=5)
    blood_type = cat_dist.register_to_dict(named_value_dict, size=5)

    # 4. Access values via the NamedValue reference OR the hash
    print(f"Heights: {height.value}")
    print(f"Weights: {named_value_dict.get_value('weight')}")
    print(f"Blood Types: {blood_type.value}")

    # 5. Serialization Check
    # This captures the parameters of the simulation
    print(normal_dist.model_dump_json(indent=2))

    # 6. Check that child_seeds works
    identical_height = identical_normal_dist.register_to_dict(named_value_dict, size=5)

    print(f"{normal_dist.pdf(x=np.array([np.linspace(150, 190, 5)]))=}")
    print(f"{normal_dist.cdf(x=np.array([np.linspace(150, 190, 5)]))=}")

    print(f"{uniform_dist.pdf(x=np.array([np.linspace(50, 100, 5)]))=}")
    print(f"{uniform_dist.cdf(x=np.array([np.linspace(50, 100, 5)]))=}")

    print(f"{cat_dist.pmf(x="O+")=}")

    breakpoint()
