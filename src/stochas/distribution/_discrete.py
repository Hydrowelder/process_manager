"""Discrete and non-numeric probability distributions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
from numpydantic import NDArray
from pydantic import Field, model_validator

from stochas.distribution._base import (
    UNDEFINED,
    DiscreteDistribution,
    DistType,
    SerializableUndefined,
    logger,
)

if TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import (  # pyright: ignore[reportMissingModuleSource]
        rv_discrete_frozen,
    )
    from scipy.stats.distributions import rv_discrete
else:
    rv_discrete_frozen = Any
    rv_discrete = Any


class DiscreteUniformDistribution(DiscreteDistribution[int]):
    """
    Represent a distribution where every integer in a range is equally likely.

    Use this for discrete counts like number of objects, gear teeth, or array indices where you need equal probability across the whole range.

    Examples:
        ```python
        dist = DiscreteUniformDistribution(name=name, low=-1, high=2)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/2.1/reference/random/generated/numpy.random.Generator.integers.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.randint.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/discrete_uniform.png" width="600" />

    """

    low: int
    high: int

    dist_type: Literal[DistType.DISCRETE_UNIFORM] = DistType.DISCRETE_UNIFORM

    @model_validator(mode="after")
    def validate_low_high(self) -> Self:
        if self.high <= self.low:
            msg = f"Distribution {self.name}: high must be greater than low"
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_discrete_frozen:
        import scipy.stats as stats

        return stats.randint(low=self.low, high=self.high + 1)

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.integers(low=self.low, high=self.high, size=size, endpoint=True)

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Probability Mass Function."""
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high}


class CategoricalDistribution[T](DiscreteDistribution[T]):
    """
    Represent a discrete distribution over a fixed set of named choices.

    Ideal for switching between discrete simulation states, such as different materials, object types, or floor textures.

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
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/categorical.png" width="600" />

    """

    choices: dict[T, float]
    """Choices for the categorical distribution. Tuples have the format (category, probability). This guarantees each category has an associated probability."""

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

    def _build_scipy(self) -> rv_discrete_frozen | rv_discrete:
        import scipy.stats as stats

        indices = np.arange(len(self.choices))
        return stats.rv_discrete(
            name=self.name, values=(indices, self.probabilities), seed=self.rng
        )

    def draw(self, size: int = 1) -> NDArray[Any, T]:
        return self.rng.choice(a=self.categories, size=size, p=self.probabilities)  # type: ignore

    def pmf(self, x: T) -> float:
        """Probability Mass Function. Returns the probability of a specific category 'x'."""
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
            return float(self._scipy.cdf(idx))
        except ValueError:
            return 0.0

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"choices": ", ".join(f"{k}: {v}" for k, v in self.choices.items())}


class PermutationDistribution[T](DiscreteDistribution[T]):
    """
    Takes a master list and returns either the original or a shuffled version.

    Trial 0 (Nominal) returns the items in their provided order.
    Trial > 0 returns a unique random permutation of those items.
    """

    items: list[T]
    """The master list to be shuffled."""

    nominal: list[T] | list[None] | SerializableUndefined = Field(default=UNDEFINED)  # pyright: ignore[reportIncompatibleVariableOverride]

    dist_type: Literal[DistType.PERMUTATION] = DistType.PERMUTATION

    def sample(self, size: int = 1) -> np.ndarray:
        """The core sampling logic for the distribution."""
        if self.is_nominal and self.nominal is not UNDEFINED:
            return np.asarray([self.nominal] * size)
        else:
            return self.draw(size=size)

    def draw(self, size: int = 1) -> np.ndarray:
        """Returns a shuffled version of the items."""
        return np.array([self.rng.permutation(self.items) for _ in range(size)])  # pyright: ignore[reportArgumentType, reportCallIssue]

    def pmf(self, x: list[T] | np.ndarray) -> float:
        """Uniform probability of 1/n! if x is a valid permutation."""
        n = len(self.items)
        if len(x) != n or set(x) != set(self.items):
            return 0.0
        return 1.0 / math.factorial(n)

    def cdf(self, x: Any) -> float:
        raise NotImplementedError("CDF not defined for ShuffledDistribution")

    def ppf(self, q: Any) -> Any:
        raise NotImplementedError("PPF not defined for ShuffledDistribution")

    @property
    def table_params(self) -> dict[str, Any]:
        return {"items": str(self.items)}


class PoissonDistribution(DiscreteDistribution[int]):
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
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/poisson.png" width="600" />

    """

    lam: float
    """Lambda: Average rate of occurrences"""

    dist_type: Literal[DistType.POISSON] = DistType.POISSON

    def _build_scipy(self) -> rv_discrete_frozen:
        import scipy.stats as stats

        return stats.poisson(mu=self.lam)

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.poisson(lam=self.lam, size=size)

    def pmf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"lam": self.lam}


class BernoulliDistribution(DiscreteDistribution[bool]):
    """
    Represent a single binary trial (Success/Failure).

    The simplest distribution, used for "on/off" flags, such as enabling or disabling a specific sensor or randomized model feature.

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
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/bernoulli.png" width="600" />

    """

    p: float
    """Probability of success (0.0 to 1.0)."""

    dist_type: Literal[DistType.BERNOULLI] = DistType.BERNOULLI

    @model_validator(mode="after")
    def validate_probability(self) -> Self:
        if not (0 <= self.p <= 1):
            msg = f"Probability p must be between 0 and 1. Got {self.p}"
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_discrete_frozen:
        import scipy.stats as stats

        return stats.bernoulli(self.p)

    def draw(self, size: int = 1) -> np.ndarray:
        # numpy doesn't have a 'bernoulli', so we use binomial with n=1
        return self.rng.binomial(n=1, p=self.p, size=size)

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"p": self.p}


class BinomialDistribution(DiscreteDistribution[int]):
    """
    Represent the number of successes in a fixed number of independent trials.

    Use this when modeling counts with a known upper bound, such as the number of sensors that report an anomaly out of a fixed-size sensor array.

    Examples:
        ```python
        # Number of faulty actuators out of 12 in a trial
        dist = BinomialDistribution(name="faulty_actuators", n=12, p=0.05)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Binomial_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/binomial.png" width="600" />

    """

    n: int
    """Number of trials. Must be at least 1."""

    p: float
    """Probability of success on each trial (0.0 to 1.0)."""

    dist_type: Literal[DistType.BINOMIAL] = DistType.BINOMIAL

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.n < 1:
            msg = f"Distribution {self.name}: n must be at least 1."
            logger.error(msg)
            raise ValueError(msg)
        if not (0 <= self.p <= 1):
            msg = f"Distribution {self.name}: p must be between 0 and 1. Got {self.p}"
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_discrete_frozen:
        import scipy.stats as stats

        return stats.binom(n=self.n, p=self.p)

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.binomial(n=self.n, p=self.p, size=size)

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Probability Mass Function."""
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"n": self.n, "p": self.p}


class NegativeBinomialDistribution(DiscreteDistribution[int]):
    """
    Represent the number of failures before achieving a fixed number of successes.

    Use this when modeling "how many attempts fail before r successes occur", such as the number of test runs that fail before a robot achieves r consecutive successful grasps.

    Examples:
        ```python
        # Number of failed grasps before 3 successful ones (85% success rate)
        dist = NegativeBinomialDistribution(name="grasp_failures", r=3, p=0.85)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.negative_binomial.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Negative_binomial_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/negative_binomial.png" width="600" />

    """

    r: int
    """Number of successes required. Must be at least 1."""

    p: float
    """Probability of success on each trial (0 < p <= 1)."""

    dist_type: Literal[DistType.NEGATIVE_BINOMIAL] = DistType.NEGATIVE_BINOMIAL

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.r < 1:
            msg = f"Distribution {self.name}: r must be at least 1."
            logger.error(msg)
            raise ValueError(msg)
        if not (0 < self.p <= 1):
            msg = f"Distribution {self.name}: p must be in (0, 1]. Got {self.p}"
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_discrete_frozen:
        import scipy.stats as stats

        return stats.nbinom(n=self.r, p=self.p)

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.negative_binomial(n=self.r, p=self.p, size=size)

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Probability Mass Function."""
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"r": self.r, "p": self.p}


class GeometricDistribution(DiscreteDistribution[int]):
    """
    Represent the number of trials until the first success.

    Use this when modeling "how many attempts before the first success", such as the number of samples taken before a sensor detects an event.

    Examples:
        ```python
        # Number of attempts before first successful detection (70% detection rate)
        dist = GeometricDistribution(name="detection_attempts", p=0.7)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geom.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Geometric_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/geometric.png" width="600" />

    """

    p: float
    """Probability of success on each trial (0 < p <= 1)."""

    dist_type: Literal[DistType.GEOMETRIC] = DistType.GEOMETRIC

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if not (0 < self.p <= 1):
            msg = f"Distribution {self.name}: p must be in (0, 1]. Got {self.p}"
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_discrete_frozen:
        import scipy.stats as stats

        return stats.geom(p=self.p)

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.geometric(p=self.p, size=size)

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Probability Mass Function."""
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"p": self.p}


class HypergeometricDistribution(DiscreteDistribution[int]):
    """
    Represent the number of successes when drawing without replacement from a finite population.

    Unlike the Binomial distribution, the Hypergeometric accounts for the fact that each draw changes the composition of the remaining population. Use this when modeling selection from a fixed pool, such as how many of M sampled objects from a bin of N total are defective (K of which are known to be defective).

    Examples:
        ```python
        # Drawing 10 items from a batch of 50 where 8 are defective
        dist = HypergeometricDistribution(name="defective_count", N=50, M=10, K=8)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.hypergeometric.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Hypergeometric_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/hypergeometric.png" width="600" />

    """

    N: int
    """Total population size. Must be at least 1."""

    M: int
    """Number of draws. Must satisfy 1 <= M <= N."""

    K: int
    """Number of success states in the population. Must satisfy 0 <= K <= N."""

    dist_type: Literal[DistType.HYPERGEOMETRIC] = DistType.HYPERGEOMETRIC

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.N < 1:
            msg = f"Distribution {self.name}: N (population size) must be at least 1."
            logger.error(msg)
            raise ValueError(msg)
        if not (1 <= self.M <= self.N):
            msg = f"Distribution {self.name}: M (draws) must satisfy 1 <= M <= N."
            logger.error(msg)
            raise ValueError(msg)
        if not (0 <= self.K <= self.N):
            msg = f"Distribution {self.name}: K (successes in population) must satisfy 0 <= K <= N."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_discrete_frozen:
        import scipy.stats as stats

        # scipy hypergeom: M=population, n=successes in pop, N=draws
        return stats.hypergeom(M=self.N, n=self.K, N=self.M)

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.hypergeometric(
            ngood=self.K, nbad=self.N - self.K, nsample=self.M, size=size
        )

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Probability Mass Function."""
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"N": self.N, "M": self.M, "K": self.K}


class BetaBinomialDistribution(DiscreteDistribution[int]):
    """
    Represent an overdispersed count distribution where the success probability is itself Beta-distributed.

    The Beta-Binomial is a compound distribution: each trial has a different success probability drawn from a Beta(alpha, beta) distribution. Use this when modeling counts that are more variable than a plain Binomial would predict, such as the number of successful grasps when grip success rate varies across trials.

    Examples:
        ```python
        # Number of successful grasps out of 10 attempts with variable success rate
        dist = BetaBinomialDistribution(name="grasp_successes", n=10, alpha=8.0, beta=2.0)
        ```

    References:
        1. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html)
        2. [Wikipedia](https://en.wikipedia.org/wiki/Beta-binomial_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/beta_binomial.png" width="600" />

    """

    n: int
    """Number of trials. Must be at least 1."""

    alpha: float
    """First shape parameter of the Beta prior. Must be positive."""

    beta: float
    """Second shape parameter of the Beta prior. Must be positive."""

    dist_type: Literal[DistType.BETA_BINOMIAL] = DistType.BETA_BINOMIAL

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.n < 1:
            msg = f"Distribution {self.name}: n must be at least 1."
            logger.error(msg)
            raise ValueError(msg)
        if self.alpha <= 0:
            msg = f"Distribution {self.name}: alpha must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        if self.beta <= 0:
            msg = f"Distribution {self.name}: beta must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_discrete_frozen:
        import scipy.stats as stats

        return stats.betabinom(n=self.n, a=self.alpha, b=self.beta)

    def draw(self, size: int = 1) -> np.ndarray:
        return np.asarray(self._scipy.rvs(size=size, random_state=self.rng))

    def pmf(self, x: int | np.ndarray) -> float | np.ndarray:
        """Probability Mass Function."""
        return self._scipy.pmf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"n": self.n, "alpha": self.alpha, "beta": self.beta}
