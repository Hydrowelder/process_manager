"""Continuous probability distributions."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import numpy as np
from pydantic import field_serializer, field_validator, model_validator

from stochas.distribution._base import (
    ContinuousDistribution,
    DistType,
    logger,
)

if TYPE_CHECKING:
    from scipy.stats._distn_infrastructure import (  # pyright: ignore[reportMissingModuleSource]
        rv_continuous_frozen,
    )
else:
    rv_continuous_frozen = Any


class NormalDistribution(ContinuousDistribution[float]):
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
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/normal.png" width="600" />

    """

    mu: float
    """Mean value of distribution."""

    sigma: float
    """Standard deviation of distribution. Must be positive."""

    dist_type: Literal[DistType.NORMAL] = DistType.NORMAL

    @model_validator(mode="after")
    def validate_sigma(self) -> Self:
        if self.sigma <= 0:
            msg = f"Distribution {self.name} has a negative standard deviation. It must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.norm(loc=self.mu, scale=self.sigma))

    def draw(self, size: int = 1):
        return self.rng.normal(loc=self.mu, scale=self.sigma, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x=x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x=x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"mu": self.mu, "sigma": self.sigma}


class UniformDistribution(ContinuousDistribution[float]):
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
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/uniform.png" width="600" />

    """

    low: float
    """Minimum value of distribution."""

    high: float
    """Maximum value of distribution."""

    dist_type: Literal[DistType.UNIFORM] = DistType.UNIFORM

    @model_validator(mode="after")
    def validate_low_high(self) -> Self:
        if self.high <= self.low:
            msg = f"Distribution {self.name}: high must be greater than low"
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.uniform(loc=self.low, scale=self.scale))

    @property
    def scale(self) -> float:
        return self.high - self.low

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.uniform(low=self.low, high=self.high, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x=x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x=x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high}


class TriangularDistribution(ContinuousDistribution[float]):
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
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/triangular.png" width="600" />

    """

    low: float
    """Minimum value of distribution."""

    mode: float
    """Peak value of distribution."""

    high: float
    """Maximum value of distribution."""

    dist_type: Literal[DistType.TRIANGULAR] = DistType.TRIANGULAR

    @model_validator(mode="after")
    def validate_logic(self) -> Self:
        if not (self.low <= self.mode <= self.high):
            msg = f"{self.name}: Must satisfy low <= mode <= high"
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        # scipy mapping: loc=low, scale=high-low, c=(mode-low)/scale
        rescale = self.high - self.low
        c = (self.mode - self.low) / rescale if rescale != 0 else 0
        return cast(
            rv_continuous_frozen, stats.triang(c=c, loc=self.low, scale=rescale)
        )

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.triangular(
            left=self.low, mode=self.mode, right=self.high, size=size
        )

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"low": self.low, "mode": self.mode, "high": self.high}


class TruncatedNormalDistribution(ContinuousDistribution[float]):
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
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/truncated_normal.png" width="600" />

    """

    mu: float
    """Mean value of distribution."""

    sigma: float
    """Standard deviation of distribution."""

    low: float = float("-inf")
    """Lower bound of distribution."""

    high: float = float("inf")
    """Upper bound of distribution."""

    dist_type: Literal[DistType.TRUNCATED_NORMAL] = DistType.TRUNCATED_NORMAL

    @field_validator("low", mode="before")
    @classmethod
    def _coerce_low(cls, v: float | None) -> float:
        return float("-inf") if v is None else float(v)

    @field_validator("high", mode="before")
    @classmethod
    def _coerce_high(cls, v: float | None) -> float:
        return float("inf") if v is None else float(v)

    @field_serializer("low")
    def _serialize_low(self, v: float) -> float | None:
        return None if math.isinf(v) else v

    @field_serializer("high")
    def _serialize_high(self, v: float) -> float | None:
        return None if math.isinf(v) else v

    @model_validator(mode="after")
    def validate_sigma(self) -> Self:
        if self.sigma <= 0:
            msg = f"Distribution {self.name} has a non-positive standard deviation. It must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        # a and b are the number of standard deviations away from the mean
        a = (self.low - self.mu) / self.sigma
        b = (self.high - self.mu) / self.sigma
        return cast(
            rv_continuous_frozen,
            stats.truncnorm(a=a, b=b, loc=self.mu, scale=self.sigma),
        )

    def draw(self, size: int = 1) -> np.ndarray:
        # numpy doesn't have a truncnorm generator
        return self._scipy.rvs(size=size, random_state=self.rng)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"mu": self.mu, "sigma": self.sigma, "low": self.low, "high": self.high}


class LogNormalDistribution(ContinuousDistribution[float]):
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
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/log_normal.png" width="600" />

    """

    s: float
    """The shape parameter (sigma of the log)"""

    scale: float = 1.0
    """exp(mu)"""

    dist_type: Literal[DistType.LOG_NORMAL] = DistType.LOG_NORMAL

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.lognorm(s=self.s, scale=self.scale))

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.lognormal(mean=np.log(self.scale), sigma=self.s, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"s": self.s, "scale": self.scale}


class ExponentialDistribution(ContinuousDistribution[float]):
    """
    Represent the time between events in a Poisson process.

    In robotics, this is often used to model time-to-failure or the duration between random perturbations.

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
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/exponential.png" width="600" />

    """

    lam: float
    """Rate parameter (lambda)."""

    dist_type: Literal[DistType.EXPONENTIAL] = DistType.EXPONENTIAL

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.expon(scale=1 / self.lam))

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.exponential(scale=1 / self.lam, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"lam": self.lam}


class RayleighDistribution(ContinuousDistribution[float]):
    """
    Represent the magnitude of a 2D vector whose components are independent, identically distributed Normal variables.

    Commonly used to model wind speed, signal amplitude noise, or radial positioning error where the underlying x and y errors are independent Gaussians.

    Examples:
        ```python
        # Modeling radial placement error of an object on a table
        dist = RayleighDistribution(name="placement_error", scale=0.02)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rayleigh.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rayleigh.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Rayleigh_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/rayleigh.png" width="600" />

    """

    scale: float
    """Scale parameter (sigma). Must be positive."""

    dist_type: Literal[DistType.RAYLEIGH] = DistType.RAYLEIGH

    @model_validator(mode="after")
    def validate_scale(self) -> Self:
        if self.scale <= 0:
            msg = f"Distribution {self.name} has a non-positive scale. It must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.rayleigh(scale=self.scale))

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.rayleigh(scale=self.scale, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"scale": self.scale}


class GammaDistribution(ContinuousDistribution[float]):
    """
    Represent a flexible, right-skewed distribution for positive-valued quantities.

    A generalization of the Exponential distribution that models the time until the alpha-th event in a Poisson process. Commonly used for link masses, stiffness scaling factors, and service times where values are strictly positive and right-skewed.

    Examples:
        ```python
        # Modeling a link mass with shape alpha=4 and scale beta=0.5 (mean=2 kg)
        dist = GammaDistribution(name="link_mass", alpha=4.0, beta=0.5)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Gamma_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/gamma.png" width="600" />

    """

    alpha: float
    """Shape parameter. Must be positive."""

    beta: float
    """Scale parameter. Must be positive. Mean = alpha * beta."""

    dist_type: Literal[DistType.GAMMA] = DistType.GAMMA

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.alpha <= 0:
            msg = f"Distribution {self.name}: alpha (shape) must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        if self.beta <= 0:
            msg = f"Distribution {self.name}: beta (scale) must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.gamma(a=self.alpha, scale=self.beta))

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.gamma(shape=self.alpha, scale=self.beta, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta}


class BetaDistribution(ContinuousDistribution[float]):
    """
    Represent a distribution over the interval (0, 1).

    Ideal for modeling probabilities, fractions, or any quantity that is naturally bounded between 0 and 1, such as friction coefficients or surface reflectance.

    Examples:
        ```python
        # Modeling friction coefficient constrained to (0, 1)
        dist = BetaDistribution(name="friction", alpha=2.0, beta=5.0)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/beta.png" width="600" />

    """

    alpha: float
    """First shape parameter. Must be positive."""

    beta: float
    """Second shape parameter. Must be positive."""

    dist_type: Literal[DistType.BETA] = DistType.BETA

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.alpha <= 0:
            msg = f"Distribution {self.name}: alpha must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        if self.beta <= 0:
            msg = f"Distribution {self.name}: beta must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.beta(a=self.alpha, b=self.beta))

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.beta(a=self.alpha, b=self.beta, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta}


class WeibullDistribution(ContinuousDistribution[float]):
    """
    Represent a flexible distribution for modeling time-to-failure and wind speed.

    Generalizes the Exponential distribution: shape < 1 models decreasing failure rate, shape = 1 reduces to Exponential, shape > 1 models increasing failure rate (wear-out). Common in reliability analysis and environmental modeling.

    Examples:
        ```python
        # Modeling component lifetime with increasing failure rate
        dist = WeibullDistribution(name="bearing_life", shape=2.5, scale=1000.0)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.weibull.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Weibull_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/weibull.png" width="600" />

    """

    shape: float
    """Shape parameter (k). Must be positive. Controls the failure rate behavior."""

    scale: float
    """Scale parameter (lambda). Must be positive. Sets the characteristic life."""

    dist_type: Literal[DistType.WEIBULL] = DistType.WEIBULL

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.shape <= 0:
            msg = f"Distribution {self.name}: shape must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        if self.scale <= 0:
            msg = f"Distribution {self.name}: scale must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(
            rv_continuous_frozen, stats.weibull_min(c=self.shape, scale=self.scale)
        )

    def draw(self, size: int = 1) -> np.ndarray:
        # numpy.weibull draws from the standard Weibull (scale=1); multiply by scale
        return self.rng.weibull(a=self.shape, size=size) * self.scale

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"shape": self.shape, "scale": self.scale}


class LogisticDistribution(ContinuousDistribution[float]):
    """
    Represent a symmetric, bell-shaped distribution with heavier tails than Normal.

    Common in modeling growth processes and any quantity where an S-shaped CDF is appropriate. Shares the same location-scale form as Normal but assigns more probability mass to extreme values.

    Examples:
        ```python
        # Modeling a centered score with moderate spread
        dist = LogisticDistribution(name="score", mu=0.0, beta=1.0)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.logistic.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Logistic_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/logistic.png" width="600" />

    """

    mu: float
    """Location parameter (mean and median)."""

    beta: float
    """Scale parameter. Must be positive. Controls the spread."""

    dist_type: Literal[DistType.LOGISTIC] = DistType.LOGISTIC

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.beta <= 0:
            msg = f"Distribution {self.name}: beta (scale) must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.logistic(loc=self.mu, scale=self.beta))

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.logistic(loc=self.mu, scale=self.beta, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"mu": self.mu, "beta": self.beta}


class ParetoDistribution(ContinuousDistribution[float]):
    """
    Represent a heavy-tailed power-law distribution for modeling extreme events.

    The Pareto distribution assigns most probability mass to small values while allowing rare, very large outliers. Use it when modeling phenomena that follow a power law, such as the magnitude of perturbation forces where small disturbances are common but large ones occasionally occur.

    Examples:
        ```python
        # Modeling rare high-magnitude disturbance forces (alpha=3, min force=0.1 N)
        dist = ParetoDistribution(name="disturbance_force", alpha=3.0, beta=0.1)
        ```

    References:
        1. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html)
        2. [Wikipedia](https://en.wikipedia.org/wiki/Pareto_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/pareto.png" width="600" />

    """

    alpha: float
    """Shape parameter. Must be positive. Controls the tail heaviness."""

    beta: float
    """Scale parameter (minimum value). Must be positive."""

    dist_type: Literal[DistType.PARETO] = DistType.PARETO

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.alpha <= 0:
            msg = f"Distribution {self.name}: alpha (shape) must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        if self.beta <= 0:
            msg = f"Distribution {self.name}: beta (scale) must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.pareto(b=self.alpha, scale=self.beta))

    def draw(self, size: int = 1) -> np.ndarray:
        # numpy pareto returns Lomax samples (x >= 0); add 1 and scale to match scipy's Pareto
        return (self.rng.pareto(a=self.alpha, size=size) + 1) * self.beta

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha, "beta": self.beta}


class StudentTDistribution(ContinuousDistribution[float]):
    """
    Represent a symmetric, bell-shaped distribution with heavier tails than Normal.

    The Student's t distribution is parameterized by degrees of freedom (nu). As nu increases, it approaches the Normal distribution. Use this when modeling noise or uncertainty estimated from a small number of samples, such as sensor calibration data with few observations.

    Examples:
        ```python
        # Modeling measurement noise estimated from 5 calibration samples
        dist = StudentTDistribution(name="measurement_noise", nu=5.0)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_t.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Student%27s_t-distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/student_t.png" width="600" />

    """

    nu: float
    """Degrees of freedom. Must be positive."""

    dist_type: Literal[DistType.STUDENT_T] = DistType.STUDENT_T

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.nu <= 0:
            msg = f"Distribution {self.name}: nu (degrees of freedom) must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.t(df=self.nu))

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.standard_t(df=self.nu, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"nu": self.nu}


class CauchyDistribution(ContinuousDistribution[float]):
    """
    Represent a symmetric, heavy-tailed distribution with undefined mean and variance.

    The Cauchy distribution is a pathological location-scale distribution whose mean and variance are undefined, making it suitable for modeling highly uncertain processes or extreme perturbations where no central tendency can be assumed.

    Examples:
        ```python
        # Modeling highly uncertain lateral force perturbations centered at zero
        dist = CauchyDistribution(name="lateral_force", theta=0.0, sigma=0.5)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_cauchy.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cauchy.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Cauchy_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/cauchy.png" width="600" />

    """

    theta: float
    """Location parameter (median)."""

    sigma: float
    """Scale parameter. Must be positive."""

    dist_type: Literal[DistType.CAUCHY] = DistType.CAUCHY

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.sigma <= 0:
            msg = f"Distribution {self.name}: sigma (scale) must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(
            rv_continuous_frozen, stats.cauchy(loc=self.theta, scale=self.sigma)
        )

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.standard_cauchy(size=size) * self.sigma + self.theta

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"theta": self.theta, "sigma": self.sigma}


class ChiSquaredDistribution(ContinuousDistribution[float]):
    """
    Represent the distribution of a sum of squared standard Normal variables.

    Parameterized by degrees of freedom p, this distribution arises naturally in variance estimation and hypothesis testing. In simulation, it can model the squared magnitude of a p-dimensional isotropic Gaussian perturbation.

    Examples:
        ```python
        # Modeling the squared norm of a 3D position error with unit variance
        dist = ChiSquaredDistribution(name="position_error_sq", p=3)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.chisquare.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Chi-squared_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/chi_squared.png" width="600" />

    """

    p: int
    """Degrees of freedom. Must be at least 1."""

    dist_type: Literal[DistType.CHI_SQUARED] = DistType.CHI_SQUARED

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.p < 1:
            msg = (
                f"Distribution {self.name}: p (degrees of freedom) must be at least 1."
            )
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.chi2(df=self.p))

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.chisquare(df=self.p, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"p": self.p}


class LaplaceDistribution(ContinuousDistribution[float]):
    """
    Represent a symmetric, double-exponential distribution with heavier tails than Normal.

    The Laplace distribution places more probability mass near the center and in the tails than Normal, making it useful for modeling sparse perturbations or sensor noise with occasional large spikes.

    Examples:
        ```python
        # Modeling zero-mean joint torque noise with occasional large spikes
        dist = LaplaceDistribution(name="torque_noise", mu=0.0, sigma=0.1)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.laplace.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.laplace.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/Laplace_distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/laplace.png" width="600" />

    """

    mu: float
    """Location parameter (mean and median)."""

    sigma: float
    """Scale parameter. Must be positive."""

    dist_type: Literal[DistType.LAPLACE] = DistType.LAPLACE

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.sigma <= 0:
            msg = f"Distribution {self.name}: sigma (scale) must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.laplace(loc=self.mu, scale=self.sigma))

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.laplace(loc=self.mu, scale=self.sigma, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"mu": self.mu, "sigma": self.sigma}


class FDistribution(ContinuousDistribution[float]):
    """
    Represent the ratio of two chi-squared distributions scaled by their degrees of freedom.

    The F distribution is defined by two positive degrees-of-freedom parameters and arises naturally when comparing variance estimates. In simulation it can model the ratio of two independent squared-error magnitudes or uncertainty in variance ratios.

    Examples:
        ```python
        # Modeling the ratio of two independent variance estimates (5 vs 10 DOF)
        dist = FDistribution(name="variance_ratio", nu1=5.0, nu2=10.0)
        ```

    References:
        1. [NumPy](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.f.html)
        2. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f.html)
        3. [Wikipedia](https://en.wikipedia.org/wiki/F-distribution)

    Note:
        <img src="https://raw.githubusercontent.com/Hydrowelder/stochas/refs/heads/main/docs/assets/distributions/f.png" width="600" />

    """

    nu1: float
    """Numerator degrees of freedom. Must be positive."""

    nu2: float
    """Denominator degrees of freedom. Must be positive."""

    dist_type: Literal[DistType.F] = DistType.F

    @model_validator(mode="after")
    def validate_params(self) -> Self:
        if self.nu1 <= 0:
            msg = f"Distribution {self.name}: nu1 (numerator degrees of freedom) must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        if self.nu2 <= 0:
            msg = f"Distribution {self.name}: nu2 (denominator degrees of freedom) must be greater than 0."
            logger.error(msg)
            raise ValueError(msg)
        return self

    def _build_scipy(self) -> rv_continuous_frozen:
        import scipy.stats as stats

        return cast(rv_continuous_frozen, stats.f(dfn=self.nu1, dfd=self.nu2))

    def draw(self, size: int = 1) -> np.ndarray:
        return self.rng.f(dfnum=self.nu1, dfden=self.nu2, size=size)

    def pdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.pdf(x)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.cdf(x)

    def ppf(self, q: float | np.ndarray) -> float | np.ndarray:
        return self._scipy.ppf(q)

    @property
    def table_params(self) -> dict[str, Any]:
        return {"nu1": self.nu1, "nu2": self.nu2}
