import numpy as np
import pytest

from process_manager import (
    BernoulliDistribution,
    CategoricalDistribution,
    DistName,
    PoissonDistribution,
)


def test_categorical_pmf_cdf():
    """Verify Categorical logic for non-numeric types."""
    choices = {"Low": 0.2, "Medium": 0.5, "High": 0.3}
    dist = CategoricalDistribution(name=DistName("risk"), choices=choices)

    assert dist.pmf("Low") == 0.2
    assert dist.pmf("Medium") == 0.5
    assert dist.pmf("None") == 0.0

    # CDF follows order of choices list: 0.2, 0.2+0.5, 0.2+0.5+0.3
    assert np.isclose(dist.cdf("Low"), 0.2)
    assert np.isclose(dist.cdf("Medium"), 0.7)
    assert np.isclose(dist.cdf("High"), 1.0)


def test_poisson_properties():
    """Verify Poisson PMF and step-function CDF."""
    lam = 2.0
    dist = PoissonDistribution(name=DistName("calls"), lam=lam)

    # PMF at k=0 is e^-lambda
    expected_pmf_0 = np.exp(-lam)
    assert np.isclose(dist.pmf(0), expected_pmf_0)

    # CDF at 0 should be same as PMF at 0
    assert np.isclose(dist.cdf(0), expected_pmf_0)
    # CDF should be a step function (cdf(0.5) == cdf(0))
    assert dist.cdf(0.5) == dist.cdf(0)

    # PPF
    assert dist.ppf(0.01) >= 0


def test_poisson_sampling():
    """Verify Poisson samples are non-negative integers."""
    dist = PoissonDistribution(name=DistName("test"), lam=5.0, seed=42)
    samples = dist.sample(100)

    assert np.all(samples >= 0)
    assert np.all(samples % 1 == 0)  # Check if integers
    assert np.isclose(np.mean(samples), 5.0, atol=1.0)


def test_bernoulli_initialization():
    """Verify Bernoulli validates probability bounds."""
    # Valid
    dist = BernoulliDistribution(name=DistName("test"), p=0.7, seed=42)
    assert dist.p == 0.7

    # Invalid
    with pytest.raises(ValueError, match="between 0 and 1"):
        BernoulliDistribution(name=DistName("bad"), p=1.2)


def test_bernoulli_math_properties():
    """Verify PMF, CDF, and PPF logic for Bernoulli."""
    p = 0.7
    dist = BernoulliDistribution(name=DistName("coin"), p=p)

    # PMF: P(X=1) = p, P(X=0) = 1-p
    assert np.isclose(dist.pmf(1), p)
    assert np.isclose(dist.pmf(0), 1 - p)
    assert dist.pmf(5) == 0.0

    # CDF: P(X <= x)
    assert np.isclose(dist.cdf(0), 1 - p)
    assert np.isclose(dist.cdf(1), 1.0)
    assert dist.cdf(-1) == 0.0

    # PPF: Inverse CDF
    assert dist.ppf(0.1) == 0  # Since P(X<=0) = 0.3, 0.1 quantile is 0
    assert dist.ppf(0.8) == 1  # Since P(X<=0) = 0.3, 0.8 quantile must be 1


def test_bernoulli_sampling():
    """Verify samples are binary and follow the distribution."""
    dist = BernoulliDistribution(name=DistName("test"), p=0.5, seed=123)
    samples = np.asarray(dist.sample(1000))

    assert set(samples).issubset({0, 1})
    # Mean of Bernoulli is p
    assert np.isclose(np.mean(samples), 0.5, atol=0.05)


def test_discrete_seeding_consistency():
    """Verify that different discrete classes respect the salted seed."""
    seed = 99
    # Same name + same seed = Same results
    p1 = PoissonDistribution(name=DistName("var"), lam=2, seed=seed)
    p2 = PoissonDistribution(name=DistName("var"), lam=2, seed=seed)
    assert np.array_equal(p1.sample(10), p2.sample(10))

    # Different name + same seed = Different results
    b1 = BernoulliDistribution(name=DistName("x"), p=0.5, seed=seed)
    b2 = BernoulliDistribution(name=DistName("y"), p=0.5, seed=seed)
    assert not np.array_equal(b1.sample(20), b2.sample(20))
