import numpy as np
import pytest

from process_manager import (
    DistName,
    ExponentialDistribution,
    LogNormalDistribution,
    NormalDistribution,
    TriangularDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
)


def test_normal_distribution_properties():
    mu, sigma = 100, 15
    dist = NormalDistribution(name=DistName("iq"), mu=mu, sigma=sigma)

    # Statistical properties
    assert dist.pdf(mu) > 0
    assert np.isclose(dist.cdf(mu), 0.5)
    assert np.isclose(dist.ppf(0.5), mu)

    # Sampling
    samples = dist.sample(1000)
    assert np.isclose(np.mean(samples), mu, atol=2.0)
    assert np.isclose(np.std(samples), sigma, atol=2.0)


def test_uniform_distribution_bounds():
    dist = UniformDistribution(name=DistName("u"), low=10, high=20)
    samples = dist.sample(100)

    assert np.all(samples >= 10)
    assert np.all(samples <= 20)
    assert np.isclose(dist.cdf(15), 0.5)


def test_triangular_validation():
    with pytest.raises(ValueError, match="Must satisfy low <= mode <= high"):
        TriangularDistribution(name=DistName("bad"), low=10, mode=5, high=20)


def test_truncated_normal_bounds():
    """Verify samples never exceed the specified lower and upper bounds."""
    mu, sigma = 100, 10
    lower, upper = 95, 105
    dist = TruncatedNormalDistribution(
        name=DistName("clamped"), mu=mu, sigma=sigma, low=lower, high=upper
    )

    samples = dist.sample(1000)
    assert np.all(samples >= lower)
    assert np.all(samples <= upper)

    # CDF at lower bound should be 0, upper should be 1
    assert np.isclose(dist.cdf(lower), 0.0, atol=1e-7)
    assert np.isclose(dist.cdf(upper), 1.0, atol=1e-7)


def test_truncated_normal_extreme_bounds():
    """Verify it handles a mean that is outside the bounds."""
    # Mean is 0, but we only allow samples between 10 and 20
    dist = TruncatedNormalDistribution(
        name=DistName("offset"), mu=0, sigma=1, low=10, high=20
    )
    samples = dist.sample(100)
    assert np.all(samples >= 10)


def test_log_normal_properties():
    """Verify LogNormal is positive and respects the shape parameter."""
    s = 0.5  # sigma of the log
    scale = np.exp(2)  # mu of the log is 2
    dist = LogNormalDistribution(name=DistName("skewed"), s=s, scale=scale)

    samples = dist.sample(1000)
    assert np.all(samples > 0)

    # Median of LogNormal is the 'scale' parameter (exp(mu))
    # So CDF(scale) should be 0.5
    assert np.isclose(dist.cdf(scale), 0.5)
    assert np.isclose(dist.ppf(0.5), scale)


def test_exponential_properties():
    """Verify Exponential distribution follows the rate lambda."""
    lam = 0.5
    dist = ExponentialDistribution(name=DistName("decay"), lam=lam)

    samples = dist.sample(1000)
    assert np.all(samples >= 0)

    # Theoretical mean of Exponential is 1/lambda
    expected_mean = 1 / lam
    assert np.isclose(np.mean(samples), expected_mean, atol=0.2)

    # CDF of Exponential is 1 - exp(-lam * x)
    # At x = 1/lam, CDF is 1 - exp(-1) ≈ 0.632
    assert np.isclose(dist.cdf(expected_mean), 1 - np.exp(-1))


@pytest.mark.parametrize(
    "dist_instance",
    [
        TruncatedNormalDistribution(
            name=DistName("tn"), mu=50, sigma=5, low=40, high=60
        ),
        LogNormalDistribution(name=DistName("ln"), s=0.5, scale=10),
        ExponentialDistribution(name=DistName("ex"), lam=2.0),
    ],
)
def test_continuous_ppf_consistency(dist_instance):
    """Generic check that PPF and CDF are true inverses."""
    qs = [0.01, 0.5, 0.99]
    for q in qs:
        val = dist_instance.ppf(q)
        assert np.isclose(dist_instance.cdf(val), q, atol=1e-7)
