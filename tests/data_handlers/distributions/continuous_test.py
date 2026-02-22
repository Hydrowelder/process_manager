import numpy as np
import pytest

from process_manager import (
    NormalDistribution,
    TriangularDistribution,
    UniformDistribution,
)


def test_normal_distribution_properties():
    mu, sigma = 100, 15
    dist = NormalDistribution(name="iq", mu=mu, sigma=sigma)

    # Statistical properties
    assert dist.pdf(mu) > 0
    assert np.isclose(dist.cdf(mu), 0.5)
    assert np.isclose(dist.ppf(0.5), mu)

    # Sampling
    samples = dist.sample(1000)
    assert np.isclose(np.mean(samples), mu, atol=2.0)
    assert np.isclose(np.std(samples), sigma, atol=2.0)


def test_uniform_distribution_bounds():
    dist = UniformDistribution(name="u", low=10, high=20)
    samples = dist.sample(100)

    assert np.all(samples >= 10)
    assert np.all(samples <= 20)
    assert np.isclose(dist.cdf(15), 0.5)


def test_triangular_validation():
    with pytest.raises(ValueError, match="Must satisfy low <= mode <= high"):
        TriangularDistribution(name="bad", low=10, mode=5, high=20)
