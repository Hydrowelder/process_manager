import numpy as np

from process_manager import CategoricalDistribution, PoissonDistribution


def test_categorical_sampling():
    choices = [("A", 0.9), ("B", 0.1)]
    dist = CategoricalDistribution(name="cat", choices=choices)

    samples = dist.sample(100)

    assert set(samples).issubset({"A", "B"})

    assert dist.pmf("A") == 0.9
    assert dist.pmf("C") == 0.0


def test_poisson_logic():
    lam = 4.0
    dist = PoissonDistribution(name="p", lam=lam)

    samples = dist.sample(1000)
    assert samples.dtype.kind in "iu"  # Must be integers
    assert np.isclose(np.mean(samples), lam, atol=0.5)

    # Test CDF override logic (Liskov compliance)
    assert dist.cdf(4.0) == dist.cdf(4.5)
