"""Tests for discrete distributions."""

import numpy as np
import pytest

from stochas import (
    BernoulliDistribution,
    BetaBinomialDistribution,
    BinomialDistribution,
    CategoricalDistribution,
    DistName,
    GeometricDistribution,
    HypergeometricDistribution,
    NegativeBinomialDistribution,
    PoissonDistribution,
)
from stochas.distribution import DistributionDict, PermutationDistribution
from stochas.named_value import NamedValueDict


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

    # An unknown category is not in the choices list, so cdf falls back to 0.0
    assert dist.cdf("Unknown") == 0.0

    assert dist.pdf("Low") == dist.pmf("Low")
    assert dist.is_continuous is False
    assert dist.ppf(0.01) == 0


def test_categorical_sampling():
    """Verify Categorical sampling draws only from the provided choices."""
    choices = {"Low": 0.2, "Medium": 0.5, "High": 0.3}
    dist = CategoricalDistribution(name=DistName("risk"), choices=choices, seed=42)

    samples = dist.sample(100)

    assert set(samples).issubset(set(choices))


def test_categorical_validate_probabilities():
    """Ensure probabilities must sum to 1."""
    with pytest.raises(ValueError, match="do not sum to 1"):
        CategoricalDistribution(name=DistName("bad"), choices={"a": 0.5, "b": 0.6})


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

    assert dist.is_continuous is False
    assert dist.pdf(0) == dist.pmf(0)


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

    assert dist.is_continuous is False
    assert dist.pdf(1) == dist.pmf(1)


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


def test_permutation_integrity():
    """Verify a single draw contains all original items exactly once and is squeezable."""
    items = ["Alpha", "Beta", "Gamma"]
    dist = PermutationDistribution[str](
        name=DistName("task_order"), items=items, seed=42, trial_num=1
    )

    # Draw size=1
    sample = dist.sample(size=1)

    # Verify 2D shape (size, items) -> (1, 3)
    assert sample.shape == (1, 3)

    # Verify Squeeze works
    squeezed = sample.squeeze()
    assert squeezed.shape == (3,)
    assert set(squeezed) == set(items)


def test_permutation_multiple_size():
    """Verify drawing multiple permutations returns a (size, N) array."""
    items = [1, 2, 3, 4]
    dist = PermutationDistribution[int](
        name=DistName("multi_test"), items=items, seed=42
    )

    size = 5
    # Move past nominal trial to use draw()
    dist.trial_num = 1
    samples = dist.sample(size=size)

    assert samples.shape == (size, 4)
    # Ensure every row is a valid shuffle
    for row in samples:
        assert set(row) == set(items)


def test_permutation_nominal_fallback():
    """Verify trial_num=0 returns the nominal value inside a 2D array."""
    items = [1, 2, 3]
    nominal_order = [3, 2, 1]

    dist = PermutationDistribution[int](
        name=DistName("nominal_test"), items=items, nominal=nominal_order
    )

    dist.trial_num = 0
    sample = dist.sample(size=1)

    # Check that it's wrapped in a list/array for 2D structure
    assert sample.shape == (1, 3)
    assert np.array_equal(sample[0], nominal_order)


def test_permutation_serialization_and_typing():
    """
    Verify that sample_and_update_dicts resolves the TypeVar 'T'
    even with the nested array structure.
    """
    items = [1.1, 2.2, 3.3]  # Float items
    dist = PermutationDistribution[float](
        name=DistName("ser_test"), items=items, seed=42, trial_num=1
    )

    dist_dict = DistributionDict()
    named_dict = NamedValueDict()

    # This call relies on: concrete_type = samples.dtype.type().item().__class__
    # Since samples is now a 2D array of floats, .item() on a float64
    # should still resolve to <class 'float'>.
    nv = dist.sample_and_update_dicts(
        dist_dict=dist_dict, named_value_dict=named_dict, size=1
    )

    # 1. Check runtime type resolution
    # nv.stored_value is the 2D array
    assert nv.stored_value.dtype.kind == "f"  # pyright: ignore[reportAttributeAccessIssue]

    # 2. Check Serialization
    dumped = nv.model_dump()

    # Should be a nested list: [[1.1, 2.2, 3.3]]
    assert isinstance(dumped["stored_value"], np.ndarray)
    assert isinstance(dumped["stored_value"][0], np.ndarray)
    assert isinstance(dumped["stored_value"][0][0], float)


def test_permutation_pmf_logic():
    """Verify probability math for a single permutation outcome."""
    items = ["A", "B", "C"]  # 3! = 6
    dist = PermutationDistribution[str](name=DistName("pmf"), items=items)

    expected_prob = 1.0 / 6.0
    # Valid permutation
    assert np.isclose(dist.pmf(np.array(["C", "A", "B"])), expected_prob)
    # Invalid permutation
    assert dist.pmf(np.array(["A", "B", "D"])) == 0.0


def test_permutation_errors():
    """Ensure unimplemented methods raise errors."""
    dist = PermutationDistribution(name=DistName("test"), items=[1, 2])
    with pytest.raises(NotImplementedError):
        dist.cdf([1, 2])

    with pytest.raises(NotImplementedError):
        dist.ppf(0.5)


def test_permutation_pdf_and_is_continuous():
    """Verify pdf proxies to pmf and is_continuous is False."""
    items = ["A", "B", "C"]
    dist = PermutationDistribution[str](name=DistName("pdf"), items=items)

    assert dist.is_continuous is False
    assert dist.pdf(np.array(["A", "B", "C"])) == dist.pmf(np.array(["A", "B", "C"]))


def test_binomial_properties():
    """Verify Binomial PMF, CDF, and mean."""
    n, p = 20, 0.4
    dist = BinomialDistribution(name=DistName("bin"), n=n, p=p)

    # PMF at k=0 should be (1-p)^n
    assert np.isclose(dist.pmf(0), (1 - p) ** n)

    # CDF is monotone non-decreasing and reaches 1 at k=n
    assert dist.cdf(n) == 1.0
    assert dist.cdf(5) <= dist.cdf(10)

    # PPF
    assert dist.ppf(0.5) >= 0

    assert dist.is_continuous is False
    assert dist.pdf(8) == dist.pmf(8)


def test_binomial_sampling():
    """Verify Binomial samples are non-negative integers bounded by n."""
    n, p = 10, 0.3
    dist = BinomialDistribution(name=DistName("bin_sample"), n=n, p=p, seed=42)
    samples = dist.sample(500)

    assert np.all(samples >= 0)
    assert np.all(samples <= n)
    assert np.all(samples % 1 == 0)
    assert np.isclose(np.mean(samples), n * p, atol=0.5)


def test_binomial_validation():
    with pytest.raises(ValueError, match="n must be at least 1"):
        BinomialDistribution(name=DistName("bad"), n=0, p=0.5)
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        BinomialDistribution(name=DistName("bad"), n=5, p=1.5)


def test_negative_binomial_properties():
    """Verify NegativeBinomial PMF/CDF and mean tracks r*(1-p)/p."""
    r, p = 5, 0.4
    dist = NegativeBinomialDistribution(name=DistName("nb"), r=r, p=p)

    # PMF at k=0 is p^r
    assert np.isclose(dist.pmf(0), p**r)

    assert np.isclose(dist.cdf(50), 1.0, atol=1e-6)
    assert dist.cdf(5) <= dist.cdf(10)

    assert dist.ppf(0.5) >= 0
    assert dist.is_continuous is False
    assert dist.pdf(3) == dist.pmf(3)


def test_negative_binomial_sampling():
    """Verify NegativeBinomial samples are non-negative integers with correct mean."""
    r, p = 3, 0.5
    dist = NegativeBinomialDistribution(name=DistName("nb_s"), r=r, p=p, seed=42)
    samples = dist.sample(1000)

    assert np.all(samples >= 0)
    assert np.all(samples % 1 == 0)
    assert np.isclose(np.mean(samples), r * (1 - p) / p, atol=0.5)


def test_negative_binomial_validation():
    with pytest.raises(ValueError, match="r must be at least 1"):
        NegativeBinomialDistribution(name=DistName("bad"), r=0, p=0.5)
    with pytest.raises(ValueError, match="p must be in"):
        NegativeBinomialDistribution(name=DistName("bad"), r=1, p=0.0)


def test_geometric_properties():
    """Verify Geometric PMF/CDF and mean tracks 1/p."""
    p = 0.3
    dist = GeometricDistribution(name=DistName("geo"), p=p)

    # PMF at k=1 is p (first trial is a success)
    assert np.isclose(dist.pmf(1), p)

    assert dist.cdf(5) <= dist.cdf(10)
    assert np.isclose(dist.cdf(100), 1.0, atol=1e-6)

    assert dist.ppf(0.5) >= 1
    assert dist.is_continuous is False
    assert dist.pdf(2) == dist.pmf(2)


def test_geometric_sampling():
    """Verify Geometric samples are positive integers (1-indexed) with correct mean."""
    p = 0.25
    dist = GeometricDistribution(name=DistName("geo_s"), p=p, seed=42)
    samples = dist.sample(1000)

    assert np.all(samples >= 1)
    assert np.all(samples % 1 == 0)
    assert np.isclose(np.mean(samples), 1 / p, atol=0.5)


def test_geometric_validation():
    with pytest.raises(ValueError, match="p must be in"):
        GeometricDistribution(name=DistName("bad"), p=0.0)
    with pytest.raises(ValueError, match="p must be in"):
        GeometricDistribution(name=DistName("bad"), p=1.5)


def test_hypergeometric_properties():
    """Verify Hypergeometric samples are in [0, min(M, K)] and mean tracks M*K/N."""
    N, M, K = 50, 10, 8
    dist = HypergeometricDistribution(
        name=DistName("hg"), N=N, M=M, K=K, seed=42, trial_num=1
    )

    samples = dist.sample(1000)
    assert np.all(samples >= 0)
    assert np.all(samples <= min(M, K))
    assert np.all(samples % 1 == 0)

    # mean of Hypergeometric = M * K / N
    expected_mean = M * K / N
    assert np.isclose(np.mean(samples), expected_mean, atol=0.3)

    # PMF at mode and CDF/PPF consistency
    assert dist.pmf(round(expected_mean)) >= 0
    assert dist.cdf(M) == 1.0
    assert dist.ppf(0.5) >= 0
    assert dist.is_continuous is False
    assert dist.pdf(2) == dist.pmf(2)


def test_hypergeometric_validation():
    with pytest.raises(ValueError, match="N"):
        HypergeometricDistribution(name=DistName("bad"), N=0, M=1, K=0)
    with pytest.raises(ValueError, match="M"):
        HypergeometricDistribution(name=DistName("bad"), N=10, M=0, K=5)
    with pytest.raises(ValueError, match="M"):
        HypergeometricDistribution(name=DistName("bad"), N=10, M=11, K=5)
    with pytest.raises(ValueError, match="K"):
        HypergeometricDistribution(name=DistName("bad"), N=10, M=5, K=11)


def test_beta_binomial_properties():
    """Verify BetaBinomial samples are in [0, n] and mean tracks n*alpha/(alpha+beta)."""
    n, alpha, beta = 20, 8.0, 2.0
    dist = BetaBinomialDistribution(
        name=DistName("bb"), n=n, alpha=alpha, beta=beta, seed=42, trial_num=1
    )

    samples = dist.sample(1000)
    assert np.all(samples >= 0)
    assert np.all(samples <= n)
    assert np.all(samples % 1 == 0)

    # mean of BetaBinomial = n * alpha / (alpha + beta)
    expected_mean = n * alpha / (alpha + beta)
    assert np.isclose(np.mean(samples), expected_mean, atol=0.5)

    assert dist.cdf(n) == 1.0
    assert dist.ppf(0.5) >= 0
    assert dist.is_continuous is False
    assert dist.pdf(10) == dist.pmf(10)


def test_beta_binomial_validation():
    with pytest.raises(ValueError, match="n must be at least 1"):
        BetaBinomialDistribution(name=DistName("bad"), n=0, alpha=1.0, beta=1.0)
    with pytest.raises(ValueError, match="alpha"):
        BetaBinomialDistribution(name=DistName("bad"), n=5, alpha=0.0, beta=1.0)
    with pytest.raises(ValueError, match="beta"):
        BetaBinomialDistribution(name=DistName("bad"), n=5, alpha=1.0, beta=-1.0)


@pytest.mark.parametrize(
    "dist, expected",
    [
        (
            PoissonDistribution(name=DistName("p"), lam=3.0),
            {"lam": 3.0},
        ),
        (
            BernoulliDistribution(name=DistName("b"), p=0.4),
            {"p": 0.4},
        ),
        (
            CategoricalDistribution(name=DistName("c"), choices={"a": 0.3, "b": 0.7}),
            {"choices": "a: 0.3, b: 0.7"},
        ),
        (
            PermutationDistribution(name=DistName("perm"), items=[1, 2, 3]),
            {"items": "[1, 2, 3]"},
        ),
        (
            BinomialDistribution(name=DistName("bin"), n=10, p=0.3),
            {"n": 10, "p": 0.3},
        ),
        (
            NegativeBinomialDistribution(name=DistName("nb"), r=3, p=0.5),
            {"r": 3, "p": 0.5},
        ),
        (
            GeometricDistribution(name=DistName("geo"), p=0.4),
            {"p": 0.4},
        ),
        (
            HypergeometricDistribution(name=DistName("hg"), N=50, M=10, K=8),
            {"N": 50, "M": 10, "K": 8},
        ),
        (
            BetaBinomialDistribution(name=DistName("bb"), n=10, alpha=2.0, beta=5.0),
            {"n": 10, "alpha": 2.0, "beta": 5.0},
        ),
    ],
)
def test_discrete_table_params(dist, expected):
    assert dist.table_params == expected


if __name__ == "__main__":
    test_permutation_integrity()
