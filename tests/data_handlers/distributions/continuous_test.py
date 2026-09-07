"""Tests for continuous distributions."""

import math

import numpy as np
import pytest

from stochas import (
    BetaDistribution,
    CauchyDistribution,
    ChiSquaredDistribution,
    DistName,
    ExponentialDistribution,
    FDistribution,
    GammaDistribution,
    LaplaceDistribution,
    LogisticDistribution,
    LogNormalDistribution,
    NormalDistribution,
    ParetoDistribution,
    RayleighDistribution,
    StudentTDistribution,
    TriangularDistribution,
    TruncatedNormalDistribution,
    UniformDistribution,
    WeibullDistribution,
)


def test_normal_distribution_properties():
    mu, sigma = 100, 15
    dist = NormalDistribution(name=DistName("iq"), mu=mu, sigma=sigma)

    # Statistical properties
    assert dist.pdf(mu) > 0
    assert np.isclose(dist.cdf(mu), 0.5)
    assert np.isclose(dist.ppf(0.5), mu)
    assert dist.is_continuous is True

    # Sampling
    samples = dist.sample(1000)
    assert np.isclose(np.mean(samples), mu, atol=2.0)
    assert np.isclose(np.std(samples), sigma, atol=2.0)


def test_normal_distribution_invalid_sigma():
    """Ensure sigma must be positive."""
    with pytest.raises(ValueError, match="negative standard deviation"):
        NormalDistribution(name=DistName("bad"), mu=0, sigma=0)


def test_normal_distribution_nominal_sample():
    """Ensure sample() returns the nominal value when trial_num is nominal."""
    dist = NormalDistribution(name=DistName("iq"), mu=100, sigma=15, nominal=110)

    samples = dist.sample(3)

    assert np.array_equal(samples, np.full(3, 110))


def test_uniform_distribution_bounds():
    dist = UniformDistribution(name=DistName("u"), low=10, high=20)
    samples = dist.sample(100)

    assert np.all(samples >= 10)
    assert np.all(samples <= 20)
    assert np.isclose(dist.cdf(15), 0.5)
    assert dist.pdf(15) > 0
    assert np.isclose(dist.ppf(0.5), 15)
    assert dist.is_continuous is True


def test_uniform_distribution_validation():
    """Ensure high must be greater than low."""
    with pytest.raises(ValueError, match="high must be greater than low"):
        UniformDistribution(name=DistName("bad"), low=20, high=10)


def test_triangular_validation():
    with pytest.raises(ValueError, match="Must satisfy low <= mode <= high"):
        TriangularDistribution(name=DistName("bad"), low=10, mode=5, high=20)


def test_triangular_properties():
    """Verify Triangular sampling, PDF, CDF, PPF, and is_continuous."""
    dist = TriangularDistribution(name=DistName("tri"), low=0, mode=5, high=10)

    samples = dist.sample(100)
    assert np.all(samples >= 0)
    assert np.all(samples <= 10)

    assert dist.pdf(5) > 0
    assert np.isclose(dist.cdf(0), 0.0)
    assert np.isclose(dist.cdf(10), 1.0)
    assert np.isclose(dist.ppf(0.5), dist.mode, atol=1.0)
    assert dist.is_continuous is True


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

    assert dist.pdf(mu) > 0
    assert dist.is_continuous is True


def test_truncated_normal_extreme_bounds():
    """Verify it handles a mean that is outside the bounds."""
    # Mean is 0, but we only allow samples between 10 and 20
    dist = TruncatedNormalDistribution(
        name=DistName("offset"), mu=0, sigma=1, low=10, high=20
    )
    samples = dist.sample(100)
    assert np.all(samples >= 10)


def test_truncated_normal_infinite_bounds_roundtrip():
    """Verify that infinite bounds serialize as None and restore correctly."""
    dist = TruncatedNormalDistribution(name=DistName("inf"), mu=0.0, sigma=1.0)
    assert math.isinf(dist.low)
    assert math.isinf(dist.high)

    data = dist.model_dump()
    assert data["low"] is None
    assert data["high"] is None

    restored = TruncatedNormalDistribution.model_validate(data)
    assert math.isinf(restored.low)
    assert math.isinf(restored.high)


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

    assert dist.pdf(scale) > 0
    assert dist.is_continuous is True


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

    assert dist.pdf(expected_mean) > 0
    assert dist.is_continuous is True


def test_rayleigh_properties():
    """Verify Rayleigh distribution follows the scale (sigma) parameter."""
    scale = 2.0
    dist = RayleighDistribution(name=DistName("error"), scale=scale)

    samples = dist.sample(1000)
    assert np.all(samples >= 0)

    # Theoretical mean of Rayleigh is scale * sqrt(pi / 2)
    expected_mean = scale * np.sqrt(np.pi / 2)
    assert np.isclose(np.mean(samples), expected_mean, atol=0.2)

    # Median of Rayleigh is scale * sqrt(ln(4))
    median = scale * np.sqrt(np.log(4))
    assert np.isclose(dist.cdf(median), 0.5)
    assert np.isclose(dist.ppf(0.5), median)

    assert dist.pdf(median) > 0
    assert dist.is_continuous is True


def test_rayleigh_validation():
    with pytest.raises(ValueError, match="non-positive scale"):
        RayleighDistribution(name=DistName("bad"), scale=0)


@pytest.mark.parametrize(
    "dist_instance",
    [
        TruncatedNormalDistribution(
            name=DistName("tn"), mu=50, sigma=5, low=40, high=60
        ),
        LogNormalDistribution(name=DistName("ln"), s=0.5, scale=10),
        ExponentialDistribution(name=DistName("ex"), lam=2.0),
        RayleighDistribution(name=DistName("ry"), scale=2.0),
        LogisticDistribution(name=DistName("lo"), mu=0.0, beta=1.0),
        ParetoDistribution(name=DistName("pa"), alpha=2.0, beta=1.0),
        StudentTDistribution(name=DistName("st"), nu=5.0),
        CauchyDistribution(name=DistName("ca"), theta=0.0, sigma=1.0),
        ChiSquaredDistribution(name=DistName("cs"), p=3),
        LaplaceDistribution(name=DistName("la"), mu=0.0, sigma=1.0),
        FDistribution(name=DistName("fd"), nu1=5.0, nu2=10.0),
    ],
)
def test_continuous_ppf_consistency(dist_instance):
    """Generic check that PPF and CDF are true inverses."""
    qs = [0.01, 0.5, 0.99]
    for q in qs:
        val = dist_instance.ppf(q)
        assert np.isclose(dist_instance.cdf(val), q, atol=1e-7)


def test_gamma_properties():
    """Verify Gamma sampling is positive and mean tracks alpha*beta."""
    alpha, beta = 3.0, 2.0
    dist = GammaDistribution(name=DistName("g"), alpha=alpha, beta=beta)

    samples = dist.sample(1000)
    assert np.all(samples > 0)
    assert np.isclose(np.mean(samples), alpha * beta, atol=0.5)

    assert dist.pdf(alpha * beta) > 0
    assert np.isclose(dist.cdf(0), 0.0, atol=1e-6)
    assert np.isclose(dist.cdf(dist.ppf(0.5)), 0.5, atol=1e-6)
    assert dist.is_continuous is True


def test_gamma_validation():
    with pytest.raises(ValueError, match="alpha"):
        GammaDistribution(name=DistName("bad"), alpha=0.0, beta=1.0)
    with pytest.raises(ValueError, match="beta"):
        GammaDistribution(name=DistName("bad"), alpha=1.0, beta=-1.0)


def test_beta_properties():
    """Verify Beta samples are in (0,1) and CDF at mode is consistent."""
    alpha, beta = 2.0, 5.0
    dist = BetaDistribution(name=DistName("b"), alpha=alpha, beta=beta)

    samples = dist.sample(1000)
    assert np.all(samples > 0)
    assert np.all(samples < 1)

    # mean of Beta(alpha, beta) = alpha / (alpha + beta)
    expected_mean = alpha / (alpha + beta)
    assert np.isclose(np.mean(samples), expected_mean, atol=0.05)

    assert dist.pdf(0.5) > 0
    assert np.isclose(dist.cdf(0.0), 0.0, atol=1e-6)
    assert np.isclose(dist.cdf(1.0), 1.0, atol=1e-6)
    assert np.isclose(dist.cdf(dist.ppf(0.5)), 0.5, atol=1e-6)
    assert dist.is_continuous is True


def test_beta_validation():
    with pytest.raises(ValueError, match="alpha"):
        BetaDistribution(name=DistName("bad"), alpha=0.0, beta=1.0)
    with pytest.raises(ValueError, match="beta"):
        BetaDistribution(name=DistName("bad"), alpha=1.0, beta=0.0)


def test_weibull_properties():
    """Verify Weibull samples are positive and PPF/CDF are consistent."""
    shape, scale = 2.0, 100.0
    dist = WeibullDistribution(name=DistName("w"), shape=shape, scale=scale)

    samples = dist.sample(1000)
    assert np.all(samples > 0)

    # median of Weibull = scale * ln(2)^(1/shape)
    median = scale * (np.log(2) ** (1 / shape))
    assert np.isclose(dist.cdf(median), 0.5, atol=0.01)
    assert np.isclose(dist.ppf(0.5), median, atol=0.5)

    assert dist.pdf(scale) > 0
    assert dist.is_continuous is True


def test_weibull_validation():
    with pytest.raises(ValueError, match="shape"):
        WeibullDistribution(name=DistName("bad"), shape=0.0, scale=1.0)
    with pytest.raises(ValueError, match="scale"):
        WeibullDistribution(name=DistName("bad"), shape=1.0, scale=-1.0)


def test_logistic_properties():
    """Verify Logistic is symmetric about mu with CDF of 0.5 at the mean."""
    mu, beta = 5.0, 2.0
    dist = LogisticDistribution(name=DistName("lo"), mu=mu, beta=beta)

    samples = dist.sample(10000)
    assert np.isclose(np.mean(samples), mu, atol=0.3)

    assert np.isclose(dist.cdf(mu), 0.5)
    assert np.isclose(dist.ppf(0.5), mu)
    assert dist.pdf(mu) > 0
    assert dist.is_continuous is True


def test_logistic_validation():
    with pytest.raises(ValueError, match="beta"):
        LogisticDistribution(name=DistName("bad"), mu=0.0, beta=0.0)


def test_pareto_properties():
    """Verify Pareto samples are >= beta and CDF/PPF are consistent."""
    alpha, beta = 3.0, 1.0
    dist = ParetoDistribution(name=DistName("pa"), alpha=alpha, beta=beta)

    samples = dist.sample(1000)
    assert np.all(samples >= beta)

    # CDF at beta (minimum) is 0; PPF should be consistent with CDF
    assert np.isclose(dist.cdf(beta), 0.0, atol=1e-6)
    assert np.isclose(dist.cdf(dist.ppf(0.5)), 0.5, atol=1e-6)

    assert dist.pdf(2 * beta) > 0
    assert dist.is_continuous is True


def test_pareto_validation():
    with pytest.raises(ValueError, match="alpha"):
        ParetoDistribution(name=DistName("bad"), alpha=0.0, beta=1.0)
    with pytest.raises(ValueError, match="beta"):
        ParetoDistribution(name=DistName("bad"), alpha=1.0, beta=-1.0)


def test_student_t_properties():
    """Verify Student-T is symmetric about zero with CDF of 0.5 at zero."""
    nu = 5.0
    dist = StudentTDistribution(name=DistName("st"), nu=nu)

    samples = dist.sample(1000)
    assert np.isclose(np.mean(samples), 0.0, atol=0.2)

    assert np.isclose(dist.cdf(0.0), 0.5)
    assert np.isclose(dist.ppf(0.5), 0.0)
    assert dist.pdf(0.0) > 0
    assert dist.is_continuous is True


def test_student_t_validation():
    with pytest.raises(ValueError, match="nu"):
        StudentTDistribution(name=DistName("bad"), nu=0.0)


def test_cauchy_properties():
    """Verify Cauchy CDF is 0.5 at theta and PPF/CDF are consistent."""
    theta, sigma = 2.0, 1.0
    dist = CauchyDistribution(name=DistName("ca"), theta=theta, sigma=sigma)

    assert np.isclose(dist.cdf(theta), 0.5)
    assert np.isclose(dist.ppf(0.5), theta)
    assert dist.pdf(theta) > 0
    assert dist.is_continuous is True

    samples = dist.sample(100)
    assert isinstance(samples, np.ndarray) and len(samples) == 100


def test_cauchy_validation():
    with pytest.raises(ValueError, match="sigma"):
        CauchyDistribution(name=DistName("bad"), theta=0.0, sigma=0.0)


def test_chi_squared_properties():
    """Verify ChiSquared samples are positive and mean tracks p."""
    p = 4
    dist = ChiSquaredDistribution(name=DistName("cs"), p=p)

    samples = dist.sample(1000)
    assert np.all(samples > 0)
    assert np.isclose(np.mean(samples), p, atol=0.3)

    assert np.isclose(dist.cdf(0.0), 0.0, atol=1e-6)
    assert np.isclose(dist.cdf(dist.ppf(0.5)), 0.5, atol=1e-6)
    assert dist.pdf(p) > 0
    assert dist.is_continuous is True


def test_chi_squared_validation():
    with pytest.raises(ValueError, match="p"):
        ChiSquaredDistribution(name=DistName("bad"), p=0)


def test_laplace_properties():
    """Verify Laplace is symmetric about mu with CDF of 0.5 at the mean."""
    mu, sigma = 3.0, 1.5
    dist = LaplaceDistribution(name=DistName("la"), mu=mu, sigma=sigma)

    samples = dist.sample(1000)
    assert np.isclose(np.mean(samples), mu, atol=0.2)

    assert np.isclose(dist.cdf(mu), 0.5)
    assert np.isclose(dist.ppf(0.5), mu)
    assert dist.pdf(mu) > 0
    assert dist.is_continuous is True


def test_laplace_validation():
    with pytest.raises(ValueError, match="sigma"):
        LaplaceDistribution(name=DistName("bad"), mu=0.0, sigma=0.0)


def test_f_distribution_properties():
    """Verify F samples are positive and mean tracks nu2 / (nu2 - 2) for nu2 > 2."""
    nu1, nu2 = 5.0, 10.0
    dist = FDistribution(name=DistName("fd"), nu1=nu1, nu2=nu2)

    samples = dist.sample(1000)
    assert np.all(samples > 0)

    # mean of F(nu1, nu2) = nu2 / (nu2 - 2) for nu2 > 2
    expected_mean = nu2 / (nu2 - 2)
    assert np.isclose(np.mean(samples), expected_mean, atol=0.3)

    assert np.isclose(dist.cdf(0.0), 0.0, atol=1e-6)
    assert np.isclose(dist.cdf(dist.ppf(0.5)), 0.5, atol=1e-6)
    assert dist.pdf(1.0) > 0
    assert dist.is_continuous is True


def test_f_distribution_validation():
    with pytest.raises(ValueError, match="nu1"):
        FDistribution(name=DistName("bad"), nu1=0.0, nu2=5.0)
    with pytest.raises(ValueError, match="nu2"):
        FDistribution(name=DistName("bad"), nu1=5.0, nu2=0.0)


@pytest.mark.parametrize(
    "dist, expected",
    [
        (
            NormalDistribution(name=DistName("n"), mu=1.0, sigma=2.0),
            {"mu": 1.0, "sigma": 2.0},
        ),
        (
            UniformDistribution(name=DistName("u"), low=0.5, high=1.5),
            {"low": 0.5, "high": 1.5},
        ),
        (
            TriangularDistribution(name=DistName("t"), low=0.0, mode=0.5, high=1.0),
            {"low": 0.0, "mode": 0.5, "high": 1.0},
        ),
        (
            TruncatedNormalDistribution(
                name=DistName("tn"), mu=0.0, sigma=1.0, low=-2.0, high=2.0
            ),
            {"mu": 0.0, "sigma": 1.0, "low": -2.0, "high": 2.0},
        ),
        (
            LogNormalDistribution(name=DistName("ln"), s=0.5, scale=2.0),
            {"s": 0.5, "scale": 2.0},
        ),
        (
            ExponentialDistribution(name=DistName("ex"), lam=1.5),
            {"lam": 1.5},
        ),
        (
            RayleighDistribution(name=DistName("ry"), scale=3.0),
            {"scale": 3.0},
        ),
        (
            GammaDistribution(name=DistName("ga"), alpha=2.0, beta=1.5),
            {"alpha": 2.0, "beta": 1.5},
        ),
        (
            BetaDistribution(name=DistName("be"), alpha=2.0, beta=5.0),
            {"alpha": 2.0, "beta": 5.0},
        ),
        (
            WeibullDistribution(name=DistName("wb"), shape=2.0, scale=100.0),
            {"shape": 2.0, "scale": 100.0},
        ),
        (
            LogisticDistribution(name=DistName("lo"), mu=1.0, beta=2.0),
            {"mu": 1.0, "beta": 2.0},
        ),
        (
            ParetoDistribution(name=DistName("pa"), alpha=3.0, beta=1.0),
            {"alpha": 3.0, "beta": 1.0},
        ),
        (
            StudentTDistribution(name=DistName("st"), nu=5.0),
            {"nu": 5.0},
        ),
        (
            CauchyDistribution(name=DistName("ca"), theta=1.0, sigma=2.0),
            {"theta": 1.0, "sigma": 2.0},
        ),
        (
            ChiSquaredDistribution(name=DistName("cs"), p=4),
            {"p": 4},
        ),
        (
            LaplaceDistribution(name=DistName("la"), mu=0.0, sigma=1.5),
            {"mu": 0.0, "sigma": 1.5},
        ),
        (
            FDistribution(name=DistName("fd"), nu1=5.0, nu2=10.0),
            {"nu1": 5.0, "nu2": 10.0},
        ),
    ],
)
def test_continuous_table_params(dist, expected):
    assert dist.table_params == expected
