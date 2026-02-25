import numpy as np

from process_manager import DistName, NormalDistribution


def test_distribution_seeding_and_salting():
    """Verify that names act as unique salts for the same seed."""
    seed = 42

    # Same name, same seed -> Identical results
    dist_a1 = NormalDistribution(name=DistName("x"), mu=0, sigma=1, seed=seed)
    dist_a2 = NormalDistribution(name=DistName("x"), mu=0, sigma=1, seed=seed)

    # Different name, same seed -> Different results
    dist_b = NormalDistribution(name=DistName("y"), mu=0, sigma=1, seed=seed)

    samples_a1 = dist_a1.sample(5)
    samples_a2 = dist_a2.sample(5)
    samples_b = dist_b.sample(5)

    assert np.array_equal(samples_a1, samples_a2)
    assert not np.array_equal(samples_a1, samples_b)


def test_serialization_roundtrip():
    """Verify Pydantic serialization preserves state."""
    dist = NormalDistribution(name=DistName("test"), mu=10, sigma=2, seed=123)
    json_data = dist.model_dump_json()

    new_dist = NormalDistribution.model_validate_json(json_data)

    assert new_dist.name == dist.name
    assert new_dist.seed == dist.seed
    assert new_dist.mu == dist.mu
    assert new_dist.nominal == dist.nominal
    assert np.array_equal(dist.sample(5), new_dist.sample(5))
