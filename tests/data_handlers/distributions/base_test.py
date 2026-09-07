"""Tests for the Distribution base class and its dict/list registries."""

import numpy as np
import pytest
from numpydantic import NDArray

from stochas import DistName, NormalDistribution, UniformDistribution
from stochas.distribution import (
    BernoulliDistribution,
    DistributionDict,
    DistributionList,
    DistType,
    PoissonDistribution,
)
from stochas.named_value import NamedValueDict
from stochas.unit_system import UnitSystem


def test_with_seed_resets_rng():
    """Verify with_seed returns self, sets the seed, and produces reproducible draws."""
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)
    result = dist.with_seed(42)

    assert result is dist
    assert dist.seed == 42

    s1 = dist.sample(5)
    dist.with_seed(42)
    s2 = dist.sample(5)
    assert np.array_equal(s1, s2)


def test_with_trial_num_resets_rng():
    """Verify with_trial_num returns self, updates trial_num, and changes draws."""
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1, seed=99)

    s1 = dist.sample(5)
    result = dist.with_trial_num(0)

    assert result is dist
    assert dist.trial_num == 0

    s2 = dist.sample(5)
    assert np.array_equal(s1, s2)

    dist.with_trial_num(1)
    s3 = dist.sample(5)
    assert not np.array_equal(s1, s3)


def test_sample_to_named_value_inherits_metadata():
    """Ensure the sampled NamedValue carries over the distribution's metadata fields."""
    dist = NormalDistribution(
        name=DistName("x"),
        mu=0,
        sigma=1,
        category="kinematics",
        source="datasheet",
        display_name="X Position",
        comment="note",
        seed=42,
        trial_num=1,
    )

    nv = dist.sample_to_named_value()

    assert nv.metadata_dict() == dist.metadata_dict()


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


def test_serialization_roundtrip_dict():
    """Verify Pydantic serialization preserves state."""
    named_dict = NamedValueDict[NDArray]()
    dist_dict = DistributionDict()
    NormalDistribution(
        name=DistName("test"), mu=10, sigma=2, seed=123
    ).sample_and_update_dicts(dist_dict=dist_dict, named_value_dict=named_dict, size=2)

    json_data = named_dict.model_dump_json()
    _new_dist_dict = NamedValueDict[NDArray].model_validate_json(json_data)


def test_sample_and_update_dicts_returns_existing_without_force():
    """Ensure a second call without force returns the already-registered NamedValue."""
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1, seed=42, trial_num=1)
    dist_dict = DistributionDict()
    named_dict = NamedValueDict()

    nv1 = dist.sample_and_update_dicts(dist_dict=dist_dict, named_value_dict=named_dict)
    nv2 = dist.sample_and_update_dicts(dist_dict=dist_dict, named_value_dict=named_dict)

    assert nv2 is nv1


def test_sample_and_update_dicts_force_overwrites_and_warns(caplog):
    """Ensure force=True overwrites the existing NamedValue and logs a warning."""
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1, seed=42, trial_num=1)
    dist_dict = DistributionDict()
    named_dict = NamedValueDict()

    nv1 = dist.sample_and_update_dicts(dist_dict=dist_dict, named_value_dict=named_dict)
    nv2 = dist.sample_and_update_dicts(
        dist_dict=dist_dict, named_value_dict=named_dict, force=True
    )

    assert nv2 is not nv1
    assert named_dict["x"] is nv2
    assert "already exists in named_value_list" in caplog.text


def test_distribution_dict_helpers():
    """Verify DistributionDict's conversion and trial-number propagation."""
    dist_dict = DistributionDict()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)
    dist_dict.update(dist)

    dist_list = dist_dict.distribution_list
    assert isinstance(dist_list, DistributionList)
    assert list(dist_list) == [dist]

    dist_dict.set_trial_nums(3)
    assert dist_dict["x"].trial_num == 3


def test_distribution_list_helpers():
    """Verify DistributionList's conversion and trial-number propagation."""
    dist_list = DistributionList()
    dist = NormalDistribution(name=DistName("y"), mu=0, sigma=1)
    dist_list.append(dist)

    dist_dict = dist_list.to_distribution_dict
    assert isinstance(dist_dict, DistributionDict)
    assert dist_dict["y"] is dist

    dist_list.set_trial_nums(2)
    assert dist_list[0].trial_num == 2


@pytest.mark.parametrize(
    "dist, expected_continuous, expected_discrete",
    [
        (NormalDistribution(name=DistName("n"), mu=0.0, sigma=1.0), True, False),
        (UniformDistribution(name=DistName("u"), low=0.0, high=1.0), True, False),
        (PoissonDistribution(name=DistName("p"), lam=2.0), False, True),
        (BernoulliDistribution(name=DistName("b"), p=0.5), False, True),
    ],
)
def test_is_continuous_and_is_discrete(dist, expected_continuous, expected_discrete):
    assert dist.is_continuous is expected_continuous
    assert dist.is_discrete is expected_discrete


def test_dist_type_table_header():
    assert DistType.NORMAL.table_header == "Normal"
    assert DistType.TRUNCATED_NORMAL.table_header == "Truncated Normal"
    assert DistType.LOG_NORMAL.table_header == "Log Normal"


def test_to_tables_single_category(tmp_path):
    """One category produces a subdirectory with one CSV per dist type."""
    dist_dict = DistributionDict()
    dist_dict.update(
        NormalDistribution(name=DistName("a"), mu=0.0, sigma=1.0, category="params")
    )
    dist_dict.update(
        NormalDistribution(name=DistName("b"), mu=5.0, sigma=2.0, category="params")
    )

    dist_dict.to_tables(tmp_path)

    csv_file = tmp_path / "params" / "normal.csv"
    assert csv_file.exists()

    lines = csv_file.read_text().splitlines()
    assert lines[0] == "Name,Units,mu,sigma"
    assert lines[1].startswith("a,")
    assert lines[2].startswith("b,")


def test_to_tables_multiple_categories(tmp_path):
    """Each distinct category value gets its own subdirectory."""
    dist_dict = DistributionDict()
    dist_dict.update(
        NormalDistribution(name=DistName("a"), mu=0.0, sigma=1.0, category="geometry")
    )
    dist_dict.update(
        UniformDistribution(
            name=DistName("b"), low=0.0, high=1.0, category="kinematics"
        )
    )

    dist_dict.to_tables(tmp_path)

    assert (tmp_path / "geometry" / "normal.csv").exists()
    assert (tmp_path / "kinematics" / "uniform.csv").exists()


def test_to_tables_mixed_dist_types_in_category(tmp_path):
    """Mixed dist types in one category produce separate CSV files in the same subdirectory."""
    dist_dict = DistributionDict()
    dist_dict.update(
        NormalDistribution(name=DistName("mass"), mu=1.0, sigma=0.1, category="phys")
    )
    dist_dict.update(
        UniformDistribution(name=DistName("drag"), low=0.1, high=0.5, category="phys")
    )

    dist_dict.to_tables(tmp_path)

    assert (tmp_path / "phys" / "normal.csv").exists()
    assert (tmp_path / "phys" / "uniform.csv").exists()


def test_to_tables_csv_is_flat(tmp_path):
    """Each CSV file contains only a header row and data rows with no section headers."""
    dist_dict = DistributionDict()
    dist_dict.update(
        NormalDistribution(name=DistName("x"), mu=1.0, sigma=0.5, category="props")
    )

    dist_dict.to_tables(tmp_path)

    lines = (tmp_path / "props" / "normal.csv").read_text().splitlines()
    assert lines[0] == "Name,Units,mu,sigma"
    assert len(lines) == 2


def test_to_tables_default_category(tmp_path):
    """Distributions with the default category land in an uncategorized subdirectory."""
    dist_dict = DistributionDict()
    dist_dict.update(NormalDistribution(name=DistName("x"), mu=0.0, sigma=1.0))

    dist_dict.to_tables(tmp_path)

    assert (tmp_path / "uncategorized" / "normal.csv").exists()


def test_to_tables_creates_nested_directory(tmp_path):
    """to_tables creates the output directory and any missing parents."""
    dist_dict = DistributionDict()
    dist_dict.update(
        NormalDistribution(name=DistName("x"), mu=0.0, sigma=1.0, category="test")
    )

    nested = tmp_path / "nested" / "output"
    dist_dict.to_tables(nested)

    assert (nested / "test" / "normal.csv").exists()


def test_to_tables_units_column_shows_unset_when_no_unit(tmp_path):
    """Units column shows 'unset' when the distribution has no unit descriptor."""
    dist_dict = DistributionDict()
    dist_dict.update(
        NormalDistribution(name=DistName("x"), mu=1.0, sigma=0.1, category="props")
    )

    dist_dict.to_tables(tmp_path)

    lines = (tmp_path / "props" / "normal.csv").read_text().splitlines()
    _, units, *_ = lines[1].split(",")
    assert units == "unset"


def test_to_tables_units_column_shows_descriptor_name_when_unit_set(tmp_path):
    """Units column shows the UnitDescriptor name when a unit is attached."""
    u = UnitSystem.si()
    dist_dict = DistributionDict()
    dist_dict.update(
        NormalDistribution(
            name=DistName("mass"), mu=2.5, sigma=0.1, category="props", unit=u.kilogram
        )
    )

    dist_dict.to_tables(tmp_path)

    lines = (tmp_path / "props" / "normal.csv").read_text().splitlines()
    _, units, *_ = lines[1].split(",")
    assert units == "kilogram"


def test_to_tables_csv_param_values_match_distribution(tmp_path):
    """CSV data row values round-trip through the distribution's table_params."""
    dist_dict = DistributionDict()
    dist_dict.update(
        NormalDistribution(
            name=DistName("link_mass"), mu=1.5, sigma=0.1, category="props"
        )
    )

    dist_dict.to_tables(tmp_path)

    lines = (tmp_path / "props" / "normal.csv").read_text().splitlines()
    name, _units, mu, sigma = lines[1].split(",")
    assert name == "link_mass"
    assert float(mu) == pytest.approx(1.5)
    assert float(sigma) == pytest.approx(0.1)


@pytest.mark.parametrize("char", [*list(r'\/:*?"<>|'), "\x00"])
def test_category_rejects_invalid_chars(char):
    """Each filesystem-illegal character raises ValueError on construction."""
    with pytest.raises(ValueError, match="category contains characters invalid"):
        NormalDistribution(
            name=DistName("x"), mu=0.0, sigma=1.0, category=f"bad{char}name"
        )


@pytest.mark.parametrize(
    "category",
    [
        "uncategorized",
        "my-category",
        "my_category",
        "with spaces",
        "v1.0",
        "sensors123",
        "café",
    ],
)
def test_category_accepts_valid_names(category):
    """Legal category names including hyphens, spaces, dots, and unicode do not raise."""
    dist = NormalDistribution(name=DistName("x"), mu=0.0, sigma=1.0, category=category)
    assert dist.category == category


if __name__ == "__main__":
    test_serialization_roundtrip_dict()
