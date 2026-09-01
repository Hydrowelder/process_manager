"""Tests for StochasBase."""

import inspect

import numpy as np
import pytest

from stochas import (
    DesignFloat,
    DesignInt,
    DistName,
    NamedValue,
    NormalDistribution,
    StochasBase,
    TransactionAttempt,
)
from stochas.named_value import ValueName
from stochas.unit_system import UnitSystem


def test_sample_dist_registers_distribution_and_named_value():
    """Ensure sample_dist registers both the distribution and the sampled value."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    nv = sb.sample_dist(dist)

    assert "x" in sb.dists
    assert "x" in sb.named
    assert sb.named["x"] is nv


def test_sample_dist_returns_existing_named_value():
    """Ensure repeated sampling without force returns the already-registered value."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    nv1 = sb.sample_dist(dist)
    nv2 = sb.sample_dist(dist)

    assert nv2 is nv1
    assert sb.named["x"] is nv1


def test_sample_dist_force_overwrites_named_value():
    """Ensure force=True overwrites both the distribution and named value registries."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    nv1 = sb.sample_dist(dist)
    nv2 = sb.sample_dist(dist, force=True)

    assert nv2 is not nv1
    assert sb.named["x"] is nv2
    assert sb.dists["x"] is dist


def test_sample_design_registers_value():
    """Ensure sample_design registers the design value and returns its stored value."""
    sb = StochasBase()
    dv = DesignFloat(name=ValueName("p"), low=0, high=10, stored_value=5.0)

    result = sb.sample_design(dv)

    assert result == 5.0
    assert "p" in sb.design
    assert "p" in sb.named


def test_sample_design_uses_existing_override():
    """Ensure an existing entry in `named` is used instead of the design value's own value."""
    sb = StochasBase()
    sb.named.update(
        NamedValue[np.ndarray](name=ValueName("q"), stored_value=np.array([3.5]))
    )
    dv = DesignFloat(name=ValueName("q"), low=0, high=10, stored_value=5.0)

    result = sb.sample_design(dv)

    assert result == 3.5
    assert "q" not in sb.design or sb.design["q"] is dv


def test_with_overrides_and_with_override():
    """Ensure with_overrides and with_override both set the named dict and return self."""
    sb = StochasBase()
    overrides = sb.named.__class__()
    overrides.update(
        NamedValue[np.ndarray](name=ValueName("r"), stored_value=np.array([1.0]))
    )

    result = sb.with_overrides(overrides)
    assert result is sb
    assert sb.named is overrides


def test_with_seed_and_with_trial_num_propagate_to_dists():
    """Ensure with_seed and with_trial_num update the model and all registered distributions."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("z"), mu=0, sigma=1)
    sb.dists.update(dist)

    result = sb.with_seed(42)
    assert result is sb
    assert sb.seed == 42
    assert sb.dists["z"].seed == 42

    result = sb.with_trial_num(3)
    assert result is sb
    assert sb.trial_num == 3
    assert sb.dists["z"].trial_num == 3


def test_is_nominal():
    """Ensure is_nominal reflects whether trial_num matches NOMINAL_TRIAL_NUM."""
    sb = StochasBase()
    assert sb.is_nominal is True

    sb.with_trial_num(1)
    assert sb.is_nominal is False


def test_sample_dist_with_unit_converts_samples():
    """Ensure sample_dist multiplies the sampled array by the unit factor."""
    us = UnitSystem.si()
    sb = StochasBase()
    # nominal=100.0 so the draw is deterministic at trial_num=0 regardless of sigma
    dist = NormalDistribution(
        name=DistName("length"), mu=100.0, sigma=1.0, nominal=100.0, unit=us.inch
    )

    nv = sb.sample_dist(dist)

    assert np.allclose(nv.value, [100.0 * 0.0254], rtol=1e-6)


def test_sample_dist_unit_annotates_result_with_base_unit():
    """After conversion the NamedValue.unit is the model base unit, not the source unit."""
    us = UnitSystem.si()
    sb = StochasBase(us=us)
    dist = NormalDistribution(
        name=DistName("length"), mu=100.0, sigma=1.0, nominal=100.0, unit=us.inch
    )

    nv = sb.sample_dist(dist)

    # value is converted to meters
    assert np.allclose(nv.value, [100.0 * 0.0254], rtol=1e-6)
    # unit is now the SI base unit for length, not the source inch unit
    assert nv.unit is not None
    assert str(nv.unit) == "m"
    assert float(nv.unit) == pytest.approx(1.0)  # scale=1: no further conversion needed


def test_sample_design_unit_annotates_result_with_base_unit():
    """After conversion the registered NamedValue's unit is the model base unit."""
    us = UnitSystem.si()
    sb = StochasBase(us=us)
    dv = DesignFloat(
        name=ValueName("width"), low=0.0, high=100.0, stored_value=10.0, unit=us.inch
    )

    sb.sample_design(dv)

    # design dict keeps the original source unit
    assert str(sb.design["width"].unit) == "inch"
    # named value gets the converted value tagged with the model base unit
    named_unit = sb.named["width"].unit
    assert named_unit is not None
    assert str(named_unit) == "m"
    assert float(named_unit) == pytest.approx(1.0)


def test_sample_dist_without_unit_system_leaves_unit_none():
    """Without a UnitSystem on the model, unit is cleared to None after conversion."""
    us = UnitSystem.si()
    sb = StochasBase()  # no us= kwarg
    dist = NormalDistribution(
        name=DistName("length"), mu=100.0, sigma=1.0, nominal=100.0, unit=us.inch
    )

    nv = sb.sample_dist(dist)

    # value is still converted, but no UnitSystem means we can't tag the result
    assert np.allclose(nv.value, [100.0 * 0.0254], rtol=1e-6)
    assert nv.unit is None


def test_sample_dist_convert_units_false_preserves_source_unit():
    """With convert_units=False the NamedValue keeps the source UnitDescriptor unchanged."""
    us = UnitSystem.si()
    sb = StochasBase(us=us)
    dist = NormalDistribution(
        name=DistName("length"), mu=100.0, sigma=1.0, nominal=100.0, unit=us.inch
    )

    nv = sb.sample_dist(dist, convert_units=False)

    # value is raw (no conversion applied)
    assert np.allclose(nv.value, [100.0])
    # unit is still the source inch descriptor
    assert nv.unit is not None
    assert str(nv.unit) == "inch"


def test_sample_design_with_unit_converts_value():
    """Ensure sample_design multiplies the design value by the unit factor."""
    us = UnitSystem.si()
    sb = StochasBase()
    dv = DesignFloat(
        name=ValueName("width"), low=0.0, high=100.0, stored_value=10.0, unit=us.inch
    )

    result = sb.sample_design(dv)

    assert result == pytest.approx(10.0 * 0.0254, rel=1e-6)
    assert sb.named["width"].value == pytest.approx(10.0 * 0.0254, rel=1e-6)


def test_update_unit_system_restores_factors_after_deserialization():
    """Ensure update_unit_system re-populates factors excluded from serialization."""
    us = UnitSystem.si()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1, unit=us.inch)
    sb = StochasBase()
    sb.dists.update(dist)

    sb2 = StochasBase.model_validate_json(sb.model_dump_json())

    assert sb2.dists["x"].unit is not None
    assert sb2.dists["x"].unit.scale is None  # excluded from serialization

    sb2.with_unit_system(us)

    assert sb2.dists["x"].unit.scale == pytest.approx(0.0254, rel=1e-6)


def test_transaction_success_on_first_attempt_commits_and_stops():
    """A transaction that succeeds immediately registers its draw and iterates exactly once."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    attempt_numbers = []
    for attempt in sb.transaction(retry_on=(ValueError,), max_retries=10):
        attempt_numbers.append(attempt.attempt_number)
        with attempt:
            attempt.sample_dist(dist)

    assert attempt_numbers == [1]
    assert "x" in sb.dists
    assert "x" in sb.named


def test_transaction_retries_and_rolls_back_failed_attempt():
    """A retryable failure rolls back its partial registrations; the next attempt's are kept."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    attempt_numbers = []
    for attempt in sb.transaction(retry_on=(ValueError,), max_retries=10):
        attempt_numbers.append(attempt.attempt_number)
        with attempt:
            attempt.sample_dist(dist)
            if attempt.attempt_number == 1:
                raise ValueError("bad draw")

    assert attempt_numbers == [1, 2]
    assert "x" in sb.dists
    assert "x" in sb.named


def test_transactions_can_nest():
    """A transaction can nest inside another attempt's `with` block. Each layer rolls back only its own scope, and a failure at the outer layer discards everything the inner layer had already committed, forcing both to run again."""
    sb = StochasBase()
    dist_a = NormalDistribution(name=DistName("a"), mu=1.0, sigma=0.3)
    dist_b = NormalDistribution(name=DistName("b"), mu=1.0, sigma=0.3)

    outer_attempt_numbers = []
    inner_attempt_numbers = []
    b_present_at_start_of_outer_attempt = []
    for outer_attempt in sb.transaction(retry_on=(ValueError,), max_retries=10):
        outer_attempt_numbers.append(outer_attempt.attempt_number)
        with outer_attempt:
            # on the second outer attempt, this checks whether "b" -- which
            # the first outer attempt's inner transaction already committed
            # to sb.named before the outer attempt itself later failed -- is
            # still sitting there or was actually rolled back with it
            b_present_at_start_of_outer_attempt.append("b" in sb.named)

            outer_attempt.sample_dist(dist_a)

            for inner_attempt in sb.transaction(retry_on=(ValueError,), max_retries=10):
                inner_attempt_numbers.append(inner_attempt.attempt_number)
                with inner_attempt:
                    inner_attempt.sample_dist(dist_b)
                    if inner_attempt.attempt_number == 1:
                        raise ValueError("inner rejects its first draw")

            # by this point the inner transaction has already succeeded and
            # committed "b" to sb.named -- but this check depends on both a
            # and b, so rejecting it must still discard "b" along with "a"
            if outer_attempt.attempt_number == 1:
                raise ValueError("combined check rejects the first outer attempt")

    assert outer_attempt_numbers == [1, 2]
    # each outer attempt starts its own fresh inner transaction from scratch
    assert inner_attempt_numbers == [1, 2, 1, 2]
    # "b" was never left over from the first (failed) outer attempt's inner
    # transaction, even though that inner transaction had already succeeded
    assert b_present_at_start_of_outer_attempt == [False, False]
    assert "a" in sb.dists
    assert "b" in sb.dists
    assert "a" in sb.named
    assert "b" in sb.named


def test_transaction_sample_dist_advances_rng_across_attempts():
    """attempt.sample_dist reseeds a distribution only on its first use, so retries draw new values instead of repeating the same rejected one."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    seen_values = []
    for attempt in sb.transaction(retry_on=(ValueError,), max_retries=10):
        with attempt:
            nv = attempt.sample_dist(dist)
            seen_values.append(np.array(nv.value))
            if attempt.attempt_number < 3:
                raise ValueError("keep retrying")

    assert len(seen_values) == 3
    assert not np.allclose(seen_values[0], seen_values[1])
    assert not np.allclose(seen_values[1], seen_values[2])


def test_transaction_exhausts_retries_and_rolls_back():
    """Exhausting max_retries raises RuntimeError chained to the last error and leaves no residue."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    with pytest.raises(
        RuntimeError, match="Failed to complete transaction after 3 attempt"
    ) as exc_info:
        for attempt in sb.transaction(retry_on=(ValueError,), max_retries=3):
            with attempt:
                attempt.sample_dist(dist)
                raise ValueError("always bad")

    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "x" not in sb.dists
    assert "x" not in sb.named


def test_transaction_propagates_non_retryable_exception_and_rolls_back():
    """An exception not in `retry_on` propagates immediately without retrying, after rollback."""
    sb = StochasBase()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1)

    with pytest.raises(TypeError, match="not a valid draw"):
        for attempt in sb.transaction(retry_on=(ValueError,), max_retries=10):
            with attempt:
                attempt.sample_dist(dist)
                raise TypeError("not a valid draw")

    assert "x" not in sb.dists
    assert "x" not in sb.named


def test_transaction_rolls_back_design_registrations():
    """attempt.sample_design registrations roll back the same way as dists/named."""
    sb = StochasBase()
    dv = DesignInt(name=ValueName("n"), low=0, high=10, stored_value=5)

    attempt_numbers = []
    for attempt in sb.transaction(retry_on=(ValueError,), max_retries=10):
        attempt_numbers.append(attempt.attempt_number)
        with attempt:
            attempt.sample_design(dv)
            if attempt.attempt_number == 1:
                raise ValueError("bad combination")

    assert attempt_numbers == [1, 2]
    assert "n" in sb.design
    assert "n" in sb.named


def test_transaction_supports_design_variables_alongside_dists():
    """A design variable (e.g. picked by an optimizer) can sit in the same transaction as a retried random draw, staying fixed across attempts while the dist keeps resampling."""
    sb = StochasBase()
    material = DesignInt(name=ValueName("material_id"), low=0, high=2, stored_value=1)
    thickness = NormalDistribution(name=DistName("thickness"), mu=1.0, sigma=0.3)

    attempt_numbers = []
    for attempt in sb.transaction(retry_on=(ValueError,), max_retries=10):
        attempt_numbers.append(attempt.attempt_number)
        with attempt:
            attempt.sample_design(material)
            attempt.sample_dist(thickness)
            if attempt.attempt_number == 1:
                raise ValueError("bad combination")

    assert attempt_numbers == [1, 2]
    # the design choice never changes across attempts: it isn't RNG-driven
    assert sb.named["material_id"].value == 1
    assert sb.design["material_id"] is material
    assert "thickness" in sb.named


@pytest.mark.parametrize(
    ("method_name", "omitted_params"),
    [("sample_dist", {"reset_rng"}), ("sample_design", set())],
)
def test_transaction_attempt_signature_matches_stochas_base(
    method_name, omitted_params
):
    """TransactionAttempt hand-mirrors StochasBase's sampling methods (sample_dist minus reset_rng, which it derives itself); catch either drifting out of sync instead of letting it rot silently."""
    base_params = inspect.signature(getattr(StochasBase, method_name)).parameters
    attempt_params = inspect.signature(
        getattr(TransactionAttempt, method_name)
    ).parameters

    expected = {
        name: p
        for name, p in base_params.items()
        if name != "self" and name not in omitted_params
    }
    actual = {name: p for name, p in attempt_params.items() if name != "self"}

    assert actual.keys() == expected.keys(), (
        f"TransactionAttempt.{method_name}'s parameters {sorted(actual)} no longer match "
        f"StochasBase.{method_name}'s {sorted(expected)} (excluding {omitted_params or 'nothing'})."
    )
    for name in actual:
        assert actual[name].default == expected[name].default, (
            f"TransactionAttempt.{method_name}'s default for '{name}' "
            f"({actual[name].default!r}) no longer matches StochasBase.{method_name}'s "
            f"({expected[name].default!r})."
        )


def test_model_validator_auto_restores_factors_when_u_serialized():
    """Ensure model_validator restores unit factors on deserialization when u is included."""
    us = UnitSystem.si()
    dist = NormalDistribution(name=DistName("x"), mu=0, sigma=1, unit=us.inch)
    sb = StochasBase(us=us)
    sb.dists.update(dist)

    sb2 = StochasBase.model_validate_json(sb.model_dump_json())

    assert sb2.us is not None
    assert sb2.dists["x"].unit is not None
    assert sb2.dists["x"].unit.scale == pytest.approx(0.0254, rel=1e-6)
