import pytest
from pydantic import ValidationError

from process_manager.data_handlers.named_value import (
    NamedValue,
    NamedValueState,
)


def test_initial_state_unset():
    nv = NamedValue[int](name="x")

    assert nv.state is NamedValueState.UNSET
    assert nv.is_set is False

    with pytest.raises(ValueError, match="has not been set"):
        _ = nv.value


def test_set_value_once():
    nv = NamedValue[int](name="x")

    nv.value = 10

    assert nv.state is NamedValueState.SET
    assert nv.is_set is True
    assert nv.value == 10


def test_double_set_raises():
    nv = NamedValue[int](name="x")
    nv.value = 10

    with pytest.raises(ValueError, match="already been set and is frozen"):
        nv.value = 20


def test_force_set_overwrites():
    nv = NamedValue[int](name="x")
    nv.value = 10

    nv.force_set_value(20)

    assert nv.value == 20
    assert nv.state is NamedValueState.SET


def test_type_enforcement_on_assignment():
    nv = NamedValue[int](name="x")

    with pytest.raises(ValidationError):
        nv.value = "not an int"  # type: ignore[arg-type]


def test_validate_assignment_enforced():
    nv = NamedValue[int](name="x")
    nv.value = 5

    with pytest.raises(ValidationError):
        nv.stored_value = "bad"  # type: ignore[assignment]


def test_serialization_round_trip():
    nv = NamedValue[int](name="x")
    nv.value = 42

    dumped = nv.model_dump_json()
    loaded = NamedValue[int].model_validate_json(dumped)

    assert loaded.name == "x"
    assert loaded.value == 42
    assert loaded.state is NamedValueState.SET


def test_unset_serialization_round_trip():
    nv = NamedValue[int](name="x")

    dumped = nv.model_dump_json()
    loaded = NamedValue[int].model_validate_json(dumped)

    assert loaded.state is NamedValueState.UNSET
    assert loaded.is_set is False

    with pytest.raises(ValueError):
        _ = loaded.value


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        NamedValue[int](name="x", extra_field=123)  # type: ignore[arg-type]


def test_string_named_value():
    nv = NamedValue[str](name="username")
    nv.value = "david"

    assert isinstance(nv.value, str)
    assert nv.value == "david"


def test_generic_type_propagation_for_type_checkers():
    nv = NamedValue[int](name="count")
    nv.value = 7

    def takes_int(x: int) -> None:
        pass

    # This should type-check statically
    takes_int(nv.value)


if __name__ == "__main__":
    test_set_value_once()
