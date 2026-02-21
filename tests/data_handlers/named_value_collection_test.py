import pytest

from process_manager.data_handlers.named_value import (
    NamedValue,
)
from process_manager.data_handlers.named_value_collections import (
    NamedValueDict,
)


def test_register_named_value():
    collection = NamedValueDict()
    nv = NamedValue[int](name="count")

    collection.update(nv)

    assert "count" in collection
    assert len(collection) == 1
    assert collection["count"] is nv


def test_register_duplicate_name_raises():
    collection = NamedValueDict()

    nv1 = NamedValue[int](name="count")
    nv2 = NamedValue[int](name="count")

    collection.update(nv1)

    with pytest.raises(KeyError, match="has already been registered"):
        collection.update(nv2)


def test_get_missing_name_raises():
    collection = NamedValueDict()

    with pytest.raises(KeyError):
        collection["missing"]


def test_get_value_returns_underlying_value():
    collection = NamedValueDict()
    nv = NamedValue[int](name="count")

    collection.update(nv)
    nv.value = 42

    assert collection.get_value("count") == 42


def test_collection_respects_named_value_freeze():
    collection = NamedValueDict()
    nv = NamedValue[int](name="x")

    collection.update(nv)
    nv.value = 10

    with pytest.raises(ValueError):
        nv.value = 20


def test_collection_returns_live_reference():
    collection = NamedValueDict()
    nv = NamedValue[int](name="a")

    collection.update(nv)

    retrieved = collection["a"]
    retrieved.value = 7

    assert nv.value == 7


def test_values_returns_all_named_values():
    collection = NamedValueDict()

    a = NamedValue[int](name="a")
    b = NamedValue[str](name="b")

    collection.update(a)
    collection.update(b)

    values = collection.values()

    assert set(v.name for v in values) == {"a", "b"}


def test_collection_serialization_roundtrip():
    collection = NamedValueDict()

    a = NamedValue[int](name="a")
    b = NamedValue[str](name="b")

    a.value = 1
    b.value = "hello"

    collection.update(a)
    collection.update(b)

    dumped = collection.model_dump_json()
    restored = NamedValueDict.model_validate_json(dumped)

    assert restored.get_value("a") == 1
    assert restored.get_value("b") == "hello"


def test_type_safety_after_deserialization():
    collection = NamedValueDict()
    nv = NamedValue[float](name="x")
    nv.value = 5.0

    collection.update(nv)

    restored = NamedValueDict.model_validate_json(collection.model_dump_json())

    x = restored["x"]
    assert isinstance(x.value, float)

    with pytest.raises(ValueError):
        x.value = "not an int"


if __name__ == "__main__":
    test_type_safety_after_deserialization()
