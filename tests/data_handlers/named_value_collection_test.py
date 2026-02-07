import pytest

from process_manager.data_handlers.named_value import (
    NamedValue,
)
from process_manager.data_handlers.named_value_collection import (
    NamedValueCollection,
)


def test_register_named_value():
    collection = NamedValueCollection()
    nv = NamedValue[int](name="count")

    collection.register(nv)

    assert "count" in collection
    assert len(collection) == 1
    assert collection.get("count") is nv


def test_register_duplicate_name_raises():
    collection = NamedValueCollection()

    nv1 = NamedValue[int](name="count")
    nv2 = NamedValue[int](name="count")

    collection.register(nv1)

    with pytest.raises(KeyError, match="has already been registered"):
        collection.register(nv2)


def test_get_missing_name_raises():
    collection = NamedValueCollection()

    with pytest.raises(KeyError):
        collection.get("missing")


def test_get_value_returns_underlying_value():
    collection = NamedValueCollection()
    nv = NamedValue[int](name="count")

    collection.register(nv)
    nv.value = 42

    assert collection.get_value("count") == 42


def test_collection_respects_named_value_freeze():
    collection = NamedValueCollection()
    nv = NamedValue[int](name="x")

    collection.register(nv)
    nv.value = 10

    with pytest.raises(ValueError):
        nv.value = 20


def test_collection_returns_live_reference():
    collection = NamedValueCollection()
    nv = NamedValue[int](name="a")

    collection.register(nv)

    retrieved = collection.get("a")
    retrieved.value = 7

    assert nv.value == 7


def test_values_returns_all_named_values():
    collection = NamedValueCollection()

    a = NamedValue[int](name="a")
    b = NamedValue[str](name="b")

    collection.register(a)
    collection.register(b)

    values = collection.values()

    assert set(v.name for v in values) == {"a", "b"}


def test_collection_serialization_roundtrip():
    collection = NamedValueCollection()

    a = NamedValue[int](name="a")
    b = NamedValue[str](name="b")

    a.value = 1
    b.value = "hello"

    collection.register(a)
    collection.register(b)

    dumped = collection.model_dump_json()
    restored = NamedValueCollection.model_validate_json(dumped)

    assert restored.get_value("a") == 1
    assert restored.get_value("b") == "hello"


def test_type_safety_after_deserialization():
    collection = NamedValueCollection()
    nv = NamedValue[float](name="x")
    nv.value = 5.0

    collection.register(nv)

    restored = NamedValueCollection.model_validate_json(collection.model_dump_json())

    x = restored.get("x")
    assert isinstance(x.value, float)

    with pytest.raises(ValueError):
        x.value = "not an int"


if __name__ == "__main__":
    test_type_safety_after_deserialization()
