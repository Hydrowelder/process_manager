import pytest

from process_manager import NamedValue, NamedValueDict, NamedValueList


def test_dict_setitem_validation():
    """Ensure __setitem__ enforces the key matches the name."""
    d = NamedValueDict()
    nv = NamedValue[int](name="correct_name")

    # Happy path
    d["correct_name"] = nv

    # Error path: mismatch
    with pytest.raises(ValueError, match="must match name"):
        d["wrong_key"] = nv


def test_dict_force_update():
    """Ensure force_update actually overwrites without error."""
    d = NamedValueDict()
    nv1 = NamedValue[int](name="x", stored_value=1)
    nv2 = NamedValue[int](name="x", stored_value=2)

    d.update(nv1)
    d.force_update(nv2)  # Should not raise KeyError

    assert d.get_value("x") == 2
    assert len(d) == 1


def test_dict_to_list_conversion():
    """Test the @property conversion to NamedValueList."""
    d = NamedValueDict()
    d.update_many(
        [
            NamedValue[int](name="a", stored_value=1),
            NamedValue[int](name="b", stored_value=2),
        ]
    )

    nv_list = d.named_value_list
    assert isinstance(nv_list, NamedValueList)
    assert len(nv_list) == 2
    assert any(x.name == "a" for x in nv_list)


def test_list_indexing_and_slicing():
    """Verify list-like access works for both ints and slices."""
    nv_list = NamedValueList()
    items = [
        NamedValue[int](name="a"),
        NamedValue[int](name="b"),
        NamedValue[int](name="c"),
    ]
    nv_list.extend(items)

    # Integer index
    assert nv_list[0].name == "a"

    # Slice
    subset = nv_list[1:]
    assert isinstance(subset, list)
    assert len(subset) == 2
    assert subset[0].name == "b"


def test_list_delitem_and_pop():
    """Ensure list removal methods work correctly."""
    nv_list = NamedValueList()
    nv = NamedValue[int](name="a")
    nv_list.append(nv)

    assert len(nv_list) == 1

    # Test del
    del nv_list[0]
    assert len(nv_list) == 0

    # Test pop
    nv_list.append(nv)
    popped = nv_list.pop()
    assert popped is nv
    assert len(nv_list) == 0


def test_list_find_by_name():
    """Test the name lookup helper in the list."""
    nv_list = NamedValueList()
    nv = NamedValue[str](name="target", stored_value="hit")
    nv_list.append(nv)

    assert nv_list.find_by_name("target").value == "hit"

    with pytest.raises(KeyError, match="not found in list"):
        nv_list.find_by_name("ghost")


def test_list_to_dict_conversion():
    """Verify ordered list can transform into a keyed dict."""
    nv_list = NamedValueList()
    nv_list.append(NamedValue[int](name="x", stored_value=100))

    d = nv_list.to_named_value_dict
    assert isinstance(d, NamedValueDict)
    assert d.get_value("x") == 100


def test_list_to_dict_duplicate_fail():
    """List to Dict conversion should fail if list contains duplicate names."""
    nv_list = NamedValueList()
    nv_list.append(NamedValue[int](name="dup"))
    nv_list.append(NamedValue[int](name="dup"))  # Valid in list

    with pytest.raises(KeyError, match="already been registered"):
        _ = nv_list.to_named_value_dict


if __name__ == "__main__":
    test_dict_setitem_validation()
