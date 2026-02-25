from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import overload

import numpy as np
from pydantic import Field, RootModel

logger = logging.getLogger(__name__)


class BaseDict[T](RootModel[dict[str, T]]):
    root: dict[str, T] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> T:
        """Get an item in the dictionary with the specified key."""
        if key not in self.root:
            msg = f"'{key}' not found."
            logger.error(msg)
            raise KeyError(msg)
        return self.root[key]

    def __setitem__(self, key: str, value: T) -> None:
        """Set the value of a single key-value pair."""
        name = getattr(value, "name", None)
        if key != name:
            msg = f"Key '{key}' must match name '{name}'"
            logger.error(msg)
            raise ValueError(msg)
        self.update(value)

    def __iter__(self) -> Iterator[str]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Create an iterable object."""
        return self.root.__iter__()

    def __len__(self) -> int:
        """Returns the number of elements in the dictionary."""
        return self.root.__len__()

    def __contains__(self, key: object) -> bool:
        """Specifies if a key already exists in the dictionary."""
        return self.root.__contains__(key)

    def keys(self):
        """Returns the keys in the dictionary."""
        return self.root.keys()

    def values(self):
        """Returns the values in the dictionary."""
        return self.root.values()

    def items(self):
        """Returns the key-value pairs in the dictionary as tuples."""
        return self.root.items()

    def update(self, value: T) -> None:
        """Add a new dictionary key value pair. The key cannot already exist in the dictionary."""
        name = getattr(value, "name", None)
        if name in self.root:
            msg = f"{name} has already been registered."
            logger.error(msg)
            raise KeyError(msg)

        self.force_update(value=value, warn=False)

    def update_many(self, values: Iterable[T]) -> None:
        """Add many new dictionary key value pair. The keys cannot already exist in the dictionary."""
        for value in values:
            self.update(value=value)

    def force_update(self, value: T, warn: bool = True) -> None:
        """Forces a key-value pair into the dictionary. Overwrites existing key if it exists."""
        name = getattr(value, "name")
        if warn:
            logger.warning(f"Forcing {name} into {self.__class__.__name__}.")
        self.root[name] = value

    def force_update_many(self, values: Iterable[T], warn: bool = True) -> None:
        """Forces adding many new dictionary key value pair. Overwrites existing keys if they exist."""
        for value in values:
            self.force_update(value=value, warn=warn)


class BaseList[T](RootModel[list[T]]):
    root: list[T] = Field(default_factory=list)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.array(self.root, dtype=dtype, copy=copy)

    def __len__(self) -> int:
        """Returns the number of elements in the list."""
        return self.root.__len__()

    def __iter__(self) -> Iterator[T]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Create an iterable object."""
        return self.root.__iter__()

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        """Gets the items in the specified index/indices."""
        return self.root[index]

    def __setitem__(self, index: int, value: T) -> None:
        """Set the object at an index."""
        self.root[index] = value

    def __delitem__(self, index: int) -> None:
        """Delete an element from the list."""
        del self.root[index]

    def append(self, value: T) -> None:
        """Append an item to the list."""
        self.root.append(value)

    def extend(self, values: Iterable[T]) -> None:
        """Append many items to the list."""
        self.root.extend(values)

    def pop(self, index: int = -1) -> T:
        """Remove and return item at index (default last)."""
        return self.root.pop(index)

    def find_by_name(self, name: str) -> T:
        """Utility to find a specific named value within the list."""
        for item in self.root:
            item_name = getattr(item, "name", None)
            if item_name == name:
                return item
        msg = f"Item '{name}' not found in list."
        logger.error(msg)
        raise KeyError(msg)


if __name__ == "__main__":
    from process_manager import NamedValue, NamedValueDict, NamedValueList, ValueName

    name = NamedValue[str](name=ValueName("name"), stored_value="john")
    age = NamedValue[int](name=ValueName("age"), stored_value=1)

    named_value_dict = NamedValueDict()
    named_value_dict.update_many([name, age])

    assert name == named_value_dict["name"]
    x: NamedValue[str] = named_value_dict["name"]

    # Not recommended. Converted from NamedValue[str] to NamedValue[Any]
    named_value_list = NamedValueList([name])

    # Recommended. Object retains NamedValue[int]
    named_value_list.append(age)

    breakpoint()
