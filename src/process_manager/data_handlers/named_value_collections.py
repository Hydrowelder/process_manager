from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any, overload

from pydantic import Field, RootModel

from process_manager.data_handlers.named_value import NamedValue

__all__ = ["NamedValueDict", "NamedValueList"]

logger = logging.getLogger(__name__)


class NamedValueDict(RootModel[dict[str, NamedValue[Any]]]):
    root: dict[str, NamedValue[Any]] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> NamedValue[Any]:
        """Get an item in the dictionary with the specified key."""
        if key not in self.root:
            msg = f"NamedValue '{key}' not found."
            logger.error(msg)
            raise KeyError(msg)
        return self.root[key]

    def __setitem__(self, key: str, value: NamedValue[Any]) -> None:
        """Set the value of a single key-value pair."""
        if key != value.name:
            msg = f"Key '{key}' must match NamedValue name '{value.name}'"
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

    def update(self, value: NamedValue[Any]) -> None:
        """Add a new dictionary key value pair. The key cannot already exist in the dictionary."""
        if value.name in self.root:
            msg = f"NamedValue {value.name} has already been registered."
            logger.error(msg)
            raise KeyError(msg)

        self.force_update(value=value, warn=False)

    def update_many(self, values: Iterable[NamedValue[Any]]) -> None:
        """Add many new dictionary key value pair. The keys cannot already exist in the dictionary."""
        for value in values:
            self.update(value=value)

    def force_update(self, value: NamedValue[Any], warn: bool = True) -> None:
        """Forces a key-value pair into the dictionary. Overwrites existing key if it exists."""
        if warn:
            logger.warning(f"Forcing {value.name} into dictionary.")
        self.root[value.name] = value

    def force_update_many(
        self, values: Iterable[NamedValue[Any]], warn: bool = True
    ) -> None:
        """Forces adding many new dictionary key value pair. Overwrites existing keys if they exist."""
        for value in values:
            self.force_update(value=value, warn=warn)

    def get_value(self, name: str) -> Any:
        """Gets the NamedValue value of a key."""
        return self[name].value

    def get_raw_value(self, name: str) -> Any | None:
        """Gets the NamedValue raw value. This includes None if the value has not yet been set."""
        return self[name].stored_value

    @property
    def named_value_list(self) -> NamedValueList:
        """Converts the NamedValueDict to a NamedValueList."""
        return NamedValueList(list(self.values()))


class NamedValueList(RootModel[list[NamedValue[Any]]]):
    root: list[NamedValue[Any]] = Field(default_factory=list)

    def __len__(self) -> int:
        """Returns the number of elements in the list."""
        return self.root.__len__()

    def __iter__(self) -> Iterator[NamedValue[Any]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Create an iterable object."""
        return self.root.__iter__()

    @overload
    def __getitem__(self, index: int) -> NamedValue[Any]: ...
    @overload
    def __getitem__(self, index: slice) -> list[NamedValue[Any]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> NamedValue[Any] | list[NamedValue[Any]]:
        """Gets the NamedValues in the specified index/indices."""
        return self.root[index]

    def __setitem__(self, index: int, value: NamedValue[Any]) -> None:
        """Set the object at an index."""
        self.root[index] = value

    def __delitem__(self, index: int) -> None:
        """Delete an element from the list."""
        del self.root[index]

    def append(self, value: NamedValue[Any]) -> None:
        """Append a NamedValue to the list."""
        self.root.append(value)

    def extend(self, values: Iterable[NamedValue[Any]]) -> None:
        """Append many NamedValues to the list."""
        self.root.extend(values)

    def pop(self, index: int = -1) -> NamedValue[Any]:
        """Remove and return item at index (default last)."""
        return self.root.pop(index)

    def find_by_name(self, name: str) -> NamedValue[Any]:
        """Utility to find a specific named value within the list."""
        for item in self.root:
            if item.name == name:
                return item
        msg = f"NamedValue '{name}' not found in list."
        logger.error(msg)
        raise KeyError(msg)

    @property
    def to_named_value_dict(self) -> NamedValueDict:
        d = NamedValueDict()
        d.update_many(self.root)
        return d


if __name__ == "__main__":
    name = NamedValue[str](name="name", stored_value="john")
    age = NamedValue[int](name="age", stored_value=1)

    named_value_dict = NamedValueDict()
    named_value_dict.update_many([name, age])

    for key in named_value_dict.keys():
        print(type(named_value_dict[key]) is NamedValue[int])
        type(named_value_dict[key])

    assert name == named_value_dict["name"]
    x: NamedValue[str] = named_value_dict["name"]
    breakpoint()
