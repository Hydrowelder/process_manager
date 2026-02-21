from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import Field, RootModel

from process_manager.data_handlers.named_value import NamedValue

__all__ = ["NamedValueDict"]


class NamedValueDict(RootModel[dict[str, NamedValue[Any]]]):
    root: dict[str, NamedValue[Any]] = Field(default_factory=dict)

    def __getitem__(self, key: str) -> NamedValue[Any]:
        if key not in self.root:
            raise KeyError(f"NamedValue '{key}' not found.")
        return self.root[key]

    def __setitem__(self, key: str, value: NamedValue[Any]) -> None:
        if key != value.name:
            raise ValueError(f"Key '{key}' must match NamedValue name '{value.name}'")
        self.update(value)

    def __iter__(self) -> Any:
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __contains__(self, name: object) -> bool:
        return name in self.root

    def keys(self):
        return self.root.keys()

    def values(self):
        return self.root.values()

    def items(self):
        return self.root.items()

    def update(self, value: NamedValue[Any]) -> None:
        if value.name in self.root:
            raise KeyError(f"NamedValue {value.name} has already been registered.")

        self.root[value.name] = value

    def update_many(self, values: Iterable[NamedValue[Any]]) -> None:
        for value in values:
            self.update(value=value)

    def get_value(self, name: str) -> Any:
        return self[name].value

    def get_raw_value(self, name: str) -> Any:
        return self[name].stored_value


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
