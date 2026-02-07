from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from process_manager.data_handlers.named_value import NamedValue

__all__ = ["NamedValueCollection"]


class NamedValueCollection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objects: dict[str, NamedValue[Any]] = Field(default_factory=dict)

    def register(self, value: NamedValue[Any]) -> None:
        if value.name in self.objects:
            raise KeyError(f"NamedValue {value.name} has already been registered.")

        self.objects[value.name] = value

    def register_many(self, values: Iterable[NamedValue[Any]]) -> None:
        for value in values:
            self.register(value=value)

    def get(self, name: str) -> NamedValue[Any]:
        try:
            return self.objects[name]
        except KeyError:
            raise KeyError(f"NamedValue {name} was not found in the registry.")

    def get_value(self, name: str) -> Any:
        return self.get(name).value

    def get_raw_value(self, name: str) -> Any:
        # for logger print a warning
        return self.get(name).stored_value

    def values(self) -> Iterable[NamedValue[Any]]:
        return self.objects.values()

    def items(self):
        return self.objects.items()

    def __contains__(self, name: str) -> bool:
        return name in self.objects

    def __len__(self) -> int:
        return len(self.objects)
