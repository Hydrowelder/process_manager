from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["NamedValue", "NamedValueState"]


class NamedValueState(StrEnum):
    UNSET = "unset"
    SET = "set"


class NamedValue[T](BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    name: str
    state: NamedValueState = Field(default=NamedValueState.UNSET)
    stored_value: T | None = Field(default=None)

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        match self.state:
            case NamedValueState.UNSET:
                if self.stored_value is not None:
                    self.state = NamedValueState.SET
            case NamedValueState.SET:
                if self.stored_value is None:
                    self.state = NamedValueState.UNSET
        return self

    @property
    def value(self) -> T:
        match self.state:
            case NamedValueState.UNSET:
                raise ValueError(f"Value for {self.name} has not been set.")
            case NamedValueState.SET:
                if self.stored_value is None:
                    # Defensive: impossible unless model was corrupted
                    raise RuntimeError(
                        f"NamedValue '{self.name}' is set but stored_value is None"
                    )
                return self.stored_value
            case _:
                raise NotImplementedError(
                    f"The enumeration for {self.state} has not been implemented."
                )

    @value.setter
    def value(self, value: T) -> None:
        match self.state:
            case NamedValueState.SET:
                raise ValueError(
                    f"Value for {self.name} has already been set and is frozen."
                )
            case NamedValueState.UNSET:
                self.force_set_value(value=value)
            case _:
                raise NotImplementedError(
                    f"The enumeration for {self.state} has not been implemented."
                )

    def force_set_value(self, value: T) -> None:
        self.stored_value = value
        self.state = NamedValueState.SET

    @property
    def is_set(self) -> bool:
        return self.state is NamedValueState.SET
