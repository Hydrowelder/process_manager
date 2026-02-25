from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any, Literal, NewType, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from process_manager.base_collections import BaseDict, BaseList
from process_manager.mixins import NumericMixin

__all__ = [
    "NamedValue",
    "NamedValueDict",
    "NamedValueList",
    "NamedValueState",
    "Val",
    "ValueName",
]

logger = logging.getLogger(__name__)

ValueName = NewType("ValueName", str)
"""Alias of string. Used to type hint a named value's name."""


class Val(BaseModel):
    """Defines a reference to a variable. This is intended for use with variables used as random inputs for Monte Carlo analysis."""

    ref: ValueName


class NamedValueState(StrEnum):
    UNSET = "unset"
    SET = "set"


class NamedValue[T](BaseModel, NumericMixin):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    name: ValueName
    state: NamedValueState = Field(default=NamedValueState.UNSET)
    stored_value: T | Literal[NamedValueState.UNSET] = Field(
        default=NamedValueState.UNSET
    )

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        match self.state:
            case NamedValueState.UNSET:
                if self.stored_value is not NamedValueState.UNSET:
                    self.state = NamedValueState.SET
            case NamedValueState.SET:
                if self.stored_value is NamedValueState.UNSET:
                    msg = f"{self.name} stored value cannot be set to `NamedValueState.UNSET`"
                    logger.error(msg)
                    raise ValueError(msg)
            case _:
                msg = f"The enumeration for {self.state} has not been implemented."
                logger.error(msg)
                raise NotImplementedError(msg)
        return self

    @property
    def value(self) -> T:
        match self.state:
            case NamedValueState.UNSET:
                msg = f"Value for NamedValue {self.name} has not been set."
                logger.error(msg)
                raise ValueError(msg)
            case NamedValueState.SET:
                if self.stored_value is NamedValueState.UNSET:
                    # Defensive: impossible unless model was corrupted
                    msg = f"NamedValue '{self.name}' is set but stored_value is `NamedValueState.SET`"
                    logger.error(msg)
                    raise RuntimeError(msg)
                return self.stored_value
            case _:
                msg = f"The enumeration for {self.state} has not been implemented."
                logger.error(msg)
                raise NotImplementedError(msg)

    @value.setter
    def value(self, value: T) -> None:
        match self.state:
            case NamedValueState.SET:
                msg = f"Value for NamedValue {self.name} has already been set and is frozen."
                logger.error(msg)
                raise ValueError(msg)
            case NamedValueState.UNSET:
                self.force_set_value(value=value, warn=False)
            case _:
                msg = f"The enumeration for {self.state} has not been implemented."
                logger.error(msg)
                raise NotImplementedError(msg)

    def force_set_value(self, value: T, warn: bool = True) -> None:
        if warn:
            logger.warning(f"Forcing value of NamedValue {self.name} to {value}")
        self.stored_value = value
        self.state = NamedValueState.SET

    @property
    def is_set(self) -> bool:
        return self.state is NamedValueState.SET


class NamedValueDict(BaseDict[NamedValue[Any]]):
    """Dictionary specifically for sampled results."""

    def get_value(self, name: ValueName | str) -> Any:
        """Gets the NamedValue value of a key."""
        return self[name].value

    def get_raw_value(self, name: ValueName | str) -> Any | None:
        """Gets the NamedValue raw value. This includes None if the value has not yet been set."""
        return self[name].stored_value

    @property
    def named_value_list(self) -> NamedValueList:
        """Converts the NamedValueDict to a NamedValueList."""
        return NamedValueList(list(self.values()))


class NamedValueList(BaseList[NamedValue[Any]]):
    """List specifically for sampled results."""

    @property
    def to_named_value_dict(self) -> NamedValueDict:
        """Converts the NamedValueList to a NamedValueDict."""
        d = NamedValueDict()
        d.update_many(self.root)
        return d
