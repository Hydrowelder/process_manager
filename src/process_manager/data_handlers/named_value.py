from __future__ import annotations

import logging
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["NamedValue", "NamedValueState"]

logger = logging.getLogger(__name__)


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
                msg = f"Value for {self.name} has not been set."
                logger.error(msg)
                raise ValueError(msg)
            case NamedValueState.SET:
                if self.stored_value is None:
                    # Defensive: impossible unless model was corrupted
                    msg = f"NamedValue '{self.name}' is set but stored_value is None"
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
                msg = f"Value for {self.name} has already been set and is frozen."
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
            logger.warning(f"Forcing value of {self.name} to {value}")
        self.stored_value = value
        self.state = NamedValueState.SET

    @property
    def is_set(self) -> bool:
        return self.state is NamedValueState.SET
