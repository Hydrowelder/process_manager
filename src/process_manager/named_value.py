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
    """
    Defines a reference to a variable name.

    This acts as a 'pointer' or 'placeholder' within other models. During processing, these references would be replaced by actual values sampled from their corresponding distributions.
    """

    ref: ValueName
    """The unique identifier for the variable to be resolved."""


class NamedValueState(StrEnum):
    """Internal state of a NamedValue."""

    UNSET = "unset"
    """The value has been initialized but not yet sampled/populated."""

    SET = "set"
    """The value has been populated and is effectively frozen."""


class NamedValue[T](BaseModel, NumericMixin):
    """
    A container for a sampled variable that tracks its initialization state.

    NamedValues act as the bridge between abstract distributions and concrete simulation parameters. They utilize a state-machine logic to ensure that values are not accidentally overwritten once sampled, and provide a NumericMixin to allow the container to behave like the underlying data in mathematical operations.

    Notes:
        Once `state` becomes `SET`, the `value` property will raise an error
        on further assignment attempts unless `force_set_value` is used.

    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    name: ValueName
    """The unique identifier for this parameter."""

    state: NamedValueState = Field(default=NamedValueState.UNSET)
    """The current lifecycle state (SET or UNSET)."""

    stored_value: T | Literal[NamedValueState.UNSET] = Field(
        default=NamedValueState.UNSET
    )
    """The actual data held by this container."""

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        """Synchronizes the state enum with the actual stored_value content."""
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
        """
        Returns the stored value if it has been set.

        Raises:
            ValueError: If the state is UNSET.
            RuntimeError: If the internal state is corrupted.

        """
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
        """
        Sets the stored value and transitions state to SET.

        Raises:
            ValueError: If the value is already SET (frozen).

        """
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
        """
        Manually overrides the stored value regardless of current state.

        Args:
            value: The new value to store.
            warn: If True, logs a warning about the override.

        """
        if warn:
            logger.warning(f"Forcing value of NamedValue {self.name} to {value}")
        self.stored_value = value
        self.state = NamedValueState.SET

    @property
    def is_set(self) -> bool:
        """True if the value has been populated."""
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
