"""Design variables (optimizable NamedValue subclasses) and their dict/list collections."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast

import numpy as np
from pydantic import Field, model_validator
from pymoo.core.variable import Binary, Choice, Integer, Real

from stochas.base_collections import BaseDict, BaseList, HasUnitsCollection
from stochas.mixins import MetadataMixin
from stochas.named_value import NamedValue, ValueName

if TYPE_CHECKING:
    import optuna

__all__ = [
    "DesignBool",
    "DesignCategorical",
    "DesignFloat",
    "DesignInt",
    "DesignValueDict",
    "DesignValueList",
]

logger = logging.getLogger(__name__)


class OptimizationSuggestor(ABC, MetadataMixin):
    @abstractmethod
    def to_optuna(self, trial: optuna.Trial) -> Any:
        pass

    @abstractmethod
    def to_pymoo(self) -> Any: ...

    @abstractmethod
    def refine(self, factor: float, best_params: dict[str, Any]) -> None:
        pass


class DesignBool(NamedValue[bool], OptimizationSuggestor):
    """A specific 0/1 toggle, useful for enabling/disabling branches."""

    type: Literal["binary"] = "binary"

    def to_optuna(self, trial: optuna.Trial) -> bool:
        return trial.suggest_categorical(self.name, [True, False])

    def to_pymoo(self) -> Binary:
        return Binary()

    def refine(self, factor: float, best_params: dict[str, Any]):
        logger.debug(f"Refinement not applicable for binary '{self.name}'.")


class DesignCategorical[T](NamedValue[T], OptimizationSuggestor):
    """A NamedValue that represents a categorical design parameter to be optimized."""

    type: Literal["categorical"] = "categorical"

    choices: Sequence[T]
    """For categorical search (e.g., choices=['stiff', 'soft'])."""

    def to_optuna(self, trial: optuna.Trial):
        return trial.suggest_categorical(
            name=self.name,
            choices=self.choices,  # type: ignore
        )

    def to_pymoo(self) -> Choice:
        return Choice(options=np.array(self.choices, dtype=object))

    def refine(self, factor: float, best_params: dict[str, Any]):
        logger.debug(f"Skipping refinement for categorical '{self.name}'.")


class DesignFloat(NamedValue[float], OptimizationSuggestor):
    """
    A NamedValue that represents a floating point design parameter to be optimized.

    Contains bounds required for search space discovery.
    """

    type: Literal["float"] = "float"

    low: float
    """Lower bound for hyperparameter tuning."""

    high: float
    """Upper bound for hyperparameter tuning."""

    log: bool = False
    """Use support for log-scale search (useful for learning rates or stiffness)."""

    step: float | None = None
    """Discretize the search space (e.g., step=0.5)."""

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.high <= self.low:
            msg = f"DesignFloat '{self.name}': high must be greater than low"
            logger.error(msg)
            raise ValueError(msg)
        if self.log and self.low <= 0:
            msg = f"DesignFloat '{self.name}': low must be greater than 0 when log=True"
            logger.error(msg)
            raise ValueError(msg)
        if self.step is not None and self.step <= 0:
            msg = f"DesignFloat '{self.name}': step must be greater than 0"
            logger.error(msg)
            raise ValueError(msg)
        return self

    def to_optuna(self, trial: optuna.Trial):
        return trial.suggest_float(
            name=self.name,
            low=self.low,
            high=self.high,
            log=self.log,
            step=self.step,
        )

    def to_pymoo(self) -> Real:
        return Real(bounds=(self.low, self.high))

    def refine(self, factor: float, best_params: dict[str, Any]):
        best_val = cast(float, best_params[self.name])
        orig_range = self.high - self.low
        new_half_range = (orig_range * factor) / 2

        self.low = max(self.low, best_val - new_half_range)
        self.high = min(self.high, best_val + new_half_range)

        logger.debug(f"Refined float '{self.name}': [{self.low:.4f}, {self.high:.4f}]")


class DesignInt(NamedValue[int], OptimizationSuggestor):
    """
    A NamedValue representing an integer design parameter (e.g., number of solver iterations).
    """

    type: Literal["int"] = "int"

    low: int
    """Lower bound (inclusive)."""

    high: int
    """Upper bound (inclusive)."""

    step: int = 1
    """Spacing between values."""

    log: bool = False
    """Whether to sample from a log-scale (useful for orders of magnitude)."""

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.high <= self.low:
            msg = f"DesignInt '{self.name}': high must be greater than low"
            logger.error(msg)
            raise ValueError(msg)
        if self.log and self.low <= 0:
            msg = f"DesignInt '{self.name}': low must be greater than 0 when log=True"
            logger.error(msg)
            raise ValueError(msg)
        if self.step <= 0:
            msg = f"DesignInt '{self.name}': step must be greater than 0"
            logger.error(msg)
            raise ValueError(msg)
        return self

    def to_optuna(self, trial: optuna.Trial) -> int:
        return trial.suggest_int(
            name=self.name,
            low=self.low,
            high=self.high,
            step=self.step,
            log=self.log,
        )

    def to_pymoo(self) -> Integer:
        return Integer(bounds=(self.low, self.high))

    def refine(self, factor: float, best_params: dict[str, Any]):
        best_val = cast(float, best_params[self.name])
        orig_range = self.high - self.low
        new_half_range = (orig_range * factor) / 2

        self.low = int(np.floor(max(self.low, best_val - new_half_range)))
        self.high = int(np.ceil(min(self.high, best_val + new_half_range)))

        logger.debug(f"Refined int '{self.name}': [{self.low:.4f}, {self.high:.4f}]")


AnyDesignValue = Annotated[
    DesignFloat | DesignCategorical | DesignInt | DesignBool,
    Field(discriminator="type"),
]


class DesignValueDict(BaseDict[AnyDesignValue], HasUnitsCollection):
    """Dictionary specifically for sampled results."""

    def __contains__(self, key: object) -> bool:
        if isinstance(key, NamedValue):
            key = key.name
        return super().__contains__(key=key)

    def get_value(self, name: ValueName | str) -> Any:
        """Gets the NamedValue value of a key."""
        return self[name].value

    def get_raw_value(self, name: ValueName | str) -> Any | None:
        """Gets the NamedValue raw value. This includes None if the value has not yet been set."""
        return self[name].stored_value

    @property
    def named_value_list(self) -> DesignValueList:
        """Converts the DesignValueDict to a DesignValueList."""
        return DesignValueList(list(self.values()))


class DesignValueList(BaseList[AnyDesignValue]):
    """List specifically for sampled results."""

    @property
    def to_named_value_dict(self) -> DesignValueDict:
        """Converts the DesignValueList to a DesignValueDict."""
        d = DesignValueDict()
        d.update_many(self.root)
        return d
