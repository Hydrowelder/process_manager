"""StochasBase, the top-level model for registering distributions and named/design values."""

from __future__ import annotations

import logging
from typing import Any, Self, TypeVar, cast, overload

import numpy as np
from pydantic import BaseModel, Field, SerializeAsAny, model_validator

from stochas.design_variable import (
    AnyDesignValue,
    DesignBool,
    DesignCategorical,
    DesignFloat,
    DesignInt,
    DesignValueDict,
)
from stochas.distribution import (
    NOMINAL_TRIAL_NUM,
    AnyDist,
    Distribution,
    DistributionDict,
)
from stochas.distribution._base import NDArray
from stochas.named_value import NamedValue, NamedValueDict
from stochas.transaction import Transaction
from stochas.unit_system import UnitDescriptor, UnitSystem

logger = logging.getLogger(__name__)

T = TypeVar("T")


class StochasBase(BaseModel):
    trial_num: int = NOMINAL_TRIAL_NUM
    """Trial number identified for this instance of the model."""

    seed: int | None = None
    """Campaign seed for calculating random numbers."""

    design: DesignValueDict = Field(default_factory=DesignValueDict)
    """Registry of parameters for hyperparameter tuning."""

    dists: DistributionDict = Field(default_factory=DistributionDict)
    """Random distributions used to generate the model."""

    named: SerializeAsAny[NamedValueDict[NDArray]] = Field(
        default_factory=NamedValueDict[NDArray]
    )
    """Final 'baked' values from a random draw, global override, or design study."""

    us: UnitSystem | None = None
    """Physical unit system for this model. Set to `UnitSystem.si()` or similar so that values in the generator can be expressed in any unit (e.g. `pos * u.inch` converts inches to meters) and telemetry channels report concrete units instead of abstract Pint dimensions."""

    def sample_dist[T](
        self,
        dist: Distribution[T],
        size: int = 1,
        force: bool = False,
        warn: bool = True,
        reset_rng: bool = True,
        convert_units: bool = True,
    ) -> NamedValue[NDArray[Any, T]]:
        """
        Sets the seed and trial number of the distribution, sample, registers it and the sampled value to the model, and returns the named value.

        If the NamedValue is already registered, the registered named value is returned.

        Args:
            dist (Dist): Distribution to sample and register.
            size (int, optional): Number of samples to take. Will be embedded in the returned NamedValue. Defaults to 1.
            force (bool, optional): Force the sampled value into the NamedValueDict if it already exists. Defaults to False.
            warn (bool, optional): Whether or not to warn if there is a conflict while forcing. Defaults to True.
            reset_rng (bool, optional): Whether or not to reset the seed and trial number for the distribution. Setting the seed and trial number will reset the random number cycle. This will not skip registering the distribution or NamedValues to the MojoModel. If you want pseudorandom number generation, setting to False will require you to manually set the seed and trial number before passing the distribution into `sample_dist`. Defaults to True.
            convert_units (bool, optional): When True (default) and `dist.unit` is set, multiplies the sampled array by `float(dist.unit)` to convert from the distribution's declared unit into the model base unit before registering and returning. The distribution's own parameters (mean, std, etc.) are unaffected, so `to_table()` continues to report values in the declared unit. Defaults to True.

        Returns:
            NamedValue[NDArray]: NamedValue containing the random draw.

        """
        if reset_rng:
            dist.with_seed(self.seed).with_trial_num(self.trial_num)

        # Distribution is abstract; any instance is necessarily one of AnyDist's
        # concrete members, so this cast just bridges the generic call-site type
        # to the closed discriminated union DistributionDict stores.
        concrete_dist = cast(AnyDist, dist)
        if dist.name not in self.dists:
            self.dists.update(concrete_dist)
        elif force:
            self.dists.force_update(concrete_dist)

        nv = dist.sample_to_named_value(
            size=size, convert_units=convert_units, unit_system=self.us
        )

        if nv in self.named and not force:
            # defined with global override or already sampled
            if warn:
                logger.warning(
                    f"NamedValue {nv.name} already registered. Returning it instead of the sampled value."
                )
            return self.named[nv.name]
        elif nv in self.named and force:
            # defined with global override but forced to update
            if warn:
                logger.warning(
                    f"NamedValue {nv.name} already registered. Force setting it to the new value in the registry.",
                )
            self.named.force_update(nv, warn=False)
        else:
            # standard random draw
            self.named.update(nv)
        return nv

    @overload
    def sample_design(
        self,
        dv: DesignFloat,
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> float: ...

    @overload
    def sample_design(
        self,
        dv: DesignInt,
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> int: ...

    @overload
    def sample_design(
        self,
        dv: DesignBool,
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> bool: ...

    @overload
    def sample_design(
        self,
        dv: DesignCategorical[T],
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> T: ...

    def sample_design(
        self,
        dv: AnyDesignValue,
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> Any:
        """
        Registers a design variable and returns its value, converting to model base units if `convert_units=True` and `dv.unit` is a UnitDescriptor.

        The raw design variable (in its declared unit) is always stored in `self.design` for reporting and optimization. When unit conversion applies, a converted `NamedValue` is stored in `self.named` so downstream code reading `self.named` always gets model-unit values.

        Args:
            dv: The design variable to register and sample.
            force (bool, optional): Overwrite an existing entry in `self.named`. Defaults to False.
            warn (bool, optional): Log a warning when using an override or forcing. Defaults to True.
            convert_units (bool, optional): When True (default) and `dv.unit` is a UnitDescriptor, multiplies the value by `float(dv.unit)` before returning and registering in `self.named`. Defaults to True.

        """
        self.design[dv.name] = dv

        if dv.name in self.named and not force:
            if warn:
                logger.info(f"Using override for design variable: {dv.name}")
            val = self.named[dv.name].value
            return val.item() if hasattr(val, "item") else val

        dv_unit = dv.unit
        val = dv.value
        if (
            convert_units
            and isinstance(dv_unit, UnitDescriptor)
            and isinstance(val, (int, float))
        ):
            val = val * float(dv_unit) + dv_unit.offset
            metadata = dv.metadata_dict()
            metadata["unit"] = (
                self.us.base_unit_for(dv_unit.name) if self.us is not None else None
            )
            self.named.update(
                NamedValue(name=dv.name, stored_value=np.asarray(val), **metadata)
            )
        else:
            self.named.update(
                NamedValue(
                    name=dv.name, stored_value=np.asarray(val), **dv.metadata_dict()
                )
            )
        return val

    @model_validator(mode="after")
    def _restore_unit_factors(self) -> Self:
        """Auto-restores UnitDescriptor factors after deserialization. When `u` is present in the serialized JSON, this fires after all fields are set and re-populates the factors that are excluded from serialization."""
        if self.us is not None:
            self.with_unit_system(self.us)
        return self

    def with_unit_system(self, us: UnitSystem) -> Self:
        """Re-resolves all `UnitDescriptor` conversion factors in registered distributions, design variables, and named values against `us`, and sets `self.u = us`. Called automatically on deserialization when `u` is set; also call explicitly to switch unit systems at runtime."""
        self.us = us
        self.dists.update_unit_system(us)
        self.design.update_unit_system(us)
        self.named.update_unit_system(us)
        logger.debug(
            f"unit system updated to {us.__class__.__name__} (length={us.length}, mass={us.mass})"
        )
        return self

    def with_overrides(self, overrides: NamedValueDict[NDArray]) -> Self:
        """
        Sets the NamedValueDict to the provided override.

        This is useful for manually setting some named values to be used.
        """
        self.named = overrides
        return self

    def with_seed(self, seed: int | None = None) -> Self:
        """
        Sets the seed to the provided value.

        This is useful for initializaing the model.
        """
        self.seed = seed
        for _, dist in self.dists.items():
            dist.with_seed(seed)
        return self

    def with_trial_num(self, trial_num: int) -> Self:
        """
        Sets the trial_num to the provided value.

        This is useful for initializaing the model.
        """
        self.trial_num = trial_num
        for _, dist in self.dists.items():
            dist.with_trial_num(trial_num)
        return self

    @property
    def is_nominal(self) -> bool:
        return self.trial_num == NOMINAL_TRIAL_NUM

    @overload
    def transaction(
        self, retry_on: type[Exception], max_retries: int = 10
    ) -> Transaction: ...
    @overload
    def transaction(
        self, retry_on: tuple[type[Exception], ...], max_retries: int = 10
    ) -> Transaction: ...
    def transaction(
        self,
        retry_on: type[Exception] | tuple[type[Exception], ...],
        max_retries: int = 10,
    ) -> Transaction:
        """
        Retries a block of correlated random draws until it succeeds or max_retries is exhausted.

        Returns an iterable of attempts; wrap your sampling code in `with attempt:`, sample through `attempt.sample_dist(...)` (not `self.sample_dist(...)`, see below), and raise one of `retry_on` to signal an invalid draw. Registrations made in `self.dists`, `self.design`, and `self.named` during a failed attempt are rolled back before the next attempt (or before the exception propagates, if it isn't one of `retry_on`).

        `attempt.sample_dist` reseeds a distribution only the first time it's sampled within the transaction; later attempts keep drawing from that distribution's RNG stream instead of resetting it, so retries actually see new values instead of repeating the same rejected draw. `attempt.sample_design` is also transaction-aware (it rolls back with everything else on a rejected attempt), though it needs no reseeding of its own since an optimizer, not stochas, picks a design variable's value. Anything else not wrapped on `attempt` is reachable via `attempt.base`.

        Args:
            retry_on (type[Exception] | tuple[type[Exception], ...]): Exception type(s) that mean "bad draw, retry" rather than a real error.
            max_retries (int, optional): Maximum number of attempts before raising RuntimeError. Defaults to 10.

        Raises:
            RuntimeError: If no attempt succeeds within max_retries.

        Example:
            >>> for attempt in model.transaction(retry_on=(ValueError,), max_retries=10):
            >>>     with attempt:
            >>>         x = attempt.sample_dist(x_dist)
            >>>         y = attempt.sample_dist(y_dist)
            >>>         if x.value + y.value > limit:
            >>>             raise ValueError("x + y exceeds the allowed limit")

        """
        return Transaction.begin(self, retry_on=retry_on, max_retries=max_retries)
