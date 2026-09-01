"""Transaction, a retryable, all-or-nothing block of correlated random draws for StochasBase."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, TypeVar, overload

from numpydantic import NDArray

from stochas.design_variable import (
    AnyDesignValue,
    DesignBool,
    DesignCategorical,
    DesignFloat,
    DesignInt,
)
from stochas.distribution import Distribution
from stochas.named_value import NamedValue

if TYPE_CHECKING:
    from stochas.base import StochasBase

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Transaction:
    """
    Iterable of retry attempts returned by `StochasBase.transaction()`.

    Construct via `Transaction.begin()`, which snapshots `base.dists`, `base.design`, and `base.named`. A failed attempt (retryable or not) rolls back to that snapshot before the exception is either suppressed (to retry) or allowed to propagate, so a transaction only ever commits the registrations made by the attempt that succeeds.
    """

    base: StochasBase = field(repr=False)
    retry_on: type[Exception] | tuple[type[Exception], ...]
    max_retries: int
    _attempts: int = field(default=0, init=False)
    _done: bool = field(default=False, init=False)
    _last_exception: BaseException | None = field(default=None, init=False)
    _seeded_names: set[str] = field(default_factory=set, init=False)
    _dists_snapshot: dict[str, Any] = field(default_factory=dict, init=False)
    _design_snapshot: dict[str, Any] = field(default_factory=dict, init=False)
    _named_snapshot: dict[str, Any] = field(default_factory=dict, init=False)

    @overload
    @classmethod
    def begin(
        cls, base: StochasBase, retry_on: type[Exception], max_retries: int = 10
    ) -> Self: ...
    @overload
    @classmethod
    def begin(
        cls,
        base: StochasBase,
        retry_on: tuple[type[Exception], ...],
        max_retries: int = 10,
    ) -> Self: ...
    @classmethod
    def begin(
        cls,
        base: StochasBase,
        retry_on: type[Exception] | tuple[type[Exception], ...],
        max_retries: int = 10,
    ) -> Self:
        """Starts a transaction, snapshotting `base.dists`, `base.design`, and `base.named` for rollback."""
        transaction = cls(base=base, retry_on=retry_on, max_retries=max_retries)
        transaction._dists_snapshot = dict(base.dists.root)
        transaction._design_snapshot = dict(base.design.root)
        transaction._named_snapshot = dict(base.named.root)
        return transaction

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> TransactionAttempt:
        if self._done:
            raise StopIteration
        if self._attempts >= self.max_retries:
            msg = f"Failed to complete transaction after {self.max_retries} attempt(s)."
            logger.error(msg)
            raise RuntimeError(msg) from self._last_exception
        return TransactionAttempt(self)

    def _handle_attempt_result(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None
    ) -> bool:
        if exc_type is None:
            self._done = True
            return False

        # roll back this attempt's partial registrations before deciding what to do next
        self.base.dists.root = dict(self._dists_snapshot)
        self.base.design.root = dict(self._design_snapshot)
        self.base.named.root = dict(self._named_snapshot)

        if not issubclass(exc_type, self.retry_on):
            return False

        self._attempts += 1
        self._last_exception = exc_value
        if self._attempts > self.max_retries // 2:
            logger.warning(
                f"High rejection rate detected for transaction. Attempt {self._attempts}/{self.max_retries}. Latest error: {exc_value}"
            )
        return True


@dataclass
class TransactionAttempt:
    """One attempt of a `StochasBase.transaction()`; use as `with attempt:`."""

    transaction: Transaction

    @property
    def base(self) -> StochasBase:
        """The `StochasBase` model this transaction is retrying against: the one `.transaction()` was called on. Use it for anything not wrapped below, e.g. `attempt.base.with_seed(...)` or `attempt.base.model_dump()`."""
        return self.transaction.base

    @property
    def attempt_number(self) -> int:
        """1-indexed number of this attempt."""
        return self.transaction._attempts + 1

    def sample_dist[T](
        self,
        dist: Distribution[T],
        size: int = 1,
        force: bool = False,
        warn: bool = True,
        convert_units: bool = True,
    ) -> NamedValue[NDArray[Any, T]]:
        """
        Samples `dist` via `attempt.base` (the model this transaction is retrying against), reseeding it only the first time it's sampled within this transaction.

        Without this, every attempt would reset the distribution's RNG to the same state and draw the exact same (rejected) value forever. `reset_rng` is therefore not exposed here; it is derived from whether this transaction has already seeded `dist.name`.
        """
        reset_rng = dist.name not in self.transaction._seeded_names
        self.transaction._seeded_names.add(dist.name)
        return self.transaction.base.sample_dist(
            dist,
            size=size,
            force=force,
            warn=warn,
            reset_rng=reset_rng,
            convert_units=convert_units,
        )

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
        Samples `dv` via `attempt.base` (the model this transaction is retrying against).

        Design variables aren't RNG-backed (an optimizer, not stochas, picks their value), so there's no reseeding to manage here: this is a plain passthrough to `StochasBase.sample_design`, kept on `TransactionAttempt` so a design variable stays fixed for the duration of one attempt and rolls back with everything else on a rejected one, exactly like `sample_dist`.
        """
        return self.transaction.base.sample_design(
            dv, force=force, warn=warn, convert_units=convert_units
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        return self.transaction._handle_attempt_result(exc_type, exc_value)
