"""
Defines numeric mixins so that classes with a value attribute can act like a number.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "NumericMixin",
]


@runtime_checkable
class HasValue(Protocol):
    """The contract: Any class using the mixin must have a .value property."""

    @property
    def value(self) -> Any: ...

    def _extract(self, other) -> Any: ...


class NumericMixin:
    """
    A mixin that proxies operations to the .value property.

    This allows a NamedValue to behave like a float, int, str, or NDArray.
    """

    # --- NumPy Masquerade ---
    # These allow you to call NamedValue.shape just like a real array
    @property
    def shape(self: HasValue) -> tuple[int, ...]:
        return self.value.shape

    @property
    def ndim(self: HasValue) -> int:
        return self.value.ndim

    @property
    def dtype(self: HasValue) -> Any:
        return self.value.dtype

    @property
    def size(self: HasValue) -> int:
        return self.value.size

    # --- Type Casting ---
    def __int__(self: HasValue) -> int:
        return int(self.value)

    def __float__(self: HasValue) -> float:
        return float(self.value)

    def __str__(self: HasValue) -> str:
        return str(self.value)

    def __bool__(self: HasValue) -> bool:
        return bool(self.value)

    # --- Array & Sequence Behavior ---
    def __len__(self: HasValue) -> int:
        return len(self.value)

    def __getitem__(self: HasValue, key):
        return self.value[key]

    def __iter__(self: HasValue):
        return iter(self.value)

    def __contains__(self: HasValue, item: Any) -> bool:
        return item in self.value

    # This allows np.array(named_value) to work
    def __array__(self: HasValue, dtype=None, copy=None):
        return np.array(self.value, dtype=dtype, copy=copy)

    # --- Math Operations ---
    # We use a helper to extract the raw value from 'other'
    # if 'other' is also a NamedValue.
    def _extract(self, other):
        return getattr(other, "value", other)

    def __add__(self: HasValue, other):
        return self.value + self._extract(other)

    def __sub__(self: HasValue, other):
        return self.value - self._extract(other)

    def __mul__(self: HasValue, other):
        return self.value * self._extract(other)

    def __truediv__(self: HasValue, other):
        return self.value / self._extract(other)

    def __pow__(self: HasValue, other):
        return self.value ** self._extract(other)

    def __matmul__(self: HasValue, other):
        return self.value @ self._extract(other)

    # Reverse operations (e.g., 5.0 + NamedValue)
    def __radd__(self: HasValue, other):
        return self._extract(other) + self.value

    def __rsub__(self: HasValue, other):
        return self._extract(other) - self.value

    def __rmul__(self: HasValue, other):
        return self._extract(other) * self.value

    def __rtruediv__(self: HasValue, other):
        return self._extract(other) / self.value

    # --- Comparisons ---
    def __lt__(self: HasValue, other):
        return self.value < self._extract(other)

    def __le__(self: HasValue, other):
        return self.value <= self._extract(other)

    def __gt__(self: HasValue, other):
        return self.value > self._extract(other)

    def __ge__(self: HasValue, other):
        return self.value >= self._extract(other)

    def __eq__(self: HasValue, other):
        return self.value == self._extract(other)
