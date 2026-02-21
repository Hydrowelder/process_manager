from __future__ import annotations

from abc import ABC

from pydantic import BaseModel, ConfigDict, Field


class RandomBase(BaseModel, ABC):
    model_config = ConfigDict(
        # validate_assignment=True,
        extra="forbid",
    )

    name: str
    seed: int | None = Field(default=None)
