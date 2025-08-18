from typing import ClassVar, Dict, Any, Type, TypeVar, Optional, get_type_hints
from pydantic import BaseModel, computed_field

from typeschema.core.type_info import TypeInfo

T = TypeVar("T", bound="TypeSchemaMixin")


class TypeSchemaMixin:
    """
    Mixin that adds type preservation capabilities to Pydantic models.

    This mixin adds a type_info computed field to ensure type information
    is included during serialization.
    """

    version: ClassVar[str] = "1.0.0"
    _type_key: ClassVar[str] = None  # Optional custom type key

    @classmethod
    def get_type_key(cls) -> str:
        """Get the type key used for type identification in serialized data."""
        if cls._type_key is not None:
            return cls._type_key
        return cls.__name__

    @computed_field
    def type_info(self) -> TypeInfo:
        """Returns the type information for this model class."""
        # Use the TypeInfo.from_obj method that already exists
        return TypeInfo.from_obj(self)
