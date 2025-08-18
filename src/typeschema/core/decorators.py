from typing import Type, TypeVar
from pydantic import computed_field

from typeflow.serialization.type_info import TypeInfo

T = TypeVar("T")


def preserve_schema(cls: Type[T]) -> Type[T]:
    """
    Decorator that adds type preservation capabilities to Pydantic models.

    This decorator adds a type_info computed field to ensure type information
    is included during serialization.

    Args:
        cls: The Pydantic model class to enhance

    Returns:
        The enhanced class with type preservation capabilities
    """
    # Add version attribute if not present
    if not hasattr(cls, "version"):
        cls.version = "1.0.0"

    # Add _type_key if not present
    if not hasattr(cls, "_type_key"):
        cls._type_key = None

    @classmethod
    def get_type_key(class_self) -> str:
        """Get the type key used for type identification in serialized data."""
        if class_self._type_key is not None:
            return class_self._type_key
        return class_self.__name__

    # Add type_info as a computed field
    @computed_field
    def type_info(self):
        """Returns the type information for this model class."""
        return TypeInfo.from_obj(self)

    # Add methods to the class
    cls.get_type_key = get_type_key
    # Add computed_field
    if not hasattr(cls, "type_info"):
        cls.type_info = type_info

    return cls
