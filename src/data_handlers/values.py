"""
Module for generating, sorting, and managing named values.
This uses pydantic dataclasses for JSON serialization to avoid overloading system memory.

The module provides a robust framework for managing named values with type safety, serialization,
and state management. It includes classes for individual named values, collections of named values
in both list and hash (dictionary) formats, and utilities for type validation and serialization.

Classes:
    NamedValueState: Enum for tracking the state of named values
    NamedValue: Base class for type-safe named values with state management
    NamedValueHash: Dictionary-like container for managing named values
    NamedValueList: List-like container for managing ordered named values

Types:
    SerializableValue: Union type defining all allowed value types
    T: Generic type variable bound to SerializableValue
"""
from __future__ import annotations

# Standard
from dataclasses import Field
from enum import Enum
from typing import Any, Iterable, List, Type, TypeVar, Union, Generic, ClassVar
from typing import get_origin, get_args, Union, Iterable
try:
    from typing import Self
except:
    from typing_extensions import Self
try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

# External
import json
from numpydantic import NDArray
import numpy as np
import pandas as pd
from pydantic import (
    ConfigDict, 
    InstanceOf, 
    SerializeAsAny, 
    Field, 
    PrivateAttr,
)
# Local
from data_handlers.base import (
    NamedObject, 
    NamedObjectHash, 
    NamedObjectList,
    ObjectRegistry
)
from data_handlers.custom_serde_definitions.pandantic import PandasDataFrame, PandasSeries
from data_handlers.mixins import ArrayDunders

__all__ = [
    'SerializableValue',
    'NamedValueState',
    'NamedValue',
    'NamedValueList',
    'NamedValueHash',
]

SerializableValue = Union[
    PandasDataFrame,
    PandasSeries,
    NDArray,
    Iterable,
    float,
    int,
    bool,
    str,
    Any,  # TODO does this need to be restricted?
    None,
]

T = TypeVar('T', bound=SerializableValue)

class NamedValueState(str, Enum):
    """
    State enumeration for NamedValue objects.
    
    This enum tracks whether a named value has been set or remains unset. Once set,
    values are typically frozen unless explicitly forced to change.
    
    Attributes:
        UNSET: Indicates no value has been set yet
        SET: Indicates value has been set and is frozen
    """
    UNSET = "unset"
    SET = "set"

    def __str__(self) -> str:
        """
        Convert state to string representation.
        
        Returns:
            str: The state value as a string ("unset" or "set")
        """
        return self.value  # Returns just "unset" or "set"

    def __repr__(self) -> str:
        """
        Get string representation for debugging.
        
        Returns:
            str: The state value as a string ("unset" or "set")
        """
        return self.value  # Returns just "unset" or "set"

@ArrayDunders.mixin
class NamedValue(NamedObject, Generic[T]):
    """
    A named value container with type safety and state management.
    
    NamedValue provides a type-safe way to store and manage values with built-in
    state tracking, serialization, and validation. Values can be frozen after
    initial setting to prevent accidental modification.
    
    Type Parameters:
        T: The type of value to store, must be a SerializableValue
    
    Attributes:
        name (str): Unique identifier for the value
        _stored_value (T | NamedValueState): The actual stored value or UNSET state
        _state (NamedValueState): Current state of the value
        _type (type): Runtime type information for validation
        
    Properties:
        value (T): Access or modify the stored value
        
    Example:
        ```python
        # Create a named integer value
        count = NamedValue[int]("item_count")
        count.value = 42  # Sets and freezes the value
        print(count.value)  # Outputs: 42
        count.value = 50  # Raises ValueError - value is frozen
        count.force_set_value(50)  # Allows value change
        ```
    """
    
    _registry_category: ClassVar[str] = "values"
    
    name: str = Field(..., description="Name of the value")
    _stored_value: T | NamedValueState = PrivateAttr(default=NamedValueState.UNSET)  # Changed this
    _state: NamedValueState = PrivateAttr(default=NamedValueState.UNSET)
    _type: type = PrivateAttr()

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        protected_namespaces=()
    )

    # def __init__(self, name: str, value: T | None = None, **data):
    #     """
    #     Initialize a new NamedValue instance.
        
    #     Args:
    #         name (str): Unique identifier for this value
    #         value (T | None, optional): Initial value to store. Defaults to None.
    #         **data: Additional keyword arguments passed to parent class
            
    #     Note:
    #         If value is provided, it will be validated and set immediately.
    #         The value will be frozen after initial setting.
    #     """
    #     data.pop('stored_value', None)
    #     data.pop('_stored_value', None)
        
    #     super().__init__(name=name, **data)
    #     self._type = self._extract_value_type()
    #     object.__setattr__(self, '_stored_value', NamedValueState.UNSET)  # Explicitly set initial value
        
    #     if value is not None:
    #         self.value = value
    
    # def __init__(self, name: str, value: T | None = None, **data):
    #     print(f"Initializing NamedValue with class: {self.__class__}")
    #     print(f"Has __orig_class__: {hasattr(self, '__orig_class__')}")
    #     if hasattr(self, '__orig_class__'):
    #         print(f"__orig_class__: {self.__orig_class__}")
    #         print(f"__orig_class__.__args__: {self.__orig_class__.__args__}")
    #     print(f"Bases: {self.__class__.__bases__}")
    #     for base in self.__class__.__bases__:
    #         if hasattr(base, '__origin__'):
    #             print(f"Base origin: {base.__origin__}")
    #             if hasattr(base, '__args__'):
    #                 print(f"Base args: {base.__args__}")
        
    #     data.pop('stored_value', None)
    #     data.pop('_stored_value', None)
        
    #     super().__init__(name=name, **data)
    #     self._type = self._extract_value_type()
    #     print(f"Extracted type: {self._type}")
    #     object.__setattr__(self, '_stored_value', NamedValueState.UNSET)
        
    #     if value is not None:
    #         self.value = value
    
    def __init__(self, name: str, value: T | None = None, **data):
        # print(f"Initializing NamedValue with class: {self.__class__}")
        # print(f"Class __args__: {getattr(self.__class__, '__args__', None)}")
        # print(f"Bases: {self.__class__.__bases__}")
        # for base in self.__class__.__bases__:
        #     print(f"Base __args__: {getattr(base, '__args__', None)}")
        
        data.pop('stored_value', None)
        data.pop('_stored_value', None)
        
        super().__init__(name=name, **data)
        self._type = self._extract_value_type()
        # print(f"Extracted type: {self._type}")
        object.__setattr__(self, '_stored_value', NamedValueState.UNSET)
        
        if value is not None:
            self.value = value

    @property
    def value(self) -> T:
        """
        Get the stored value.
        
        Returns:
            T: The currently stored value
            
        Raises:
            ValueError: If attempting to access before a value has been set
            
        Note:
            This property provides read access to the stored value. Once set,
            the value is frozen and can only be changed using force_set_value().
        """
        if self._state == NamedValueState.UNSET or self._stored_value is NamedValueState.UNSET:
            raise ValueError(f"Value '{self.name}' has not been set yet.")
        return self._stored_value

    @value.setter
    def value(self, new_value: T):
        """
        Set the value if it hasn't been set before.
        
        Args:
            new_value (T): Value to store
            
        Raises:
            ValueError: If value has already been set (frozen)
            TypeError: If value doesn't match the expected type T
            
        Note:
            Once set, the value becomes frozen and can only be changed
            using force_set_value().
        """
        # print('\n\n\n\n\n\n\nSETTER CALLED!!!!!!\n\n\n\n\n\n')
        if self._state == NamedValueState.SET:
            raise ValueError(
                f"Value '{self.name}' has already been set and is frozen. "
                "Use force_set_value() if you need to override it."
            )
        
        validated_value = self._validate_type(new_value)
        object.__setattr__(self, '_stored_value', validated_value)
        object.__setattr__(self, '_state', NamedValueState.SET)

    def force_set_value(self, new_value: T) -> None:
        """
        Force set the value regardless of its current state.
        
        This method bypasses the normal freezing mechanism and allows
        changing an already-set value.
        
        Args:
            new_value (T): New value to store
            
        Raises:
            TypeError: If value doesn't match the expected type T
        """
        object.__setattr__(self, '_stored_value', NamedValueState.UNSET)
        object.__setattr__(self, '_state', NamedValueState.UNSET)

        # if new_value == 'not a series':
        #     breakpoint()
        self.value = new_value
    
    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Custom serialization to include value state and stored value.
        
        Extends the parent class serialization to include the value's state
        and stored value (if set) in the serialized data.
        
        Args:
            **kwargs: Additional arguments passed to parent serialization
            
        Returns:
            dict[str, Any]: Dictionary containing serialized state
            
        Example:
            ```python
            value = NamedValue("example", 42)
            data = value.model_dump()
            print(data)  # Contains 'state' and 'stored_value'
            ```
        """
        data = super().model_dump(**kwargs)
        data['state'] = self._state
        if self._state == NamedValueState.SET:
            data['stored_value'] = self._stored_value
        return data

    @classmethod
    def model_validate(cls, data: Any) -> NamedValue:
        """
        Custom deserialization to restore value state and stored value.
        
        Reconstructs a NamedValue instance from serialized data, properly
        restoring both the value state and any stored value.
        
        Args:
            data (Any): Serialized data to deserialize
            
        Returns:
            NamedValue: New instance with restored state
            
        Example:
            ```python
            data = {'name': 'example', 'state': 'set', 'stored_value': 42}
            value = NamedValue.model_validate(data)
            print(value.value)  # Outputs: 42
            ```
        """
        if not isinstance(data, dict):
            return super().model_validate(data)

        data_copy = data.copy()
        state = NamedValueState(data_copy.pop('state', NamedValueState.UNSET))
        stored_value = data_copy.pop('stored_value', None)
        
        instance = super().model_validate(data_copy)
        
        # Only set the value if state was SET
        if state == NamedValueState.SET and stored_value is not None:
            instance.force_set_value(stored_value)
        
        return instance

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Prevent direct modification of protected attributes.
        
        Overrides attribute setting to prevent direct modification of internal
        state attributes. These attributes should only be modified through
        appropriate methods.
        
        Args:
            name (str): Name of the attribute to set
            value (Any): Value to set
            
        Raises:
            AttributeError: If attempting to modify protected attributes directly
            
        Example:
            ```python
            value = NamedValue("example")
            value._stored_value = 42  # Raises AttributeError
            ```
        """
        if name in ('_stored_value', '_state'):
            raise AttributeError(f"Cannot modify {name} directly. Use appropriate methods instead.")
        super().__setattr__(name, value)

    def _extract_value_type(self) -> type:
        """
        Extract the type parameter T from the class's generic type information.
        """
        cls = self.__class__

        def parse_type_str(type_str: str) -> type:
            """Parse a type string into an actual type."""
            # Handle Annotated types
            if type_str.startswith('Annotated['):
                # Extract the first type from Annotated[Type, ...]
                inner_types = type_str[len('Annotated['):].split(',')[0].strip()
                return parse_type_str(inner_types)
                
            # Map of type strings to actual types
            TYPE_MAP = {
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'DataFrame': pd.DataFrame,
                'Series': pd.Series,
                'NDArray': np.ndarray,
                'ndarray': np.ndarray,
                'PandasDataFrame': pd.DataFrame,
                'PandasSeries': pd.Series,
            }
            
            return TYPE_MAP.get(type_str, Any)

        # Parse the class name for type information
        name = cls.__qualname__
        if '[' in name:
            # Extract everything between the first [ and the last ]
            type_part = name.split('[', 1)[1].rsplit(']', 1)[0]
            # print(f"Debug - Extracted type part: {type_part}")
            return parse_type_str(type_part)
        
        # Check base classes for inherited types
        for base in cls.__bases__:
            if '[' in base.__qualname__:
                type_part = base.__qualname__.split('[', 1)[1].rsplit(']', 1)[0]
                # print(f"Debug - Extracted type part from base: {type_part}")
                return parse_type_str(type_part)

        return Any

    def _validate_type(self, value: Any) -> T:
        """
        Validate and potentially convert a value to the expected type T.
        """
        expected_type = self._type
        
        # Skip validation for Any type or None
        if expected_type is Any:
            return value
        if expected_type is type(None) and value is None:
            return value

        def type_error(message: str = None) -> TypeError:
            if message:
                return TypeError(message)
            base_type = get_base_type(expected_type)
            type_name = getattr(base_type, '__name__', str(base_type))
            return TypeError(f"Cannot convert {type(value).__name__} to {type_name}")

        def get_base_type(t: type) -> type:
            """Get the base type from an Annotated or other complex type"""
            # Handle Annotated types
            if get_origin(t) is Annotated:
                return get_args(t)[0]
            # Handle other type origins (List, Union, etc)
            if hasattr(t, '__origin__'):
                return t.__origin__
            return t

        def get_actual_type(t: type) -> type:
            """Get the actual type to use for isinstance checks"""
            base = get_base_type(t)
            if base is PandasDataFrame:
                return pd.DataFrame
            if base is PandasSeries:
                return pd.Series
            if base is NDArray:
                return np.ndarray
            return base

        def is_valid_type(value: Any, expected: type) -> bool:
            """Check if value matches the expected type"""
            actual_type = get_actual_type(expected)
            
            # Handle None
            if actual_type is type(None):
                return value is None

            # Handle pandas DataFrame
            if actual_type is pd.DataFrame:
                return isinstance(value, pd.DataFrame)
                
            # Handle pandas Series
            if actual_type is pd.Series:
                return isinstance(value, pd.Series)
                
            # Handle numpy arrays
            if actual_type is np.ndarray:
                return isinstance(value, np.ndarray)
                
            # Handle Iterable (but not strings)
            if actual_type is Iterable and not isinstance(value, str):
                return hasattr(value, '__iter__')
                
            # Handle basic types
            try:
                return isinstance(value, actual_type)
            except TypeError:
                return False

        try:
            actual_type = get_actual_type(expected_type)

            # Handle basic validations first
            if is_valid_type(value, expected_type):
                return value

            # Handle type conversions
            if actual_type is pd.DataFrame:
                try:
                    return pd.DataFrame(value)
                except:
                    raise type_error()
                    
            if actual_type is pd.Series:
                try:
                    return pd.Series(value)
                except:
                    raise type_error()
                    
            if actual_type is np.ndarray:
                try:
                    return np.array(value)
                except:
                    raise type_error()

            if actual_type is int:
                try:
                    if isinstance(value, float):
                        if not value.is_integer():
                            raise type_error("Cannot convert decimal value to integer")
                        return int(value)
                    if isinstance(value, (str, np.integer)):
                        float_val = float(str(value).strip())
                        if not float_val.is_integer():
                            raise type_error("Cannot convert decimal value to integer")
                        return int(float_val)
                    return int(value)
                except (ValueError, TypeError):
                    raise type_error()

            if actual_type is float:
                try:
                    if isinstance(value, (int, float, np.number)):
                        return float(value)
                    if isinstance(value, str):
                        return float(value.strip())
                    raise type_error()
                except ValueError:
                    raise type_error()

            if actual_type is bool:
                if isinstance(value, (bool, np.bool_)):
                    return bool(value)
                raise type_error()

            if actual_type is str:
                return str(value)

            # Try general conversion as last resort
            try:
                result = actual_type(value)
                if is_valid_type(result, expected_type):
                    return result
                raise type_error()
            except (ValueError, TypeError):
                raise type_error()

        except TypeError as e:
            if "Subscripted generics" in str(e):
                # Handle generic type checking errors
                if is_valid_type(value, expected_type):
                    return value
            raise type_error()

    def append_to_value_list(self, l: NamedValueList) -> Self:
        """
        Appends this value instance to a NamedValueList.
        
        Convenience method for adding this value to a list while enabling
        method chaining.
        
        Args:
            l (NamedValueList): The list to append this value to
            
        Returns:
            Self: This instance for method chaining
            
        Example:
            ```python
            value_list = NamedValueList()
            value = NamedValue("example", 42)
            value.append_to_value_list(value_list).force_set_value(43)
            ```
        """
        l.append(self)
        return self

    def register_to_value_hash(self, h: NamedValueHash) -> Self:
        """
        Registers this value instance in a NamedValueHash.
        
        Registers this value in the provided hash container. If the hash contains
        value overrides, this value's current value may be overridden during
        registration.
        
        Args:
            h (NamedValueHash): The hash to register this value in
            
        Returns:
            Self: This instance for method chaining
            
        Example:
            ```python
            value_hash = NamedValueHash()
            value = NamedValue("example", 42)
            value.register_to_value_hash(value_hash).force_set_value(43)
            ```
        """
        h.register_value(self)
        return self

    def model_dump_json(self, **kwargs) -> str:
        """
        Custom JSON serialization of the named value.
        
        Serializes the named value instance to a JSON string, including
        all state information and stored value.
        
        Args:
            **kwargs: JSON serialization options like indent, ensure_ascii, etc.
            
        Returns:
            str: JSON string representation
            
        Example:
            ```python
            value = NamedValue("example", 42)
            json_str = value.model_dump_json(indent=2)
            print(json_str)  # Pretty-printed JSON
            ```
        """
        # Separate JSON-specific kwargs from model_dump kwargs
        json_kwargs = {k: v for k, v in kwargs.items() if k in {'indent', 'ensure_ascii', 'separators'}}
        dump_kwargs = {k: v for k, v in kwargs.items() if k not in json_kwargs}
        
        # Get model data with stored value
        data = self.model_dump(**dump_kwargs)
        return json.dumps(data, **json_kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str, **kwargs) -> NamedValue:
        """
        Custom JSON deserialization to NamedValue instance.
        
        Reconstructs a NamedValue instance from a JSON string representation,
        restoring all state and stored value information.
        
        Args:
            json_data (str): JSON string to deserialize
            **kwargs: Additional validation options
            
        Returns:
            NamedValue: New instance with restored state
            
        Example:
            ```python
            json_str = '{"name": "example", "state": "set", "stored_value": 42}'
            value = NamedValue.model_validate_json(json_str)
            print(value.value)  # Outputs: 42
            ```
        """
        data = json.loads(json_data)
        return cls.model_validate(data, **kwargs)

class NamedValueHash(NamedObjectHash):
    """
    A type-safe dictionary for storing and managing NamedValue objects.
    
    NamedValueHash provides a dictionary-like interface for managing a collection
    of NamedValue instances, using their names as keys. It ensures type safety
    and provides convenient methods for accessing and managing the stored values.
    
    The hash maintains unique naming across all stored values and supports
    serialization/deserialization of the entire collection.
    
    Attributes:
        _registry_category (str): Category identifier for object registration
        model_config (ConfigDict): Pydantic configuration for model behavior
        
    Example:
        ```python
        value_hash = NamedValueHash()
        value_hash.register_value(NamedValue("count", 42))
        print(value_hash.get_raw_value("count"))  # Outputs: 42
        ```
    """
    _registry_category = "values"

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        protected_namespaces=()
    )

    def register_value(self, value: NamedValue) -> Self:
        """
        Register a named value in the hash.
        
        Args:
            value (NamedValue): The value to register
            
        Returns:
            Self: Returns self for method chaining
            
        Raises:
            ValueError: If a value with the same name already exists
            
        Example:
            ```python
            hash = NamedValueHash()
            value = NamedValue("price", 10.99)
            hash.register_value(value).register_value(NamedValue("qty", 5))
            ```
        """
        return self.register_object(value)
    
    def get_value(self, name: str) -> NamedValue:
        """
        Retrieve a named value by its name.
        
        Args:
            name (str): Name of the value to retrieve
            
        Returns:
            NamedValue: The requested named value
            
        Raises:
            KeyError: If no value exists with the given name
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("price", 10.99))
            price = hash.get_value("price")
            print(price.value)  # Outputs: 10.99
            ```
        """
        return self.get_object(name)
    
    def get_values(self) -> Iterable[NamedValue]:
        """
        Get all registered named values.
        
        Returns:
            Iterable[NamedValue]: An iterator over all stored named values
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            hash.register_value(NamedValue("y", 2))
            for value in hash.get_values():
                print(f"{value.name}: {value.value}")
            ```
        """
        return self.get_objects()
    
    def get_value_names(self) -> Iterable[str]:
        """
        Get names of all registered values.
        
        Returns:
            Iterable[str]: An iterator over all value names
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            hash.register_value(NamedValue("y", 2))
            print(list(hash.get_value_names()))  # Outputs: ['x', 'y']
            ```
        """
        return self.get_object_names()
    
    def get_value_by_type(self, value_type: Type) -> Iterable[NamedValue]:
        """
        Get all values of a specific type.
        
        Args:
            value_type (Type): Type to filter values by
            
        Returns:
            Iterable[NamedValue]: Values matching the specified type
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            hash.register_value(NamedValue("name", "test"))
            integers = list(hash.get_value_by_type(int))
            ```
        """
        return [val for val in self.get_values() if isinstance(val, value_type)]
    
    def get_raw_values(self) -> Iterable[Any]:
        """
        Get the underlying values of all named values.
        
        Returns:
            Iterable[Any]: Iterator over the actual values stored in each NamedValue
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            hash.register_value(NamedValue("y", 2))
            print(list(hash.get_raw_values()))  # Outputs: [1, 2]
            ```
        """
        return (val.value for val in self.get_values())
    
    def get_raw_value(self, name: str) -> Any:
        """
        Get the underlying value by name.
        
        Args:
            name (str): Name of the value to retrieve
            
        Returns:
            Any: The actual value stored in the named value
            
        Raises:
            KeyError: If no value exists with the given name
            ValueError: If the value hasn't been set yet
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("price", 10.99))
            print(hash.get_raw_value("price"))  # Outputs: 10.99
            ```
        """
        return self.get_value(name).value
    
    def set_raw_value(self, name: str, value: Any) -> None:
        """
        Set the underlying value for a named value.
        
        Args:
            name (str): Name of the value to update
            value (Any): New value to set
            
        Raises:
            KeyError: If no value exists with the given name
            TypeError: If value type doesn't match the expected type
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("price", 10.99))
            hash.set_raw_value("price", 11.99)
            ```
        """
        self.get_value(name).value = value

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Custom serialization to preserve stored values and their states.
        
        Creates a dictionary representation of the hash that includes full
        serialization of all contained NamedValue objects, preserving their
        values and states.
        
        Args:
            **kwargs: Additional serialization options passed to all nested objects
            
        Returns:
            dict[str, Any]: Dictionary containing the complete hash state
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            data = hash.model_dump()
            print(data['objects']['x']['stored_value'])  # Outputs: 1
            ```
        """
        data = super().model_dump(**kwargs)
        # Ensure each object's stored value is included
        if 'objects' in data:
            for name, obj in self.objects.items():
                if isinstance(obj, NamedValue):
                    # Get the full dump including stored value
                    obj_data = obj.model_dump(**kwargs)
                    data['objects'][name] = obj_data
        return data

    @classmethod
    def model_validate(cls, data: Any) -> NamedValueHash:
        """
        Custom validation to restore hash state from serialized data.
        
        Reconstructs a NamedValueHash instance from serialized data, including
        all contained NamedValue objects with their values and states.
        
        Args:
            data (Any): Serialized data to deserialize. Should be a dictionary
                containing an 'objects' key with serialized NamedValue instances
            
        Returns:
            NamedValueHash: New instance with all values restored
            
        Example:
            ```python
            data = {
                'objects': {
                    'x': {'name': 'x', 'type': 'NamedValue', 'stored_value': 1}
                }
            }
            hash = NamedValueHash.model_validate(data)
            print(hash.get_raw_value('x'))  # Outputs: 1
            ```
        """
        if not isinstance(data, dict):
            return super().model_validate(data)
        
        instance = cls()
        
        # Process each object in the data
        for name, obj_data in data.get('objects', {}).items():
            if isinstance(obj_data, dict):
                obj_type = obj_data.get('type')
                if obj_type:
                    # Get the appropriate class from registry
                    value_class = ObjectRegistry.get(cls._registry_category, obj_type)
                    # Create and validate the object with its stored value
                    value_obj = value_class.model_validate(obj_data)
                    instance.register_value(value_obj)
        
        return instance

    def model_dump_json(self, **kwargs) -> str:
        """
        Custom JSON serialization of the entire hash.
        
        Serializes the NamedValueHash instance and all contained NamedValue
        objects to a JSON string representation. Handles both the hash structure
        and the nested value serialization.
        
        Args:
            **kwargs: JSON serialization options such as:
                - indent: Number of spaces for pretty printing
                - ensure_ascii: Escape non-ASCII characters
                - separators: Tuple of (item_sep, key_sep) for custom formatting
            
        Returns:
            str: JSON string representation of the hash
            
        Example:
            ```python
            hash = NamedValueHash()
            hash.register_value(NamedValue("x", 1))
            json_str = hash.model_dump_json(indent=2)
            print(json_str)  # Pretty-printed JSON with nested values
            ```
        """
        # Separate JSON-specific kwargs from model_dump kwargs
        json_kwargs = {k: v for k, v in kwargs.items() if k in {'indent', 'ensure_ascii', 'separators'}}
        dump_kwargs = {k: v for k, v in kwargs.items() if k not in json_kwargs}
        
        # Get model data
        data = self.model_dump(**dump_kwargs)
        # Serialize to JSON
        return json.dumps(data, **json_kwargs)
    
    @classmethod
    def model_validate_json(cls, json_data: str, **kwargs) -> NamedValueHash:
        """
        Custom JSON deserialization to NamedValueHash instance.
        
        Reconstructs a NamedValueHash instance from a JSON string representation,
        including all contained NamedValue objects with their complete state.
        
        Args:
            json_data (str): JSON string containing serialized hash data
            **kwargs: Additional validation options for nested objects
            
        Returns:
            NamedValueHash: New instance with all values restored
            
        Example:
            ```python
            json_str = '''
            {
                "objects": {
                    "x": {"name": "x", "type": "NamedValue", "stored_value": 1}
                }
            }
            '''
            hash = NamedValueHash.model_validate_json(json_str)
            print(hash.get_raw_value('x'))  # Outputs: 1
            ```
        """
        data = json.loads(json_data)
        return cls.model_validate(data, **kwargs)

class NamedValueList(NamedObjectList):
    """
    An ordered list container for managing NamedValue objects.
    
    NamedValueList maintains an ordered collection of NamedValue objects while
    providing type safety and convenient access methods. It preserves insertion
    order while also allowing access by name.
    
    Attributes:
        _registry_category (str): Category identifier for object registration
        objects (List[SerializeAsAny[InstanceOf[NamedValue]]]): The list of stored values
        
    Example:
        ```python
        value_list = NamedValueList()
        value_list.append(NamedValue("first", 1))
        value_list.append(NamedValue("second", 2))
        print([v.value for v in value_list])  # Outputs: [1, 2]
        ```
    """
    _registry_category = "values"
    
    # Attributes
    objects: List[SerializeAsAny[InstanceOf[NamedValue]]] = Field(default_factory=list)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow',
        protected_namespaces=()
    )

    def append(self, value: NamedValue) -> Self:
        """
        Append a named value to the end of the list.
        
        Args:
            value (NamedValue): Named value to append
            
        Returns:
            Self: The list instance for method chaining
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.append(NamedValue("x", 1)).append(NamedValue("y", 2))
            ```
        """
        return super().append(value)

    def extend(self, values: Iterable[NamedValue]) -> Self:
        """
        Extend the list with multiple named values.
        
        Args:
            values (Iterable[NamedValue]): Collection of named values to add
            
        Returns:
            Self: The list instance for method chaining
            
        Example:
            ```python
            value_list = NamedValueList()
            new_values = [NamedValue("x", 1), NamedValue("y", 2)]
            value_list.extend(new_values)
            ```
        """
        return super().extend(values)

    def __getitem__(self, idx: int) -> NamedValue:
        """Get a named value by its index in the list.

        Args:
            idx (int): Index of the named value to retrieve

        Returns:
            NamedValue: The named value at the specified index

        Raises:
            IndexError: If the index is out of range

        Example:
            ```python
            value_list = NamedValueList()
            value_list.append(NamedValue(name="price", value=10.5))
            first_value = value_list[0] # Get first named value
            ```
        """
        return super().__getitem__(idx)

    def register_value(self, value: NamedValue) -> Self:
        """
        Register a named value to the list.
        
        Similar to append but uses the register_object method internally,
        which may perform additional validation.
        
        Args:
            value (NamedValue): Named value to register
            
        Returns:
            Self: The list instance for method chaining
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.register_value(NamedValue("x", 1))
            ```
        """
        return self.register_object(value)

    def get_value(self, name: str) -> NamedValue:
        """
        Get a registered named value by name.
        
        Args:
            name (str): Name of the value to retrieve
            
        Returns:
            NamedValue: The requested named value
            
        Raises:
            KeyError: If no value exists with the given name
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.append(NamedValue("x", 1))
            x = value_list.get_value("x")
            print(x.value)  # Outputs: 1
            ```
        """
        return self.get_object(name)

    def get_values(self) -> Iterable[NamedValue]:
        """
        Get all registered named values.
        
        Returns:
            Iterable[NamedValue]: Iterator over all stored named values
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.extend([NamedValue("x", 1), NamedValue("y", 2)])
            for value in value_list.get_values():
                print(f"{value.name}: {value.value}")
            ```
        """
        return self.get_objects()

    def get_value_by_type(self, value_type: Type) -> Iterable[NamedValue]:
        """
        Get all values whose stored value is of a specific type.
        
        Args:
            value_type (Type): Type to filter values by
            
        Returns:
            Iterable[NamedValue]: Values whose stored value matches the specified type
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.append(NamedValue("x", 1))
            value_list.append(NamedValue("name", "test"))
            integers = list(value_list.get_value_by_type(int))
            ```
        """
        for value in self.get_values():
            try:
                if isinstance(value.value, value_type):
                    yield value
            except ValueError:
                # Skip unset values
                continue
    
    def model_dump(self, **kwargs) -> dict[str, Any]:
        """
        Custom serialization to preserve stored values.
        
        Extends the parent class serialization to ensure proper serialization
        of all stored named values and their states.
        
        Args:
            **kwargs: Additional serialization options
            
        Returns:
            dict[str, Any]: Dictionary containing serialized state
        """
        data = super().model_dump(**kwargs)
        if 'objects' in data:
            # Ensure each object's stored value is included
            data['objects'] = [
                obj.model_dump(**kwargs) if isinstance(obj, NamedValue) else obj
                for obj in self.objects
            ]
        return data

    @classmethod
    def model_validate(cls, data: Any) -> NamedValueList:
        """
        Custom validation to restore stored values.
        
        Reconstructs a NamedValueList instance from serialized data,
        properly restoring all contained named values and their states.
        
        Args:
            data (Any): Serialized data to deserialize
            
        Returns:
            NamedValueList: New instance with restored values
        """
        if not isinstance(data, dict):
            return super().model_validate(data)
        
        instance = cls()
        
        # Process each object in the data
        for obj_data in data.get('objects', []):
            if isinstance(obj_data, dict):
                obj_type = obj_data.get('type')
                if obj_type:
                    # Get the appropriate class from registry
                    value_class = ObjectRegistry.get(cls._registry_category, obj_type)
                    # Create and validate the object
                    value_obj = value_class.model_validate(obj_data)
                    instance.append(value_obj)
        
        return instance

    def model_dump_json(self, **kwargs) -> str:
        """
        Custom JSON serialization of the value list.
        
        Serializes the NamedValueList instance and all contained NamedValue
        objects to a JSON string representation. Preserves the order of values
        and their complete state.
        
        Args:
            **kwargs: JSON serialization options such as:
                - indent: Number of spaces for pretty printing
                - ensure_ascii: Escape non-ASCII characters
                - separators: Tuple of (item_sep, key_sep) for custom formatting
            
        Returns:
            str: JSON string representation of the list
            
        Example:
            ```python
            value_list = NamedValueList()
            value_list.append(NamedValue("x", 1))
            value_list.append(NamedValue("y", 2))
            json_str = value_list.model_dump_json(indent=2)
            print(json_str)  # Pretty-printed JSON with ordered values
            ```
        """
        # Separate JSON-specific kwargs from model_dump kwargs
        json_kwargs = {k: v for k, v in kwargs.items() if k in {'indent', 'ensure_ascii', 'separators'}}
        dump_kwargs = {k: v for k, v in kwargs.items() if k not in json_kwargs}
        
        # Get model data
        data = self.model_dump(**dump_kwargs)
        # Serialize to JSON
        return json.dumps(data, **json_kwargs)

    @classmethod
    def model_validate_json(cls, json_data: str, **kwargs) -> NamedValueList:
        """
        Custom JSON deserialization to NamedValueList instance.
        
        Reconstructs a NamedValueList instance from a JSON string representation,
        preserving the order of values and restoring their complete state.
        
        Args:
            json_data (str): JSON string containing serialized list data
            **kwargs: Additional validation options for nested objects
            
        Returns:
            NamedValueList: New instance with all values restored in order
            
        Example:
            ```python
            json_str = '''
            {
                "objects": [
                    {"name": "x", "type": "NamedValue", "stored_value": 1},
                    {"name": "y", "type": "NamedValue", "stored_value": 2}
                ]
            }
            '''
            value_list = NamedValueList.model_validate_json(json_str)
            print([v.value for v in value_list])  # Outputs: [1, 2]
            ```
        """
        data = json.loads(json_data)
        return cls.model_validate(data, **kwargs)
