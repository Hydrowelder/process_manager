from enum import Enum
from typing import Dict, Type, Set, Optional, Any, List, ClassVar


class SecurityError(Exception):
    """Exception raised for security violations in type imports."""

    pass


class SecurityMode(Enum):
    """Security modes for type resolution."""

    STRICT = "strict"  # Only allow registered types
    ALLOWLIST = "allowlist"  # Allow types from allowlisted modules
    PERMISSIVE = "permissive"  # Allow any import (dangerous!)


class TypeRegistry:
    """Registry of known types that can be safely deserialized."""

    _types: Dict[str, Type] = {}
    _allowed_modules: Set[str] = {
        # Standard library modules that are generally safe
        "builtins",
        "typing",
        "datetime",
        "uuid",
        "decimal",
        "pathlib",
        "collections",
        "enum",
        "dataclasses",
        "numbers",
        "fractions",
        "re",
        "ipaddress",
        "types",
        "zoneinfo",
        "time",
        "json",
    }
    _security_mode: ClassVar[SecurityMode] = SecurityMode.STRICT

    @classmethod
    def register(cls, type_class: Type) -> Type:
        """
        Register a type for safe deserialization.

        Args:
            type_class: The class to register

        Returns:
            The registered class (for use as a decorator)
        """
        key = f"{type_class.__module__}.{type_class.__name__}"
        cls._types[key] = type_class
        return type_class

    @classmethod
    def register_all(cls, type_classes: List[Type]) -> None:
        """
        Register multiple types at once.

        Args:
            type_classes: List of classes to register
        """
        for type_class in type_classes:
            cls.register(type_class)

    @classmethod
    def register_module(cls, module_name: str) -> None:
        """
        Add a module to the allowlist.

        Args:
            module_name: The module name to add to the allowlist
        """
        cls._allowed_modules.add(module_name)

    @classmethod
    def register_modules(cls, module_names: List[str]) -> None:
        """
        Add multiple modules to the allowlist.

        Args:
            module_names: List of module names to add
        """
        cls._allowed_modules.update(module_names)

    @classmethod
    def get(cls, module: str, name: str) -> Optional[Type]:
        """
        Get a registered type by module and name.

        Args:
            module: The module name
            name: The class name

        Returns:
            The registered type or None if not found
        """
        key = f"{module}.{name}"
        return cls._types.get(key)

    @classmethod
    def is_allowed_module(cls, module_name: str) -> bool:
        """
        Check if a module is in the allowlist.

        Args:
            module_name: The module name to check

        Returns:
            True if the module is allowed, False otherwise
        """
        for allowed in cls._allowed_modules:
            if module_name == allowed or module_name.startswith(f"{allowed}."):
                return True
        return False

    @classmethod
    def set_security_mode(cls, mode: SecurityMode) -> None:
        """
        Set the security mode for type imports.

        Args:
            mode: The security mode to set
        """
        cls._security_mode = mode

    @classmethod
    def get_security_mode(cls) -> SecurityMode:
        """
        Get the current security mode.

        Returns:
            The current security mode
        """
        return cls._security_mode

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered types."""
        cls._types.clear()

    @classmethod
    def is_registered(cls, type_class: Type) -> bool:
        """
        Check if a type is registered.

        Args:
            type_class: The class to check

        Returns:
            True if the type is registered, False otherwise
        """
        key = f"{type_class.__module__}.{type_class.__name__}"
        return key in cls._types
