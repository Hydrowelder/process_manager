from typing import Optional, ClassVar
from pydantic import BaseModel
from typeschema import TypeSchemaMixin, preserve_schema
import json


# Example using the mixin approach
class PersonMixin(TypeSchemaMixin, BaseModel):
    """A person model using the TypeSchemaMixin approach"""

    version: ClassVar[str] = "1.0.0"

    full_name: str
    age: int
    email: Optional[str] = None


# Example using the decorator approach
@preserve_schema
class PersonDecorated(BaseModel):
    """A person model using the @preserve_schema decorator approach"""

    version: ClassVar[str] = "1.0.0"

    full_name: str
    age: int
    email: Optional[str] = None


def main():
    # Create instances
    person_mixin = PersonMixin(full_name="John Doe", age=30, email="john@example.com")
    person_decorated = PersonDecorated(full_name="Jane Smith", age=28)

    # Serialize to dict with type information using Pydantic's model_dump
    mixin_dict = person_mixin.model_dump()
    decorated_dict = person_decorated.model_dump()

    print("Mixin serialization:")
    print(json.dumps(mixin_dict, indent=2))
    print("\nDecorator serialization:")
    print(json.dumps(decorated_dict, indent=2))

    # Serialize to JSON directly using Pydantic's model_dump_json
    mixin_json = person_mixin.model_dump_json(indent=2)
    decorated_json = person_decorated.model_dump_json(indent=2)

    print("\nMixin JSON serialization:")
    print(mixin_json)
    print("\nDecorator JSON serialization:")
    print(decorated_json)

    # Deserialize back to objects using Pydantic's model_validate
    restored_mixin = PersonMixin.model_validate(mixin_dict)
    restored_decorated = PersonDecorated.model_validate(decorated_dict)

    print("\nRestored mixin:", restored_mixin)
    print("Restored decorated:", restored_decorated)

    # Demonstrate deserializing from JSON using Pydantic's model_validate_json
    restored_mixin_from_json = PersonMixin.model_validate_json(mixin_json)
    restored_decorated_from_json = PersonDecorated.model_validate_json(decorated_json)

    print("\nRestored mixin from JSON:", restored_mixin_from_json)
    print("Restored decorated from JSON:", restored_decorated_from_json)


if __name__ == "__main__":
    main()
