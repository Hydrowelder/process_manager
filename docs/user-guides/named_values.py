"""Example: creating and managing NamedValue containers."""

# ---8<--- [start: basics]
import stochas

# 1. Initialize a container with a name and a value
# The state automatically moves to 'SET'
friction = stochas.NamedValue(
    name=stochas.ValueName("joint_friction"), stored_value=0.25
)

# 2. Use it directly in physics calculations
# No need to call .value - the mixin handles the math
torque_loss = friction * 300

# 3. Manage a group of values
state_registry = stochas.NamedValueDict()
state_registry.update(friction)

# Access the underlying data safely
current_friction = state_registry["joint_friction"]
# ---8<--- [end: basics]

# ---8<--- [start: protection]
# Updating a NamedValue already in the NamedValueDict
new_friction = stochas.NamedValue(
    name=stochas.ValueName("joint_friction"), stored_value=1234
)

try:
    # "joint_friction" is already in state_registry (fails to update)
    state_registry.update(new_friction)
except Exception:
    print("Failed to update registry!")

# force the value in anyway
state_registry.force_update(new_friction)
# ---8<--- [end: protection]
