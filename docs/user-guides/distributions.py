"""Example: sampling distributions and registering results as named values."""

# ---8<--- [start: basics]
from numpydantic import NDArray

import stochas

# 1. Define the rule
motor_rule = stochas.NormalDistribution(
    name=stochas.DistName("motor_torque"),
    mu=5.0,
    sigma=0.2,
)

# 2. Setup the registries
rules = stochas.DistributionDict()
results = stochas.NamedValueDict[NDArray]()

# 3. Sample and Register
# This returns a NamedValue and saves it to 'results'
val_1 = motor_rule.sample_and_update_dicts(
    dist_dict=rules,
    named_value_dict=results,
).squeeze()

# 4. Subsequent calls return the SAME value
val_2 = motor_rule.sample_and_update_dicts(
    dist_dict=rules,
    named_value_dict=results,
).squeeze()

print(val_1.value == val_2.value)  # True
# ---8<--- [end: basics]
# ---8<--- [start: repeat_draw]
rules = stochas.DistributionDict()
results = stochas.NamedValueDict[NDArray]()

friction_rule = (
    stochas.UniformDistribution(
        name=stochas.DistName("friction"),
        low=0.2,
        high=0.4,
    )
    .with_seed(42)  # using set seed
    .with_trial_num(10)  # and set trial_num
)

# These will all be different random numbers
draw_1 = friction_rule.sample_to_named_value().squeeze()  # 0.378
draw_2 = friction_rule.sample_to_named_value().squeeze()  # 0.205
draw_3 = friction_rule.sample_to_named_value().squeeze()  # 0.216

print(draw_1.value, draw_2.value, draw_3.value)  # Three different values

# just be sure to add them to a collection when done!
rules.update(friction_rule)
results.update(draw_3)
# ---8<--- [end: repeat_draw]
