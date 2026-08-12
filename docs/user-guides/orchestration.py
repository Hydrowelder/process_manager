"""Example: orchestrating a model run with seeds, trials, and overrides."""

import numpy as np

import stochas


class MyModel(stochas.StochasBase):
    """Put your logic for running your model here!"""


overrides = stochas.NamedValueDict()
overrides.update(
    stochas.NamedValue(
        name=stochas.ValueName("overridden_value"), stored_value=np.array([3.14])
    )
)

model = MyModel().with_seed(42).with_trial_num(1).with_overrides(overrides)

# 1. Random draw
noise = model.sample_dist(
    stochas.NormalDistribution(
        name=stochas.DistName("sensor_noise"), nominal=0, mu=0, sigma=0.1
    )
)

# 2. Tunable parameter
width = model.sample_design(
    stochas.DesignFloat(
        name=stochas.ValueName("base_width"), low=1.0, high=5.0, stored_value=2.5
    )
)

# Both are now registered in model.named for later analysis.

# fails to update named value dict! overridden_value is already in the named value dict
overridden_value = model.sample_dist(
    stochas.PoissonDistribution(name=stochas.DistName("overridden_value"), lam=4)
)
assert overridden_value == np.array([3.14])  # this check passes!
