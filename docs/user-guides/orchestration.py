"""Example: orchestrating a model run with seeds, trials, and overrides."""

import numpy as np

import stochas

# ---8<--- [start: quick_implementation]


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
# ---8<--- [end: quick_implementation]

# ---8<--- [start: transactions]
for attempt in model.transaction(retry_on=ValueError, max_retries=20):
    with attempt:
        # attempt.sample_dist, not model.sample_dist: a retry then draws a
        # fresh value instead of repeating the one that just got rejected
        inner_radius = attempt.sample_dist(
            stochas.NormalDistribution(
                name=stochas.DistName("inner_radius"), mu=1.0, sigma=0.3
            )
        )
        outer_radius = attempt.sample_dist(
            stochas.NormalDistribution(
                name=stochas.DistName("outer_radius"), mu=1.15, sigma=0.3
            )
        )
        if not (outer_radius.value > inner_radius.value):
            raise ValueError("outer_radius must be bigger than inner_radius")

        print(f"Found a valid ring on attempt {attempt.attempt_number}")

# read back from the model, not from the (now-gone) attempt/local variables
print(
    f"inner={model.named['inner_radius'].value}, "
    f"outer={model.named['outer_radius'].value}"
)
# ---8<--- [end: transactions]

# ---8<--- [start: nested_transactions]
for paycheck_attempt in model.transaction(retry_on=ValueError, max_retries=20):
    with paycheck_attempt:
        paycheck = paycheck_attempt.sample_dist(
            stochas.NormalDistribution(
                name=stochas.DistName("paycheck"), mu=1000.0, sigma=50.0
            )
        )

        # bill only has to be a plausible bill on its own, so a rejection
        # only retries the bill, not the paycheck
        for bill_attempt in model.transaction(retry_on=ValueError, max_retries=20):
            with bill_attempt:
                bill = bill_attempt.sample_dist(
                    stochas.NormalDistribution(
                        name=stochas.DistName("bill"), mu=200.0, sigma=50.0
                    )
                )
                if not (50.0 <= bill.value <= 400.0):
                    raise ValueError("bill outside a plausible range")

        # this check depends on both numbers, so rejecting it redraws the
        # paycheck and restarts the bill's transaction from scratch
        balance = paycheck.value - model.named["bill"].value
        if not (balance >= 700.0):
            raise ValueError("balance dips below the minimum reserve")

print(f"paycheck={model.named['paycheck'].value}, bill={model.named['bill'].value}")
# ---8<--- [end: nested_transactions]
