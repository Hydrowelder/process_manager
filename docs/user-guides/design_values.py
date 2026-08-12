"""Example: defining and optimizing design variables with Optuna."""

import logging

import numpy as np
import optuna

import stochas

# Setup logging
logging.basicConfig(level=logging.INFO)


class SystemModel(stochas.StochasBase):
    """A generic system model that uses design variables."""

    pass


def generate_system(model: SystemModel):
    # 1. Define a continuous float (e.g., structural stiffness)
    # The optimizer will explore between 50 and 500.
    stiffness = model.sample_design(
        stochas.DesignFloat(
            name=stochas.ValueName("k_stiffness"),
            low=50.0,
            high=500.0,
            stored_value=150.0,  # the initial guess
        )
    )

    # 2. Define an integer count (e.g., number of supports)
    n_supports = model.sample_design(
        stochas.DesignInt(
            name=stochas.ValueName("n_supports"), low=2, high=8, stored_value=4
        )
    )

    # 3. Define a categorical choice (e.g., material type)
    material = model.sample_design(
        stochas.DesignCategorical(
            name=stochas.ValueName("material"),
            choices=["aluminum", "steel", "titanium"],
            stored_value="steel",
        )
    )

    print(
        f"Generated System: {material} frame, {n_supports} supports, k={stiffness:.2f}"
    )


# --- Optimization Workflow ---

# 1. Create a model
model = SystemModel()

# 2. Discovery Pass: Map the design space
generate_system(model)
design_space = model.design

# 3. Translate to Optuna
study = optuna.create_study(direction="minimize")


def objective(trial):
    # Ask variables to suggest values to Optuna
    for dv in design_space.values():
        dv.to_optuna(trial)

    # In a real run, you'd apply these suggestions to the model.named registry
    # and execute your physics/cost simulation here.
    score = np.random.random()  # Placeholder for simulation result
    return score


study.optimize(objective, n_trials=10)
