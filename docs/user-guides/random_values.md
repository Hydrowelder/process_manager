# Distributions Guide

This guide outlines how to use the `process_manager` distribution system to sample parameters, manage random seeds, and integrate results into your simulation via [`NamedValueDict`][process_manager.named_value.NamedValueDict].

## Overview

The distribution system is built for **reproducibility** and **type safety**. Every distribution inherits from a base [`Distribution[T]`][process_manager.distribution.Distribution] class that handles internal seeding logic. This ensures that your Monte Carlo trials are repeatable-even if you add or remove parameters from your model-by using a combination of a global seed, the distribution's unique name, and the current run number.

---

## Core Usage Pattern

The typical workflow involves defining your distributions, collecting them into a [`DistributionDict`][process_manager.distribution.DistributionDict], and then "sampling" them into a [`NamedValueDict`][process_manager.named_value.NamedValueDict].

### 1. Defining Distributions
Each distribution requires a [`name`][process_manager.distribution.Distribution.name] and its specific parameters. You should also provide a [`seed`][process_manager.distribution.Distribution.seed] if you want reproducible results.

```python
from process_manager import NormalDistribution, DistName

# Define a parameter for a robot link mass
mass_dist = NormalDistribution(
    name=DistName("link_mass"),
    mu=1.5,
    sigma=0.15,
    seed=42,
)
```

### 2. The Sampling Workflow
To keep track of multiple parameters, we use the [`DistributionDict`][process_manager.distribution.DistributionDict]. When you call [`update_dicts`][process_manager.distribution.Distribution.update_dicts], the distribution draws a sample and registers it into a [`NamedValueDict`][process_manager.named_value.NamedValueDict].

```python
from process_manager import DistributionDict, NamedValueDict

dists = DistributionDict()
results = NamedValueDict()

# Sample 1 value and register it into the dictionaries
mass_dist.update_dicts(dists, results, size=1)

# Access the sampled value via the dictionary or the returned NamedValue
print(f"Sampled Mass: {results.get_value('link_mass')}")
```

### 3. Managing Monte Carlo Trials
The [`run_num`][process_manager.distribution.Distribution.run_num] attribute acts as a "salt" for the random seed. Changing the `run_num` allows you to generate a new set of independent samples while maintaining the ability to recreate any specific trial later.

```python
# Set up for trial #5
dist_dict.set_run_nums(5)

# Sampling now produces values unique to trial 5
trial_5_mass = mass_dist.sample()
```

---

## Integration with Named Values

When a distribution is sampled, it creates a [`NamedValue`][process_manager.named_value.NamedValue]. These objects are the "glue" that connects your abstract distributions to your physical models.

* **`NamedValue`**: Holds the actual data (like a NumPy array of 1.5kg).
* **`NamedValueDict`**: A collection of these values, keyed by their [`DistName`][process_manager.distribution.DistName].


---

## The "Nominal" Case
If you define a [`nominal`][process_manager.distribution.Distribution.nominal] value, the distribution will return that value whenever `run_num == 0` (defined as `NOMINAL_RUN_NUM`), bypassing the random draw entirely. This is useful for "Golden Run" simulations where you want to test the system with default values.

```python
dist = NormalDistribution(name="test", mu=10, sigma=1, nominal=10.0)

# run_num 0 returns exactly 10.0
# run_num 1+ returns a random draw around 10.0
```

---

!!! success
    [Click here to see all the supported distribution types][process_manager.distribution].
