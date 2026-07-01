# Unit System: Working Across Unit Conventions

!!! abstract

    The **Unit System** bridges the gap between the unit conventions what you think in (inches, pounds, RPM) and the coherent base units your simulation requires (meters, kilograms, radians/second). By attaching a `UnitSystem` to your model and tagging distributions and design variables with a `unit`, all sampling paths convert values automatically so your input data stays readable while your simulation receives physically consistent numbers.

---

When building physical simulations, engineers often think in the units that come naturally to their domain. One team may use inches and slugs, another team may work in millimeters, and a third could be using centipoise. Manually converting everything to SI before it enters your model is error-prone and obscures intent.

`stochas` solves this with `UnitSystem` and `UnitDescriptor`. A `UnitSystem` knows the base unit for each physical dimension in your model. Any named unit can be resolved against it to produce a conversion factor.

---

## Declaring a Unit System

Four coherent presets cover the most common choices. Each uses a self-consistent mass that makes the natural force unit exactly 1.0 (no `gc` constant needed):

| Factory       | Length      | Mass     | Force     |
|:--------------|:------------|:---------|:----------|
| `si()`        | meter       | kilogram | newton    |
| `cgs()`       | centimeter  | gram     | dyne      |
| `fps()`       | foot        | slug     | lbf       |
| `ips()`       | inch        | slinch   | lbf       |

You can also construct a custom system by providing explicit base unit names:

```python
from stochas.unit_system import UnitSystem

u = UnitSystem(length="meter", mass="kilogram", temperature="kelvin", current="ampere")
```

Attribute access on a `UnitSystem` resolves any [Pint](https://pint.readthedocs.io/) recognized unit (including compound units like `kilometer_per_hour` or `newton_meter`) and returns a `UnitDescriptor`:

```python
--8<-- "docs/user-guides/unit_system.py:declare"
```

---

## Constructing Unit Descriptors

There are four ways to get a `UnitDescriptor` from a `UnitSystem`:

| Approach              | Example                                        | When to use                                                       |
|:----------------------|:-----------------------------------------------|:------------------------------------------------------------------|
| Attribute             | `us.meter`, `us.inch`, `us.pound`              | any Pint-recognized unit name                                     |
| Underscore compound   | `us.meter_per_second`, `us.kilometer_per_hour` | Pint already knows the compound name                              |
| Descriptor arithmetic | `us.m / us.s`, `us.m / us.s**2`, `us.m**2`     | any combination you can express with `*`, `/`, `**`               |
| String subscript      | `us["m/s"]`, `us["inch/second"]`               | expressions with slashes or `**` that can't be Python identifiers |

```python
--8<-- "docs/user-guides/unit_system.py:construct"
```

!!! note

    Offset units (`degF`, `degC`) follow the same four approaches but have restrictions on arithmetic: raising to a power or multiplying/dividing two offset units raises `ValueError` because the result has no unique affine representation. Convert to an absolute unit first (e.g. `us.kelvin` or `us.rankine`) before composing.

---

## Using Unit Descriptors in Expressions

`UnitDescriptor` supports the usual arithmetic operators, so it slots directly into array expressions. Multiply a value **by** the descriptor to convert it from that unit into the model's base unit:

```python
--8<-- "docs/user-guides/unit_system.py:multiply"
```

!!! tip

    This is the same pattern as Pint quantities, but without attaching a unit to the value itself. The `float(descriptor)` factor is computed once at `UnitSystem` construction time and reused for every multiply.

---

## Automatic Conversion in Sampling

The most powerful integration is tagging a distribution or design variable with a `units` descriptor and letting `StochasBase` handle the conversion automatically.

### Distributions

Set the `units` field on any distribution to the `UnitDescriptor` for the unit your parameters are expressed in. When `sample_dist` is called (with the default `convert_units=True`), the sampled array is multiplied by `float(dist.units)` before being stored in `model.named`. The distribution's own parameters (mean, std, bounds) will remain untouched, so `to_tables()` still reports values in the declared unit.

```python
--8<-- "docs/user-guides/unit_system.py:dist_unit"
```

!!! note

    Conversion is skipped automatically for non-numeric distributions (e.g. `CategoricalDistribution` with string choices). A `np.issubdtype` guard ensures no attempt is made to multiply strings or object arrays.

### Design Variables

The same `units` field is available on `DesignFloat` and `DesignInt`. `sample_design` converts the value before returning it and stores the converted `NamedValue` in `model.named`, while `model.design` always retains the original declared value for optimizer feedback and reporting.

```python
--8<-- "docs/user-guides/unit_system.py:design_unit"
```

---

## Serialization and Restoration

`UnitDescriptor` serializes as `{"name": "inch"}` only since the conversion factor is excluded because it is context-dependent (the same name maps to a different factor in SI vs IPS). This keeps serialized models portable.

When deserializing, there are two paths:

- **With `us` in the JSON** (recommended): include `us=UnitSystem.si()` on the model before serializing. On `model_validate_json`, the built-in `model_validator` detects a non-None `us` and calls `update_unit_system` automatically, all factors are restored with no extra code.
- **Without `us` in the JSON**: call `model.update_unit_system(us)` explicitly after loading.

```python
--8<-- "docs/user-guides/unit_system.py:serialization"
```

---

## Built-in Unit Systems at a Glance

```python
from stochas.unit_system import UnitSystem

si  = UnitSystem.si()   # meter / kilogram / second / kelvin / ampere / mole / candela
cgs = UnitSystem.cgs()  # centimeter / gram / second
fps = UnitSystem.fps()  # foot / slug / second
ips = UnitSystem.ips()  # inch / slinch / second
```

Any additional SI base dimensions (temperature, current, amount, luminosity) default to `None` on `cgs`, `fps`, and `ips`. Add them via `model_copy`:

```python
ips_thermal = UnitSystem.ips().model_copy(update={"temperature": "rankine"})
```
