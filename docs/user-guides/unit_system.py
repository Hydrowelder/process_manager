"""Example of how to use the UnitSystem class."""

# ---8<--- [start: declare]
import stochas
from stochas.unit_system import UnitSystem

# declare SI as the model's base unit system
us = UnitSystem.si()

# float() gives the conversion factor (FROM that unit INTO the base unit)
assert float(us.meter) == 1.0
assert float(us.inch) == 0.0254
assert float(us.kilometer) == 1000.0

# str() gives the unit name back as a string
assert str(us.inch) == "inch"

# base dimension names are plain attributes
assert us.length == "m"
assert us.mass == "kg"
# ---8<--- [end: declare]

# ---8<--- [start: multiply]
import numpy as np

us = UnitSystem.si()

# scalar multiply
mass_kg = 2.205 * us.pound  # 2.205 lb * 0.4536 kg/lb ≈ 1.0 kg
print(f"mass: {mass_kg:.4f} kg")

# or since u tracks its own base dimensions
print(f"mass: {2.205 * us.pound:.4f} {us.mass}")  # prints mass: 1.0 kilogram

# array multiply: descriptor on the left so pyright resolves the return type as ndarray
pos_m = us.inch * np.array([1.0, 2.0, 3.0])
print(f"pos: {pos_m} m")  # [0.0254, 0.0508, 0.0762]
# ---8<--- [end: multiply]

# ---8<--- [start: construct]
us = UnitSystem.si()

# attribute access: any Pint-recognized unit name
assert float(us.meter) == 1.0
assert float(us.inch) == 0.0254
assert abs(float(us.pound) - 0.45359237) < 1e-9  # kg per pound

# Pint underscore names for compound units Pint already knows
assert float(us.meter_per_second) == 1.0
assert float(us.kilometer_per_hour) == 1 / 3.6  # km/h -> m/s

# descriptor arithmetic: *, /, ** compose name and scale
velocity = us.m / us.s  # UnitDescriptor("m / s",      scale=1.0)
accel = us.m / us.s**2  # UnitDescriptor("m / s ** 2", scale=1.0)
area = us.m**2  # UnitDescriptor("m ** 2",     scale=1.0)

# scales compose for non-SI too
inch_per_sec = us.inch / us.second  # scale = 0.0254 / 1.0 = 0.0254
assert abs(float(inch_per_sec) - 0.0254) < 1e-12

# string subscript: any Pint expression (slashes, **, spaces all OK)
assert float(us["m/s"]) == 1.0
assert float(us["inch/second"]) == 0.0254  # us.inch_per_second raises AttributeError
assert float(us["m/s**2"]) == 1.0
# ---8<--- [end: construct]

# ---8<--- [start: dist_unit]
us = UnitSystem.si()

# sample_dist automatically converts inches -> meters and tags the result with the model base unit
model = stochas.StochasBase().with_seed(42).with_trial_num(1).with_unit_system(us)
arm_length = model.sample_dist(
    stochas.NormalDistribution(
        name=stochas.DistName("arm_length"),
        mu=12.0,  # 12 inches mean
        sigma=0.1,  # 0.1 inch std
        unit=us.inch,  # attach a UnitDescriptor to a distribution
    ),
    convert_units=True,  # defaults to True
).squeeze()

# value is now in meters
print(f"arm_length (m): {arm_length.value}")  # ~ 0.305 m

# unit reflects the model base unit for length (not the source inch unit)
assert arm_length.unit is not None
assert str(arm_length.unit) == "m"  # SI length base unit
assert float(arm_length.unit) == 1.0  # scale=1 means no further conversion needed
# ---8<--- [end: dist_unit]

# ---8<--- [start: design_unit]
us = UnitSystem.si()

model = stochas.StochasBase(us=us)

link_velocity = stochas.DesignFloat(
    name=stochas.ValueName("link_velocity"),
    low=0.5,
    high=4.0,
    stored_value=2.0,  # declared in inches per second
    unit=us.inch / us.second,  # combine units with /, *, or **
)

vel_m_per_s = model.sample_design(
    link_velocity,
    convert_units=True,  # also defaults to True
)
assert abs(vel_m_per_s - 0.0508) < 1e-9  # 2.0 in/s * 0.0254 m/in = 0.0508 m/s

# model.design keeps the original declared unit
assert str(model.design["link_velocity"].unit) == "inch / second"

# model.named holds the converted value tagged with the model's compound base unit
named_unit = model.named["link_velocity"].unit
assert named_unit is not None
assert str(named_unit) == "m / s"  # value is in meters per second; unit says so
# ---8<--- [end: design_unit]

# ---8<--- [start: serialization]

us = UnitSystem.si()
model = stochas.StochasBase(us=us)

dist = stochas.NormalDistribution(
    name=stochas.DistName("goofy_velocity"),
    mu=10.0,
    sigma=0.5,
    unit=us["furlong / fortnight"],  # string subscript accepts any Pint expression
)

model.dists.update(dist)

data = model.model_dump()

# the unit descriptor serializes as {"name": "furlong / fortnight"} (scale/offset are excluded)
assert data["dists"]["goofy_velocity"]["unit"] == {"name": "furlong / fortnight"}

# when us is present in the JSON, validation restores all scale/offset values automatically
restored_model = stochas.StochasBase.model_validate(data)
goofy_unit = restored_model.dists["goofy_velocity"].unit
assert goofy_unit is not None and goofy_unit.scale
assert abs(goofy_unit.scale - float(us["furlong / fortnight"])) < 1e-12

# if you serialize without us, call with_unit_system() after loading
unitless_model = stochas.StochasBase()
unitless_model.dists.update(
    stochas.NormalDistribution(
        name=stochas.DistName("mass"), mu=1.0, sigma=0.05, unit=us.pound
    )
)

# round trip
unitless_raw = unitless_model.model_dump_json()
unitless_restored = stochas.StochasBase.model_validate_json(unitless_raw)

assert unitless_restored.us is None
mass_unit = unitless_restored.dists["mass"].unit
assert mass_unit is not None
assert mass_unit.scale is None  # scale missing until restored

has_units_model = unitless_restored.with_unit_system(us)
assert isinstance(has_units_model.us, UnitSystem)
restored_mass_unit = has_units_model.dists["mass"].unit
assert restored_mass_unit is not None
assert abs(float(restored_mass_unit) - float(us.pound)) < 1e-9
# ---8<--- [end: serialization]
