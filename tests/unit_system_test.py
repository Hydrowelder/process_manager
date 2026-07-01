"""Tests for the unit conversion system."""

import numpy as np
import pytest

from stochas.unit_system import UnitDescriptor, UnitSystem


def test_unit_descriptor_float_returns_scale() -> None:
    d = UnitDescriptor(name="inch", scale=0.0254)
    assert float(d) == pytest.approx(0.0254)


def test_unit_descriptor_str_returns_name() -> None:
    d = UnitDescriptor(name="inch", scale=0.0254)
    assert str(d) == "inch"


def test_unit_descriptor_rmul_scales_scalar() -> None:
    d = UnitDescriptor(name="inch", scale=0.0254)
    assert 2.0 * d == pytest.approx(0.0508)


def test_unit_descriptor_mul_scales_array() -> None:
    d = UnitDescriptor(name="inch", scale=0.0254)
    result = np.array([1.0, 2.0, 3.0]) * d
    assert np.allclose(result, [0.0254, 0.0508, 0.0762])


def test_unit_system_si_base_unit_accessors() -> None:
    u = UnitSystem.si()
    assert u.length == "m"
    assert u.mass == "kg"
    assert u.time == "s"


def test_unit_system_ips_base_unit_accessors() -> None:
    u = UnitSystem.ips()
    assert u.length == "in"
    assert u.mass == "slinch"
    assert u.time == "s"


def test_unit_system_fps_base_unit_accessors() -> None:
    u = UnitSystem.fps()
    assert u.length == "ft"
    assert u.mass == "slug"
    assert u.time == "s"


def test_unit_system_cgs_base_unit_accessors() -> None:
    u = UnitSystem.cgs()
    assert u.length == "cm"
    assert u.mass == "g"
    assert u.time == "s"


def test_unit_system_si_meter_factor_is_one() -> None:
    u = UnitSystem.si()
    assert float(u.meter) == pytest.approx(1.0)
    assert str(u.meter) == "meter"


def test_unit_system_si_inch_factor() -> None:
    u = UnitSystem.si()
    assert float(u.inch) == pytest.approx(0.0254, rel=1e-6)
    assert str(u.inch) == "inch"


def test_unit_system_si_kilometer_factor() -> None:
    u = UnitSystem.si()
    assert float(u.kilometer) == pytest.approx(1000.0, rel=1e-6)


def test_unit_system_si_kilogram_factor_is_one() -> None:
    u = UnitSystem.si()
    assert float(u.kilogram) == pytest.approx(1.0)


def test_unit_system_si_gram_factor() -> None:
    u = UnitSystem.si()
    assert float(u.gram) == pytest.approx(0.001, rel=1e-6)


def test_unit_system_ips_meter_factor() -> None:
    u = UnitSystem.ips()
    assert float(u.meter) == pytest.approx(1.0 / 0.0254, rel=1e-6)


def test_unit_system_ips_coherent_force_is_lbf() -> None:
    """IPS uses slinch so that 1 slinch*in/s^2 = 1 lbf; factor for lbf must be 1.0."""
    u = UnitSystem.ips()
    assert float(u.lbf) == pytest.approx(1.0, rel=1e-6)


def test_unit_system_fps_coherent_force_is_lbf() -> None:
    """FPS uses slug so that 1 slug*ft/s^2 = 1 lbf; factor for lbf must be 1.0."""
    u = UnitSystem.fps()
    assert float(u.lbf) == pytest.approx(1.0, rel=1e-6)


def test_unit_system_cgs_coherent_force_is_dyne() -> None:
    """CGS: 1 gram*cm/s^2 = 1 dyne; factor for dyne must be 1.0."""
    u = UnitSystem.cgs()
    assert float(u.dyne) == pytest.approx(1.0, rel=1e-6)


def test_unit_system_cgs_newton_factor() -> None:
    """1 newton = 1e5 dyne, so newton factor in CGS is 1e5."""
    u = UnitSystem.cgs()
    assert float(u.newton) == pytest.approx(1e5, rel=1e-5)


def test_unit_system_fps_newton_factor() -> None:
    """1 newton in FPS model units (slug*ft/s^2 = lbf) is ~0.2248 lbf."""
    u = UnitSystem.fps()
    assert float(u.newton) == pytest.approx(1.0 / 4.44822, rel=1e-4)


def test_unit_system_unknown_unit_raises_attribute_error() -> None:
    u = UnitSystem.si()
    with pytest.raises(AttributeError, match="not a recognized Pint unit"):
        _ = u.not_a_real_unit


def test_unit_system_compound_unit_velocity() -> None:
    """Compound units like velocity are resolved from base dimensions."""
    u = UnitSystem.si()
    # 1 km/h in m/s = 1/3.6
    assert float(u.kilometer_per_hour) == pytest.approx(1.0 / 3.6, rel=1e-5)


def test_unit_system_compound_unit_force_si() -> None:
    """1 newton equals 1 kg*m/s^2 in SI model units."""
    u = UnitSystem.si()
    assert float(u.newton) == pytest.approx(1.0)


def test_unit_system_compound_unit_force_conversion() -> None:
    """1 pound-force in SI model units (newtons)."""
    u = UnitSystem.si()
    assert float(u.lbf) == pytest.approx(4.44822, rel=1e-4)


def test_unit_system_compound_unit_power() -> None:
    """1 horsepower in SI model units (watts = kg*m^2/s^3)."""
    u = UnitSystem.si()
    assert float(u.horsepower) == pytest.approx(745.7, rel=1e-3)


def test_unit_system_repr() -> None:
    u = UnitSystem.si()
    assert "length='m'" in repr(u)
    assert "mass='kg'" in repr(u)


def test_unit_descriptor_repr() -> None:
    d = UnitDescriptor(name="inch", scale=0.0254)
    assert "inch" in repr(d)
    assert "scale=0.0254" in repr(d)
    assert "offset=0.0" in repr(d)


def test_unit_system_multiply_array_in_generator_pattern() -> None:
    """End-to-end: multiply a numpy array by a unit descriptor to convert units."""
    u = UnitSystem.si()
    pos_in_inches = np.array([1.0, 2.0, 3.0])
    pos_in_meters = pos_in_inches * u.inch
    assert np.allclose(pos_in_meters, [0.0254, 0.0508, 0.0762])


def test_unit_descriptor_float_raises_when_scale_none() -> None:
    """float() on a descriptor with no scale raises RuntimeError."""
    d = UnitDescriptor(name="inch")
    with pytest.raises(RuntimeError, match="has no conversion scale"):
        float(d)


def test_unit_descriptor_mul_scales_scalar() -> None:
    """Descriptor * scalar delegates to __mul__."""
    d = UnitDescriptor(name="inch", scale=0.0254)
    assert d * 2.0 == pytest.approx(0.0508)


def test_unit_descriptor_truediv() -> None:
    """Descriptor / scalar delegates to __truediv__."""
    d = UnitDescriptor(name="inch", scale=0.0254)
    assert d / 2.0 == pytest.approx(0.0127)


def test_unit_descriptor_rtruediv() -> None:
    """Scalar / descriptor delegates to __rtruediv__."""
    d = UnitDescriptor(name="inch", scale=0.0254)
    assert 0.0508 / d == pytest.approx(2.0)


def test_unit_system_private_attr_raises() -> None:
    """Accessing a _-prefixed name on UnitSystem raises AttributeError without hitting Pint."""
    u = UnitSystem.si()
    with pytest.raises(AttributeError):
        _ = u._fake_private


def test_unit_system_unconfigured_dimension_raises() -> None:
    """Accessing a unit whose dimension has no configured base unit raises AttributeError."""
    u = UnitSystem(length="meter", mass="kilogram")  # no temperature
    with pytest.raises(AttributeError, match="no base unit configured"):
        _ = u.kelvin


def test_unit_system_temperature_offset_degF_to_kelvin() -> None:
    """degF->K conversion has a non-zero offset; 32 degF == 273.15 K and 212 degF == 373.15 K."""
    u = UnitSystem.si()  # temperature="K"
    degF = u.degF
    assert degF.offset != 0.0
    assert degF.scale == pytest.approx(5 / 9, rel=1e-6)
    # apply affine conversion: K = scale * F + offset
    assert degF * 32.0 == pytest.approx(273.15, rel=1e-4)
    assert degF * 212.0 == pytest.approx(373.15, rel=1e-4)


def test_unit_system_temperature_offset_degC_to_kelvin() -> None:
    """degC->K has scale=1 and offset=273.15; 0 degC == 273.15 K."""
    u = UnitSystem.si()  # temperature="K"
    degC = u.degC
    assert degC.scale == pytest.approx(1.0, rel=1e-6)
    assert degC.offset == pytest.approx(273.15, rel=1e-4)
    assert degC * 0.0 == pytest.approx(273.15, rel=1e-4)
    assert degC * 100.0 == pytest.approx(373.15, rel=1e-4)


def test_unit_descriptor_offset_applies_in_rmul() -> None:
    """Scalar * descriptor applies the full affine transform, not just the scale."""
    d = UnitDescriptor(name="degF", scale=5 / 9, offset=255.3722222)
    assert 32.0 * d == pytest.approx(273.15, rel=1e-4)
    assert 212.0 * d == pytest.approx(373.15, rel=1e-4)


def test_unit_descriptor_div_two_descriptors_returns_descriptor() -> None:
    """Dividing two UnitDescriptors produces a compound UnitDescriptor."""
    us = UnitSystem.si()
    velocity = us.m / us.s
    assert isinstance(velocity, UnitDescriptor)
    assert str(velocity) == "m / s"
    assert float(velocity) == pytest.approx(1.0)


def test_unit_descriptor_div_non_si_composes_scales() -> None:
    """Dividing non-SI descriptors composes their scales: inch/s scale = 0.0254."""
    us = UnitSystem.si()
    result = us.inch / us.second
    assert isinstance(result, UnitDescriptor)
    assert str(result) == "inch / second"
    assert float(result) == pytest.approx(0.0254, rel=1e-6)


def test_unit_descriptor_mul_two_descriptors_returns_descriptor() -> None:
    """Multiplying two UnitDescriptors produces a compound UnitDescriptor."""
    us = UnitSystem.si()
    result = us.kg * us.m
    assert isinstance(result, UnitDescriptor)
    assert str(result) == "kg * m"
    assert float(result) == pytest.approx(1.0)


def test_unit_descriptor_pow_returns_descriptor() -> None:
    """Raising a UnitDescriptor to a power produces a compound UnitDescriptor."""
    us = UnitSystem.si()
    area = us.m**2
    assert isinstance(area, UnitDescriptor)
    assert str(area) == "m ** 2"
    assert float(area) == pytest.approx(1.0)


def test_unit_descriptor_pow_wraps_compound_name_in_parens() -> None:
    """Compound names get parenthesized before the exponent to avoid ambiguity."""
    us = UnitSystem.si()
    vel_sq = (us.m / us.s) ** 2
    assert str(vel_sq) == "(m / s) ** 2"
    assert float(vel_sq) == pytest.approx(1.0)


def test_unit_descriptor_acceleration_arithmetic() -> None:
    """us.m / us.s ** 2 is equivalent to us.meter_per_second_squared."""
    us = UnitSystem.si()
    accel_arith = us.m / us.s**2
    accel_pint = us.meter_per_second_squared
    assert str(accel_arith) == "m / s ** 2"
    assert float(accel_arith) == pytest.approx(float(accel_pint), rel=1e-9)


def test_unit_descriptor_none_scale_propagates_in_compound() -> None:
    """A descriptor with scale=None produces a compound with scale=None."""
    a = UnitDescriptor(name="inch")  # no scale
    b = UnitDescriptor(name="second", scale=1.0)
    compound = a / b
    assert compound.scale is None


def test_unit_system_getitem_simple_unit() -> None:
    """us['m'] resolves the same as us.m."""
    us = UnitSystem.si()
    assert float(us["m"]) == pytest.approx(float(us.m))
    assert str(us["m"]) == "m"


def test_unit_system_getitem_compound_expression() -> None:
    """us['m/s'] resolves correctly via Pint expression parsing."""
    us = UnitSystem.si()
    vel = us["m/s"]
    assert isinstance(vel, UnitDescriptor)
    assert str(vel) == "m/s"
    assert float(vel) == pytest.approx(1.0)


def test_unit_system_getitem_inch_per_second() -> None:
    """us['inch/second'] handles a non-SI compound unit Pint can parse but __getattr__ cannot."""
    us = UnitSystem.si()
    result = us["inch/second"]
    assert float(result) == pytest.approx(0.0254, rel=1e-6)


def test_unit_system_getitem_unknown_raises_key_error() -> None:
    """us['not_a_unit'] raises KeyError."""
    us = UnitSystem.si()
    with pytest.raises(KeyError):
        _ = us["not_a_unit"]
