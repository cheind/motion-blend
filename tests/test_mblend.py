from cmath import atan
import pytest
import numpy as np
from numpy.testing import assert_allclose
import mblend


def test_polynomial_motion_constant():
    t = np.linspace(0, 10, 100)
    m = mblend.PolynomialMotion(0.0, [0.0, 0.0, 1.0])  # Linear
    x = m.at(t)
    v = m.d_at(t)
    assert_allclose(x, 1.0)
    assert_allclose(v, 0.0)

    m = mblend.PolynomialMotion(1.0, [0.0, 0.0, 1.0])
    x = m.at(t)
    v = m.d_at(t)
    assert_allclose(x, 1.0)
    assert_allclose(v, 0.0)


def test_polynomial_motion_linear():
    t = np.linspace(0, 10, 100)
    m = mblend.PolynomialMotion(0.0, [0.0, 1.0, 1.0])
    x = m.at(t)
    v = m.d_at(t)
    assert_allclose(x, t * 1.0 + 1.0)
    assert_allclose(v, 1.0)

    m = mblend.PolynomialMotion(1.0, [0.0, 1.0, 1.0])
    x = m.at(t)
    v = m.d_at(t)
    assert_allclose(x, (t - 1.0) * 1.0 + 1.0)
    assert_allclose(v, 1.0)


def test_polynomial_motion_quadric():
    t = np.linspace(0, 10, 100)
    m = mblend.PolynomialMotion(0.0, [1.0, 0.0, 0.0])
    x = m.at(t)
    v = m.d_at(t)
    assert_allclose(x, t ** 2)
    assert_allclose(v, 2 * t)

    m = mblend.PolynomialMotion(1.0, [1.0, 0.0, 0.0])
    x = m.at(t)
    v = m.d_at(t)
    assert_allclose(x, (t - 1.0) ** 2)
    assert_allclose(v, 2 * (t - 1.0))


def test_polynomial_blend_constant():
    m1 = mblend.PolynomialMotion(0.0, [1.0])
    m2 = mblend.PolynomialMotion(0.0, [2.0])
    mb = mblend.poly_blend_3(m1, m2, 1.0, 1.0)

    assert_allclose(mb.at(1.0), m1.at(1.0))
    assert_allclose(mb.at(2.0), m2.at(2.0))

    assert_allclose(mb.d_at(1.0), 0.0)
    assert_allclose(mb.d_at(2.0), 0.0)


def test_polynomial_blend_linear():
    m1 = mblend.PolynomialMotion(0.0, [1.0, 1.0])
    m2 = mblend.PolynomialMotion(0.0, [0.5, 2.0])
    mb = mblend.poly_blend_3(m1, m2, 1.0, 1.0)

    assert_allclose(mb.at(1.0), m1.at(1.0))
    assert_allclose(mb.at(2.0), m2.at(2.0))

    assert_allclose(mb.d_at(1.0), 1.0)
    assert_allclose(mb.d_at(2.0), 0.5)


def test_polynomial_blend_quadric():
    m1 = mblend.PolynomialMotion(0.0, np.random.randn(3))
    m2 = mblend.PolynomialMotion(-1.0, np.random.randn(3))
    mb = mblend.poly_blend_3(m1, m2, 1.0, 1.0)

    assert_allclose(mb.at(1.0), m1.at(1.0))
    assert_allclose(mb.at(2.0), m2.at(2.0))

    assert_allclose(mb.d_at(1.0), m1.d_at(1.0))
    assert_allclose(mb.d_at(2.0), m2.d_at(2.0))


def test_singularity_no_horizon():
    # [0,0,0,1] = m1(start)
    # [0,0,1,0] = d/dt m1(start)
    # [h**3,h**2,h,1] = m2(end)
    # [3h**2,2h,1,0] = d/dt m2(end)

    m1 = mblend.PolynomialMotion(0.0, np.random.randn(3))
    m2 = mblend.PolynomialMotion(-1.0, np.random.randn(3))

    with pytest.raises(ValueError):
        mblend.poly_blend_3(m1, m2, 1.0, 0.0)  # zero horizon


def test_singularity_same_motion():

    # Quadric
    m1 = mblend.PolynomialMotion(0.0, np.random.randn(3))

    # 3rd order blend
    mb = mblend.poly_blend_3(m1, m1, 0.0, 10.0)

    assert_allclose(mb.coeffs[0], 0.0, atol=1e-5)
    assert_allclose(mb.coeffs[1:], m1.coeffs, atol=1e-5)
