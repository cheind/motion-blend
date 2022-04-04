from multiprocessing.sharedctypes import Value
from typing import Protocol, Union
import dataclasses

import numpy as np


class Motion(Protocol):
    """Protocol of a 1D motion."""

    """Shift of motion along time axis."""
    offset: float

    def at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the position at time(s)."""
        ...

    def d_at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the velocity at time(s)."""
        ...


@dataclasses.dataclass
class PolynomialMotion(Motion):
    """One-dimensional motion represented by a polynomial of degree N.

    Args:
        offset: Global time offset of this motion
        coeffs: N+1 polynomial coefficients starting with the highest term.
    """

    offset: float
    coeffs: np.ndarray
    degree: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.degree = len(self.coeffs) - 1
        self.coeffs = np.asarray(self.coeffs).reshape(-1, 1)

    def at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the position at time(s)."""
        scalar = np.isscalar(t)
        t = np.atleast_1d(t)

        v = np.vander(t - self.offset, self.degree + 1)  # Nx(D+1)
        x = v @ self.coeffs  # Nx1

        if scalar:
            return x.item()
        else:
            return x.squeeze(-1)

    def d_at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Returns the velocity at time(s)."""
        scalar = np.isscalar(t)
        t = np.atleast_1d(t) - self.offset

        dv = np.array(
            [i * t ** (i - 1) for i in reversed(range(1, self.degree + 1))]
        )  # NxD
        dx = dv.T @ self.coeffs[:-1]

        if scalar:
            return dx.item()
        else:
            return dx.squeeze(-1)


def poly_blend_3(m1: Motion, m2: Motion, tnow: float, h: float) -> PolynomialMotion:
    """Returns a third-degree polynomial function that blends two motions.

    Args:
        m1: First motion
        m2: Second motion
        tnow: Start of blend
        h: Horizon of blend

    Returns:
        mblend: Polynomial motion blending m1 and m2 in segment [tnow, tnow+h].
    """
    if h <= 0.0:
        raise ValueError("Horizon has to be > 0.0")

    A = np.zeros((4, 4))
    b = np.zeros(4)

    # Position at start (tnow) should match m1
    # Note, the offset (shift) of blended motion will be tnow
    A[0, 0] = 0
    A[0, 1] = 0
    A[0, 2] = 0
    A[0, 3] = 1
    b[0] = m1.at(tnow)

    # Position at end of horizon should match m2
    A[1, 0] = h ** 3
    A[1, 1] = h ** 2
    A[1, 2] = h
    A[1, 3] = 1
    b[1] = m2.at(tnow + h)

    # Velocity at start should match m1
    A[2, 0] = 0
    A[2, 1] = 0
    A[2, 2] = 1
    A[2, 3] = 0
    b[2] = m1.d_at(tnow)

    # Velocity at end should match m2
    A[3, 0] = 3 * h ** 2
    A[3, 1] = 2 * h
    A[3, 2] = 1
    A[3, 3] = 0
    b[3] = m2.d_at(tnow + h)

    coeffs = np.linalg.solve(A, b)  # TODO: handle singularities
    return PolynomialMotion(tnow, coeffs)


@dataclasses.dataclass
class PolynomialMotionBlend(Motion):
    """A piecewise blended motion with C1 smoothness.

    The blended motion consists of three pieces
     - m1 when t < start
     - blend when start <= t <= end of blending
     - m2 when end < t

    At joint points the positions and first order derivatives match up.

    If `flatten` is True, m1 and m2 will be simplified assuming that t is
    monotonically increasing and values of `t < start` are not of interest.
    Otherwise, recursive blending may lead to memory overflow.
    """

    m1: Motion
    m2: Motion
    offset: float
    horizon: float
    blend: Motion = dataclasses.field(init=False)
    flatten: dataclasses.InitVar[bool] = False

    def __post_init__(self, flatten: bool):
        if flatten:
            self.m1 = _flatten(self.m1, self.offset)
            self.m2 = _flatten(self.m2, self.offset)
        self.blend = poly_blend_3(self.m1, self.m2, self.offset, self.horizon)

    def at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self._compute(t, "at")

    def d_at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self._compute(t, "d_at")

    @property
    def range(self):
        return (self.offset, self.offset + self.horizon)

    def _compute(
        self, t: Union[float, np.ndarray], attr: str
    ) -> Union[float, np.ndarray]:
        scalar = np.isscalar(t)
        t = np.atleast_1d(t)

        low, high = self.range
        x = np.empty_like(t)
        mask = t < low
        x[mask] = getattr(self.m1, attr)(t[mask])
        mask = t > high
        x[mask] = getattr(self.m2, attr)(t[mask])
        mask = np.logical_and(t >= low, t <= high)
        x[mask] = getattr(self.blend, attr)(t[mask])

        if scalar:
            return x.item()
        else:
            return x


def _flatten(m: Motion, offset: float) -> Motion:
    """Recursively simplify older motions to avoid stacking of blends.

    The resulting motion is identical fo `t>=offset`, but may change for
    values less than offset.
    """
    if isinstance(m, PolynomialMotionBlend):
        if m.range[1] < offset:
            return m.m2
        elif m.range[0] < offset:
            return m.blend
        else:
            return _flatten(m.m1, offset)
    else:
        return m
