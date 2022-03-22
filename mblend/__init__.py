from typing import Protocol, Union
import dataclasses

import numpy as np
from numpy import isin


class Motion(Protocol):
    offset: float

    def at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        ...

    def d_at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        ...


@dataclasses.dataclass
class PolynomialMotion(Motion):
    offset: float
    coeffs: np.ndarray
    degree: int = dataclasses.field(init=False)

    def __post_init__(self):
        self.degree = len(self.coeffs) - 1
        self.coeffs = np.asarray(self.coeffs).reshape(-1, 1)

    def at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        scalar = np.isscalar(t)
        t = np.atleast_1d(t)

        v = np.vander(t - self.offset, self.degree + 1)  # Nx(D+1)
        x = v @ self.coeffs  # Nx1

        if scalar:
            return x.item()
        else:
            return x.squeeze(-1)

    def d_at(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
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
    A = np.zeros((4, 4))
    b = np.zeros(4)

    # Position at beginning (tnow) should match
    A[0, 0] = 0
    A[0, 1] = 0
    A[0, 2] = 0
    A[0, 3] = 1
    b[0] = m1.at(tnow)

    # Position at end of horizon should match
    A[1, 0] = h ** 3
    A[1, 1] = h ** 2
    A[1, 2] = h
    A[1, 3] = 1
    b[1] = m2.at(tnow + h)

    # at beginning and end

    A[2, 0] = 0
    A[2, 1] = 0
    A[2, 2] = 1
    A[2, 3] = 0
    b[2] = m1.d_at(tnow)

    A[3, 0] = 3 * h ** 2
    A[3, 1] = 2 * h
    A[3, 2] = 1
    A[3, 3] = 0
    b[3] = m2.d_at(tnow + h)

    coeffs = np.linalg.solve(A, b) # TODO: handle singularities
    return PolynomialMotion(tnow, coeffs)

@dataclasses.dataclass
class PolynomialMotionBlend(Motion):
    m1: Motion
    m2: Motion
    offset: float
    horizon: float
    blend: Motion = dataclasses.field(init=False)
    flatten: dataclasses.InitVar[bool] = False

    def __post_init__(self, flatten:bool):
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

def _flatten(m: Motion, offset:float) -> Motion:        
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
