# motion-blend

This tiny library blends multiple projectile motions in 1D. This is useful when, for example, time-shifted motion estimates need to be integrated into a smooth motion trajectory and more complex filters such as Kalman are not an option. 

The image below shows two projectile motion estimates arriving at the integrator at different times. The integrator follows `motion1` up to the integration point `now` at which is blends smoothly to `motion2` during `horizon`.

<div align="center">
  <img src=./etc/simple.svg>
</div>

### Properties

Motions are polynomial (per default of order 2). The blending motion is a piecewise function, composed as follows
 - `motion1(t)` if `t < now`
 - `motion2(t)` if  `t > now + horizon`
  - `blend(t)` otherwise (smoothing polynomial defined below).

The smoothing function `blend` is an order 3 polynomial having the following smoothness properties
 - at `now` the position `x` equals `motion 1`
 - at `now+horizon` the position `x` equals `motion 2`
 - d/dt matches d/dt of `motion1` at `now`
 - d/dt matches d/dt of `motion2` at `now+horizon`.

Blended motions are composable, so that blending might occur recursively as the following example shows

<div align="center">
  <img src=./etc/double_blend.svg>
</div>

### Code
See [`__main__.py`](mblend/__main__.py) for a complete listing

```python
import numpy as np

from mblend import PolynomialMotion, PolynomialMotionBlend

# Two motions providing vectorized .at(t) and .d_at(t) methods
m1 = PolynomialMotion(offset=0.0, coeffs=[-0.8, 1.0, 0.5])
m2 = PolynomialMotion(offset=1.0, coeffs=[0, 3.0, 5.0])

tnow = 2.5
h = 2.0

# Blended motion between m1 & m2. Same interface
blend = PolynomialMotionBlend(m1, m2, tnow, h)

# Blended motions are blendable
m3 = PolynomialMotion(offset=3.0, coeffs=[1.2, 5.0, 7.0])
bblend = PolynomialMotionBlend(blend, m2, 3.5, h)

t = np.linspace(0,10,100)
x = bblend.at(t)
dxdt = bblend.d_at(t)
``` 

### Install
To install run,

```
pip install git+https://github.com/cheind/motion-blend.git
```

which requires Python 3.9