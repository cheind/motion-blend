import numpy as np
import matplotlib.pyplot as plt

from mblend import PolynomialMotion, PolynomialMotionBlend


def simple():
    t = np.linspace(0, 10, 100)

    m1 = PolynomialMotion(offset=0.0, coeffs=[-0.8, 1.0, 0.5])
    m2 = PolynomialMotion(offset=1.0, coeffs=[0, 3.0, 5.0])

    tnow = 2.5
    h = 2.0
    mb1 = PolynomialMotionBlend(m1, m2, tnow, h)

    fig, ax = plt.subplots()
    ax.plot(t[t >= m1.offset], m1.at(t[t >= m1.offset]), label="motion 1", linewidth=3)
    ax.plot(t[t >= m2.offset], m2.at(t[t >= m2.offset]), label="motion 2", linewidth=3)
    # ax.plot(t[t >= m3.t0], m3(t[t >= m3.t0]), label="m3")
    mask = t >= m1.offset
    ax.plot(t[mask], mb1.at(t[mask]), label="blend")
    # ax.plot(t[t >= mb2.t0], mb2(t[t >= mb2.offset & t <=]), label="b2")
    ax.axvline(tnow, linestyle="--", label="now", linewidth=1, c="k")
    ax.axvline(tnow + h, linestyle="--", label="now+horizon", linewidth=1)
    plt.legend()
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    fig.savefig("etc/simple.svg")
    plt.show()


def double_blend(flatten:bool):
    t = np.linspace(0, 10, 100)

    m1 = PolynomialMotion(offset=0.0, coeffs=[-0.8, 1.0, 0.5])
    m2 = PolynomialMotion(offset=1.0, coeffs=[0, 3.0, 5.0])

    h = 3.0
    mb1 = PolynomialMotionBlend(m1, m2, 2.5, h, flatten=flatten)

    m3 = PolynomialMotion(offset=3.0, coeffs=[1.2, 5.0, 7.0])
    mb2 = PolynomialMotionBlend(mb1, m3, 3.5, h, flatten=flatten)

    fig, ax = plt.subplots()
    ax.plot(t[t >= m1.offset], m1.at(t[t >= m1.offset]), label="motion 1", linewidth=3)
    ax.plot(t[t >= m2.offset], m2.at(t[t >= m2.offset]), label="motion 2", linewidth=3)
    ax.plot(t[t >= m3.offset], m3.at(t[t >= m3.offset]), label="motion 3", linewidth=3)

    # ax.plot(t[t >= m3.t0], m3(t[t >= m3.t0]), label="m3")
    mask = t >= m1.offset
    ax.plot(t[mask], mb2.at(t[mask]), label="blend2")
    ax.axvline(2.5, linestyle="--", label="start blend1 1<->2", linewidth=1, c="k")
    ax.axvline(2.5 + h, linestyle=":", label="end blend1", linewidth=1, c="k")
    ax.axvline(3.5, linestyle="--", label="start blend2 blend1<->3", linewidth=1)
    ax.axvline(3.5 + h, linestyle=":", label="end blend2", linewidth=1)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    plt.legend(loc="lower center", ncol=2)
    fig.savefig(f"etc/double-blend-flatten={flatten}.svg")
    plt.show()



simple()
double_blend(flatten=False)
double_blend(flatten=True)
