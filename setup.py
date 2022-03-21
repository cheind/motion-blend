from setuptools import setup, find_packages
from pathlib import Path

THISDIR = Path(__file__).parent

with open(THISDIR / "requirements.txt") as f:
    required = f.read().splitlines()

main_ns = {}
with open(THISDIR / "mblend" / "__version__.py") as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="mblend",
    author="Christoph Heindl",
    description="Temporal blending of projectile motion estimates in 1D",
    license="MIT",
    version=main_ns["__version__"],
    packages=find_packages(".", include="mblend*"),
    install_requires=required,
    zip_safe=False,
)
