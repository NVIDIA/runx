from pathlib import Path
from setuptools import setup, find_packages

# Version info -- read without importing
_locals = {}
with open("runx/_version.py") as fp:
    exec(fp.read(), None, _locals)
    version = _locals["__version__"]

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

parent = Path(__file__).resolve().parent
setup(
    name="runx",
    version=version,
    author="Andrew Tao",
    author_email="atao@nvidia.com",
    description="runx - experiment manager for machine learning research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/runx",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=requirements,
    python_requires='>=3.6',
)
