from pathlib import Path
import setuptools

try:
    from pl.utils.deploy import get_version, get_package_requirements
except ImportError:
    print("pl.utils.deploy not found, version number and requirements not valid for upload")
    # Useful for debugging
    def get_version(*args, **kwargs):
        return "0.0.0" 

    def get_package_requirements(*args, **kwargs):
        return []

with open("README.md", "r") as fh:
    long_description = fh.read()

parent = Path(__file__).resolve().parent
setuptools.setup(
    name="runx",
    version=get_version(version_path=parent / ".version"),
    author="Andrew Tao",
    author_email="atao@nvidia.com",
    description="runx - experiment manager for machine learning research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/runx",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=get_package_requirements(parent / "requirements.txt",
                                              start_flag="start package requirements"),
    python_requires='>=3.6',
    zip_safe=False
)
