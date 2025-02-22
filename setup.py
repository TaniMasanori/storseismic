from setuptools import setup, find_packages

setup(
    name="storseismic",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.5.0",
        "numpy>=1.19.0",
    ],
) 