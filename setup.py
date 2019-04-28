import io
import os
import sys

from setuptools import setup, find_packages

install_requires = [
    "mordred==1.1.1*",
    "numpy==1.*",
    "pandas",
    "scikit-learn==0.20.2",
    "rdkit"]





setup(
    name="Orc_Band",
    version="1.0.0",
    description="predict organic bandgap",
    long_description=README.md,
    license="MIT",
    author="Yuhuan Meng, Liang Xu, Zhi Peng, Hongbo Qiao",
    author_email="xuliang1@uw.edu",
    url="https://github.com/HongboQiao/Orc_Band",
    platforms=["any"],
    keywords="Bandgap Prediction",
    packages=find_packages(),
    install_requires=install_requires,
    cmdclass={"test": None},
)
