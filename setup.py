"""
Setup module.
For install package use "pip install ." or "pip install -e ."
"""
from setuptools import setup, find_packages

from lunavl.version import VERSION

setup(
    name="lunavl",
    description="Python interface for VisionLabs Luna platform",
    version=VERSION,
    author="VisionLabs",
    author_email="m.limonov@visionlabs.ru",
    packages=find_packages(exclude=["docs", "tests", "tests.*", "examples"]),
    zip_safe=False,
    install_requires=[
        'requests',
        'numpy',
    ]
)
