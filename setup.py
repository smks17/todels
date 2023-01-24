import os
from setuptools import find_packages
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, "README.md"), encoding="utf-8").read()
except OSError:
    README = ""

install_requires = [
    "torch>=1.13",
    "torchsummary>=1.5"
]

tests_require = []

__version__ = "0.0.1"

setup(
    name="Todels",
    version=__version__,
    description="Todels: A package that implement some famous models and  utils in pytorch",
    long_description="\n\n".join([README]),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    keywords="pytorch",
    author="Mahdi Kashani",
    author_email="Esmokes17@gmail.com",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        "testing": tests_require,
    },
)
