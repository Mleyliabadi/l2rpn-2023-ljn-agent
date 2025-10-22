import os
import setuptools
from setuptools import setup

__version__ = '0.1.0'

pkgs = {
    "required": [
        "cvxpy",
        "grid2op",
        "l2rpn_baselines",
        "lightsim2grid",
        "numpy",
        "torch",
        "stable-baselines3",
        "imageio",
        "numba",
        "lxml",
    ],
    "extras": {
        "recommended": [
            "curriculum_agent",
        ],
    }
}

pkgs["extras"]["codabench"] += pkgs["extras"]["recommended"]
pkgs["extras"]["test"] += pkgs["extras"]["recommended"]
pkgs["extras"]["test"] += pkgs["extras"]["docs"]
pkgs["extras"]["test"] += pkgs["extras"]["codabench"]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name='LJN_AGENT',
      version=__version__,
      description='LJN_AGENT to solve power grid congestion and overload',
      long_description=long_description,
      long_description_content_type="text/markdown",
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3.10',
          "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
          "Intended Audience :: Developers",
          "Intended Audience :: Education",
          "Intended Audience :: Science/Research",
          "Natural Language :: English"
      ],
      keywords='L2RPN_2023_challenge_IDF',
      author='Javaness',
      url="https://github.com/lajavaness/l2rpn-2023-ljn-agent/",
      license='MPL',
      packages=setuptools.find_packages(),
      include_package_data=True,
      package_data={
            # If any package contains *.txt or *.rst files, include them:
            "": ["*.ini", "*.zip", "*.npz"],
            },
      install_requires=pkgs["required"],
      extras_require=pkgs["extras"],
      zip_safe=False,
      entry_points={
          'console_scripts': []
     }
)