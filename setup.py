# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from setuptools import find_packages, setup

setup(
    name="hwtome",
    version="0.1",
    author="Meta",
    url="https://github.com/nickjeliopoulos/HardwareAwareToMe",
    description="Token Merging for Vision Transformers (Hardware Aware)",
    install_requires=[
        "torchvision",
        "numpy",
        "timm>=0.9.0",
        "pillow",
        "tqdm",
        "scipy",
    ],
    packages=find_packages(exclude=("examples", "build")),
)