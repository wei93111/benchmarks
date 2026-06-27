#!/usr/bin/env python3
"""Build the QuadTree Attention CUDA extension modules used by benchmarks."""

from __future__ import annotations

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="quadtree_attention_benchmark_kernels",
    ext_modules=[
        CUDAExtension(
            "score_computation_cuda",
            [
                "src/score_computation.cpp",
                "src/score_computation_kernal.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        ),
        CUDAExtension(
            "value_aggregation_cuda",
            [
                "src/value_aggregation.cpp",
                "src/value_aggregation_kernel.cu",
            ],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
