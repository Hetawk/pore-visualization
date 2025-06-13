#!/usr/bin/env python3
"""
Simplified setup script for fast_renderer_cpp extension
"""
from setuptools import setup, Extension
import pybind11
import sys
import os

# Get pybind11 includes
pybind11_includes = [pybind11.get_include()]

# Add Python includes
python_includes = [
    f"{sys.base_prefix}/include/python{sys.version_info.major}.{sys.version_info.minor}"
]

# Compiler and linker flags for macOS
compile_args = [
    '-std=c++17',
    '-O3',
    '-mmacosx-version-min=10.14',
    '-stdlib=libc++',
    '-fvisibility=hidden',
    '-DVERSION_INFO="dev"'
]

link_args = [
    '-mmacosx-version-min=10.14',
    '-stdlib=libc++'
]

# Force system compiler
os.environ['CC'] = '/usr/bin/clang'
os.environ['CXX'] = '/usr/bin/clang++'

# Create extension
ext_modules = [
    Extension(
        'fast_renderer_cpp',
        sources=[
            'python_bindings.cpp',
            'fast_renderer.cpp'
        ],
        include_dirs=pybind11_includes + python_includes,
        language='c++',
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    name='fast_renderer_cpp',
    ext_modules=ext_modules,
    zip_safe=False,
)
