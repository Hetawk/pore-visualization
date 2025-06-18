#!/usr/bin/env python3
"""
Minimal Windows setup script for fast_renderer_cpp extension
Focus on compatibility and minimal dependencies
"""
import sys
import os
import platform
import subprocess
from pathlib import Path


def install_requirements():
    """Install required packages if not available"""
    try:
        import pybind11
    except ImportError:
        print("Installing pybind11...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pybind11"])
        import pybind11

    try:
        import numpy
    except ImportError:
        print("Installing numpy...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "numpy"])
        import numpy


def create_minimal_setup():
    """Create a minimal setup.py for better compatibility"""

    install_requirements()
    from setuptools import setup, Extension
    import pybind11
    from pybind11.setup_helpers import build_ext
    from pybind11 import get_cmake_dir
    import numpy

    # Include directories
    include_dirs = [
        pybind11.get_include(),
        numpy.get_include(),
    ]

    # Minimal compiler flags for compatibility
    if platform.system() == "Windows":
        extra_compile_args = [
            '/std:c++17',
            '/O2',
            '/MD',
            '/EHsc',
            '/DVERSION_INFO="minimal"',
            '/DNOMINMAX',  # Avoid Windows.h min/max conflicts
            '/D_USE_MATH_DEFINES',  # For M_PI etc.
        ]
        extra_link_args = []
    else:
        extra_compile_args = ['-std=c++17', '-O2', '-fPIC']
        extra_link_args = []

    # Create extension with minimal dependencies
    ext_modules = [
        Extension(
            'fast_renderer_cpp',
            sources=[
                'python_bindings.cpp',
                'fast_renderer.cpp'
            ],
            include_dirs=include_dirs,
            language='c++',
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[
                ('VERSION_INFO', '"minimal"'),
                ('NOMINMAX', '1'),
                ('_USE_MATH_DEFINES', '1'),
            ],
        )
    ]

    return ext_modules


def main():
    """Main setup function"""
    print(f"Setting up minimal C++ extension...")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.system()} {platform.architecture()}")
    print(f"Current directory: {os.getcwd()}")

    # Check if source files exist
    required_files = ['python_bindings.cpp', 'fast_renderer.cpp']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Required file {file} not found!")
            return False

    try:
        from setuptools import setup
        ext_modules = create_minimal_setup()

        # Build the extension
        setup(
            name='fast_renderer_cpp',
            ext_modules=ext_modules,
            zip_safe=False,
            cmdclass={'build_ext': build_ext},
        )

        print("✅ Extension built successfully!")
        return True

    except Exception as e:
        print(f"❌ Build failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
