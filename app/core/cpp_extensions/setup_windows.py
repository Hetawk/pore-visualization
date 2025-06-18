#!/usr/bin/env python3
"""
Windows-specific setup script for fast_renderer_cpp extension
Optimized for Windows without requiring full Visual Studio
"""
from setuptools import setup, Extension
import sys
import os
import platform


def get_windows_sdk_path():
    """Try to find Windows SDK path"""
    possible_paths = [
        r"C:\Program Files (x86)\Windows Kits\10\Include",
        r"C:\Program Files\Windows Kits\10\Include",
        r"C:\Program Files (x86)\Microsoft SDKs\Windows",
        r"C:\Program Files\Microsoft SDKs\Windows"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def build_extension():
    """Build the C++ extension for Windows"""

    # Try to import pybind11
    try:
        import pybind11
        from pybind11.setup_helpers import Pybind11Extension, build_ext
        pybind11_available = True
        include_dirs = [pybind11.get_include()]
    except ImportError:
        print("Installing pybind11...")
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pybind11"])

        import pybind11
        from pybind11.setup_helpers import Pybind11Extension, build_ext
        pybind11_available = True
        include_dirs = [pybind11.get_include()]

    # Windows-specific compiler flags
    if platform.system() == "Windows":
        extra_compile_args = [
            '/std:c++17',           # C++17 standard
            '/O2',                  # Optimization
            '/MD',                  # Runtime library
            '/DVERSION_INFO="dev"',  # Version definition
            '/EHsc',                # Exception handling
            '/bigobj',              # Support large object files
        ]
        extra_link_args = []

        # Try to add Windows SDK includes if available
        sdk_path = get_windows_sdk_path()
        if sdk_path:
            print(f"Found Windows SDK at: {sdk_path}")
    else:
        # Fallback for other systems
        extra_compile_args = ['-std=c++17', '-O3']
        extra_link_args = []

    # Create the extension
    if pybind11_available:
        ext_modules = [
            Pybind11Extension(
                "fast_renderer_cpp",
                [
                    "python_bindings.cpp",
                    "fast_renderer.cpp",
                ],
                include_dirs=include_dirs,
                language='c++',
                cxx_std=17,
                define_macros=[
                    ('VERSION_INFO', '"dev"'),
                ],
            ),
        ]
    else:
        ext_modules = [
            Extension(
                "fast_renderer_cpp",
                [
                    "python_bindings.cpp",
                    "fast_renderer.cpp",
                ],
                include_dirs=include_dirs,
                language='c++',
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                define_macros=[
                    ('VERSION_INFO', '"dev"'),
                ],
            ),
        ]

    return ext_modules, build_ext


if __name__ == "__main__":
    print("Setting up C++ extensions for Windows...")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()} {platform.architecture()}")

    ext_modules, build_ext_class = build_extension()

    setup(
        name="fast_renderer_cpp",
        version="1.0.0",
        author="Scientific Visualization Lab",
        description="High-performance C++ renderer for 3D pore visualization (Windows)",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext_class},
        zip_safe=False,
        python_requires=">=3.7",
        install_requires=[
            "numpy>=1.19.0",
            "pybind11>=2.6.0",
        ],
    )
