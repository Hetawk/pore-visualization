#!/usr/bin/env python3
"""
Robust C++ Extension Setup for Windows
Handles common DLL loading issues and dependency problems
"""

import os
import sys
import platform
from pathlib import Path
from distutils.core import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11


def check_visual_studio():
    """Check if Visual Studio is available"""
    try:
        import distutils.msvc9compiler
        return True
    except:
        pass

    try:
        import distutils.msvccompiler
        return True
    except:
        pass

    return False


def get_compiler_flags():
    """Get appropriate compiler flags for Windows"""
    compile_args = []
    link_args = []

    if platform.system() == "Windows":
        # Windows-specific flags
        compile_args.extend([
            "/std:c++17",           # C++17 standard
            "/O2",                  # Optimization
            "/MD",                  # Use dynamic runtime
            "/DWIN32",              # Windows define
            "/D_WINDOWS",           # Windows define
            "/EHsc",                # Exception handling
            "/bigobj",              # Support large object files
        ])

        link_args.extend([
            "/MACHINE:X64",         # 64-bit target
            "/SUBSYSTEM:WINDOWS",   # Windows subsystem
        ])

    return compile_args, link_args


def main():
    print("=== Robust C++ Extension Builder ===")
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Pybind11 version: {pybind11.__version__}")

    # Check Visual Studio
    if not check_visual_studio():
        print("Warning: Visual Studio compiler tools not detected")

    # Source files
    cpp_files = [
        "python_bindings.cpp",
        "fast_renderer.cpp"
    ]

    # Verify source files exist
    missing_files = []
    for file in cpp_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"Error: Missing source files: {missing_files}")
        return False

    print(f"Source files found: {cpp_files}")

    # Get compiler flags
    compile_args, link_args = get_compiler_flags()

    print(f"Compile args: {compile_args}")
    print(f"Link args: {link_args}")

    # Create extension
    ext_modules = [
        Pybind11Extension(
            "fast_renderer_cpp",
            cpp_files,
            include_dirs=[
                pybind11.get_include(),
                ".",  # Current directory for headers
            ],
            cxx_std=17,
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=[
                ("VERSION_INFO", '"dev"'),
                ("PYBIND11_DETAILED_ERROR_MESSAGES", None),
            ],
        ),
    ]

    # Setup
    try:
        setup(
            name="fast_renderer_cpp",
            ext_modules=ext_modules,
            cmdclass={"build_ext": build_ext},
            zip_safe=False,
            python_requires=">=3.6",
        )
        print("✅ C++ extension built successfully!")
        return True
    except Exception as e:
        print(f"❌ Build failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
