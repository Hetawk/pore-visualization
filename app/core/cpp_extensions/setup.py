from setuptools import setup, Extension
import sys
import os

# Try to import pybind11
try:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pybind11 import get_cmake_dir
    PYBIND11_AVAILABLE = True

    include_dirs = [pybind11.get_include()]

except ImportError:
    print("Warning: pybind11 not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pybind11"])

    try:
        import pybind11
        from pybind11.setup_helpers import Pybind11Extension, build_ext
        from pybind11 import get_cmake_dir
        PYBIND11_AVAILABLE = True
        include_dirs = [pybind11.get_include()]
    except ImportError:
        print("Failed to install pybind11. Falling back to manual setup.")
        PYBIND11_AVAILABLE = False
        from setuptools import Extension
        from distutils.command.build_ext import build_ext

        # Try to find pybind11 headers manually
        possible_paths = [
            "/usr/local/include",
            "/opt/homebrew/include",
            "/usr/include",
            os.path.expanduser("~/.local/include"),
        ]

        include_dirs = []
        for path in possible_paths:
            pybind_path = os.path.join(path, "pybind11")
            if os.path.exists(pybind_path):
                include_dirs.append(path)
                break

        if not include_dirs:
            print("Error: Could not find pybind11 headers. Please install pybind11:")
            print("  pip install pybind11")
            print("  or")
            print("  conda install pybind11")
            sys.exit(1)

# Compiler flags
extra_compile_args = ['-std=c++17', '-O3']
extra_link_args = []

# Platform-specific settings
if sys.platform == "darwin":  # macOS
    # Use a more compatible macOS version and avoid SDK issues
    extra_compile_args.extend(['-mmacosx-version-min=10.14', '-stdlib=libc++'])
    extra_link_args.extend(['-mmacosx-version-min=10.14', '-stdlib=libc++'])

    # Force use of system clang to avoid TAPI/SDK issues
    os.environ['CXX'] = '/usr/bin/clang++'
    os.environ['CC'] = '/usr/bin/clang'

    # Add system SDK path
    try:
        import subprocess
        result = subprocess.run(['xcrun', '--show-sdk-path'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            sdk_path = result.stdout.strip()
            extra_compile_args.extend([f'-isysroot{sdk_path}'])
            extra_link_args.extend([f'-isysroot{sdk_path}'])
    except:
        pass

elif sys.platform.startswith("linux"):
    extra_compile_args.extend(['-pthread'])
    extra_link_args.extend(['-pthread'])

# Define the extension module
if PYBIND11_AVAILABLE:
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

setup(
    name="fast_renderer_cpp",
    version="1.0.0",
    author="Scientific Visualization Lab",
    author_email="",
    description="High-performance C++ renderer for 3D pore visualization",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        "test": ["pytest"],
    },
)
