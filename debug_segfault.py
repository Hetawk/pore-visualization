#!/usr/bin/env python3
"""
Minimal test to isolate the segmentation fault issue
This will help us identify if the crash is from PyQt5, C++ extension, or matplotlib
"""

import sys
import os
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent / 'app'
sys.path.insert(0, str(app_dir))

# Initialize logging to out/ directory
try:
    from core.logger import init_debug_logging
    log_dir = app_dir / "out" / "logs"
    logger = init_debug_logging(log_dir)
    main_log = logger.main_logger
    main_log.info("Starting debug segfault test")
    print(f"📝 Debug logs will be saved to: {log_dir}")
except Exception as e:
    print(f"⚠️  Failed to initialize logging: {e}")
    logger = None
    main_log = None


def test_basic_imports():
    """Test basic imports step by step"""
    print("🧪 Testing basic imports...")

    try:
        print("  1. Testing numpy...")
        import numpy as np
        print("     ✓ numpy OK")

        print("  2. Testing pandas...")
        import pandas as pd
        print("     ✓ pandas OK")

        print("  3. Testing matplotlib...")
        import matplotlib
        matplotlib.use('Agg')  # Use non-GUI backend first
        import matplotlib.pyplot as plt
        print("     ✓ matplotlib OK")

        print("  4. Testing PyQt5 widgets...")
        from PyQt5.QtWidgets import QApplication
        print("     ✓ PyQt5.QtWidgets OK")

        print("  5. Testing PyQt5 GUI...")
        from PyQt5.QtGui import QPalette, QColor
        print("     ✓ PyQt5.QtGui OK")

        print("  6. Testing PyQt5 Core...")
        from PyQt5.QtCore import Qt
        print("     ✓ PyQt5.QtCore OK")

        return True

    except Exception as e:
        print(f"     ❌ Import failed: {e}")
        return False


def test_cpp_extension():
    """Test C++ extension separately"""
    print("🧪 Testing C++ extension...")

    try:
        from core.high_performance_renderer import CPP_AVAILABLE
        print(f"     C++ Available: {CPP_AVAILABLE}")

        if CPP_AVAILABLE:
            # Try basic C++ operations
            from core.high_performance_renderer import HighPerformanceRenderer
            print("     ✓ HighPerformanceRenderer imported")

            renderer = HighPerformanceRenderer(use_cpp=True)
            print("     ✓ HighPerformanceRenderer created")

            # Try basic operations without actual rendering
            import numpy as np
            centers = np.array(
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.float32)
            radii = np.array([0.5, 0.6, 0.7], dtype=np.float32)

            renderer.set_spheres_from_data(centers, radii)
            print("     ✓ Basic C++ operations OK")

        return True

    except Exception as e:
        print(f"     ❌ C++ extension failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minimal_qapp():
    """Test minimal QApplication creation"""
    print("🧪 Testing minimal QApplication...")

    try:
        from PyQt5.QtWidgets import QApplication

        # Create minimal app
        app = QApplication(sys.argv)
        print("     ✓ QApplication created")

        app.setApplicationName("Test")
        print("     ✓ QApplication configured")

        # Don't start event loop, just create and destroy
        app.quit()
        print("     ✓ QApplication terminated cleanly")

        return True

    except Exception as e:
        print(f"     ❌ QApplication failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minimal_gui():
    """Test minimal GUI creation without showing"""
    print("🧪 Testing minimal GUI creation...")

    try:
        from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget

        app = QApplication(sys.argv)
        print("     ✓ QApplication created")

        window = QMainWindow()
        print("     ✓ QMainWindow created")

        central = QWidget()
        window.setCentralWidget(central)
        print("     ✓ Central widget set")

        # DON'T show the window - just create it
        print("     ✓ GUI created (not shown)")

        app.quit()
        print("     ✓ GUI terminated cleanly")

        return True

    except Exception as e:
        print(f"     ❌ GUI creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_matplotlib_qt():
    """Test matplotlib with Qt backend"""
    print("🧪 Testing matplotlib with Qt backend...")

    try:
        import matplotlib
        matplotlib.use('Qt5Agg')  # This might cause issues
        print("     ✓ Qt5Agg backend set")

        import matplotlib.pyplot as plt
        print("     ✓ matplotlib.pyplot imported")

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        print("     ✓ Qt5Agg backend imported")

        return True

    except Exception as e:
        print(f"     ❌ matplotlib Qt backend failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests in sequence"""
    print("🚀 Starting segfault isolation tests...")
    print("=" * 50)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("C++ Extension", test_cpp_extension),
        ("Minimal QApp", test_minimal_qapp),
        ("Minimal GUI", test_minimal_gui),
        ("Matplotlib Qt", test_matplotlib_qt),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"     💥 Test crashed: {e}")
            results[test_name] = False
            import traceback
            traceback.print_exc()
            break  # Stop at first crash

    print("\n" + "=" * 50)
    print("📊 Test Results:")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print("\n💡 If the program crashed during a specific test, that's likely the cause of the segfault.")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
