#!/usr/bin/env python3
"""
Pore Network Visualizer - Main Launcher
Clean entry point for the Professional Pore Network Visualization Application
"""

import sys
import os
from pathlib import Path


def main():
    """Main launcher for the Pore Network Visualizer application."""

    # Add app directory to Python path
    app_dir = Path(__file__).parent / 'app'
    sys.path.insert(0, str(app_dir))

    # Initialize comprehensive logging FIRST - save to out/logs/
    try:
        from core.logger import init_debug_logging, get_logger, close_debug_logging
        log_dir = app_dir / "out" / "logs"
        logger = init_debug_logging(log_dir)
        main_log = logger.main_logger
        main_log.info("Starting Pore Network Visualizer launcher")
        main_log.info(f"Logs will be saved to: {log_dir}")
    except Exception as e:
        print(f"⚠️  Failed to initialize logging: {e}")
        print("Continuing without advanced logging...")
        logger = None

    try:
        main_log.info("Adding app directory to Python path")
        main_log.info(f"App directory: {app_dir}")

        # Test imports one by one with logging
        main_log.info("Testing core imports...")

        try:
            import PyQt5
            logger.log_import_attempt("PyQt5", True)
        except ImportError as e:
            logger.log_import_attempt("PyQt5", False, str(e))
            raise

        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QPalette, QColor
            logger.log_import_attempt("PyQt5.QtWidgets/QtGui", True)
        except ImportError as e:
            logger.log_import_attempt("PyQt5.QtWidgets/QtGui", False, str(e))
            raise

        try:
            from pore_visualizer_gui import PoreVisualizerGUI
            logger.log_import_attempt("pore_visualizer_gui", True)
        except ImportError as e:
            logger.log_import_attempt("pore_visualizer_gui", False, str(e))
            raise

        print("🚀 Launching Professional Pore Network Visualizer...")
        main_log.info("Creating QApplication instance")

        # Create QApplication with detailed logging
        app = QApplication(sys.argv)
        logger.log_gui_event("QApplication created", f"Args: {sys.argv}")

        app.setApplicationName("Professional Pore Network Visualizer")
        app.setApplicationVersion("2.0")
        app.setOrganizationName("Scientific Visualization Lab")
        main_log.info("QApplication properties set")

        # Apply dark theme with logging
        main_log.info("Applying dark theme...")
        app.setStyle('Fusion')
        logger.log_gui_event("Style set", "Fusion")

        # Dark palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, QColor(0, 0, 0))
        palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        app.setPalette(palette)
        logger.log_gui_event("Dark palette applied", "Custom dark theme")

        # Create main window with detailed logging
        main_log.info("Creating PoreVisualizerGUI instance...")
        try:
            window = PoreVisualizerGUI()
            logger.log_gui_event("Main window created",
                                 "PoreVisualizerGUI instantiated")
            main_log.info("✓ Main window created successfully")
        except Exception as e:
            logger.log_exception(e, "Creating main window")
            raise

        # Show window with logging
        main_log.info("Showing main window...")
        try:
            window.show()
            logger.log_gui_event("Main window shown", "window.show() called")
            main_log.info("✓ Main window displayed")
        except Exception as e:
            logger.log_exception(e, "Showing main window")
            raise

        print("✅ GUI window created and displayed")
        main_log.info("Starting QApplication event loop...")

        # Start event loop with proper error handling
        try:
            exit_code = app.exec_()
            main_log.info(
                f"QApplication event loop exited with code: {exit_code}")
            return exit_code
        except Exception as e:
            logger.log_exception(e, "QApplication event loop")
            raise

    except ImportError as e:
        error_msg = f"Failed to import GUI components: {e}"
        print(f"❌ {error_msg}")
        if logger:
            logger.log_exception(e, "GUI component imports")
        print("🔧 Make sure PyQt5 is installed: pip install PyQt5")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Application closed by user.")
        if logger:
            main_log.info("Application interrupted by user (Ctrl+C)")
        return 0
    except Exception as e:
        error_msg = f"Failed to launch application: {e}"
        print(f"❌ {error_msg}")
        if logger:
            logger.log_exception(e, "Application launch")
        print("\n🔧 Try running from the app directory directly:")
        print(f"   cd {app_dir}")
        print("   python pore_visualizer_gui.py")
        return 1
    finally:
        # Clean up logging
        if logger:
            try:
                close_debug_logging()
            except:
                pass


if __name__ == "__main__":
    sys.exit(main())
