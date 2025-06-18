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

    # Helper function to safely log messages
    def safe_log(message, level='info'):
        """Safely log a message, fallback to print if logger not available."""
        if main_log:
            getattr(main_log, level)(message)
        else:
            print(f"[{level.upper()}] {message}")

    # Helper function to safely log import attempts
    def safe_log_import(module_name, success, error_msg=None):
        """Safely log import attempts."""
        if logger:
            logger.log_import_attempt(module_name, success, error_msg)
        else:
            status = "✓" if success else "✗"
            msg = f"Import {module_name}: {status}"
            if error_msg:
                msg += f" - {error_msg}"
            print(msg)

    # Helper function to safely log GUI events
    def safe_log_gui_event(event, details):
        """Safely log GUI events."""
        if logger:
            logger.log_gui_event(event, details)
        else:
            print(f"[GUI] {event}: {details}")

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
        main_log = None

    try:
        safe_log("Adding app directory to Python path")
        safe_log(f"App directory: {app_dir}")

        # Test imports one by one with logging
        safe_log("Testing core imports...")

        try:
            import PyQt5
            safe_log_import("PyQt5", True)
        except ImportError as e:
            safe_log_import("PyQt5", False, str(e))
            raise

        try:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtGui import QPalette, QColor
            safe_log_import("PyQt5.QtWidgets/QtGui", True)
        except ImportError as e:
            safe_log_import("PyQt5.QtWidgets/QtGui", False, str(e))
            raise

        try:
            from pore_visualizer_gui import PoreVisualizerGUI
            safe_log_import("pore_visualizer_gui", True)
        except ImportError as e:
            safe_log_import("pore_visualizer_gui", False, str(e))
            raise

        print("🚀 Launching Professional Pore Network Visualizer...")
        safe_log("Creating QApplication instance")

        # Create QApplication with detailed logging
        app = QApplication(sys.argv)
        safe_log_gui_event("QApplication created", f"Args: {sys.argv}")

        app.setApplicationName("Professional Pore Network Visualizer")
        app.setApplicationVersion("2.0")
        app.setOrganizationName("Scientific Visualization Lab")
        safe_log("QApplication properties set")

        # Apply dark theme with logging
        safe_log("Applying dark theme...")
        app.setStyle('Fusion')
        safe_log_gui_event("Style set", "Fusion")

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
        safe_log_gui_event("Dark palette applied", "Custom dark theme")

        # Create main window with detailed logging
        safe_log("Creating PoreVisualizerGUI instance...")
        try:
            window = PoreVisualizerGUI()
            safe_log_gui_event("Main window created",
                               "PoreVisualizerGUI instantiated")
            safe_log("✓ Main window created successfully")
        except Exception as e:
            if logger:
                logger.log_exception(e, "Creating main window")
            else:
                safe_log(f"Exception creating main window: {e}", "error")
            raise

        # Show window with logging
        safe_log("Showing main window...")
        try:
            window.show()
            safe_log_gui_event("Main window shown", "window.show() called")
            safe_log("✓ Main window displayed")
        except Exception as e:
            if logger:
                logger.log_exception(e, "Showing main window")
            else:
                safe_log(f"Exception showing main window: {e}", "error")
            raise

        print("✅ GUI window created and displayed")
        safe_log("Starting QApplication event loop...")

        # Start event loop with proper error handling
        try:
            exit_code = app.exec_()
            safe_log(f"QApplication event loop exited with code: {exit_code}")
            return exit_code
        except Exception as e:
            if logger:
                logger.log_exception(e, "QApplication event loop")
            else:
                safe_log(f"Exception in QApplication event loop: {e}", "error")
            raise

    except ImportError as e:
        error_msg = f"Failed to import GUI components: {e}"
        print(f"❌ {error_msg}")
        if logger:
            logger.log_exception(e, "GUI component imports")
        else:
            safe_log(f"Import error: {e}", "error")
        print("🔧 Make sure PyQt5 is installed: pip install PyQt5")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Application closed by user.")
        safe_log("Application interrupted by user (Ctrl+C)")
        return 0
    except Exception as e:
        error_msg = f"Failed to launch application: {e}"
        print(f"❌ {error_msg}")
        if logger:
            logger.log_exception(e, "Application launch")
        else:
            safe_log(f"Application launch error: {e}", "error")
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
