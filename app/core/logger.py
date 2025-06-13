#!/usr/bin/env python3
"""
Comprehensive logging system for Pore Network Visualizer
Provides detailed debugging capabilities and crash reporting
"""

import logging
import sys
import os
import traceback
import signal
import faulthandler
from pathlib import Path
from datetime import datetime
from typing import Optional


class DebugLogger:
    """Enhanced logging system with crash detection and debugging features."""

    def __init__(self, log_dir: str = "logs", log_level: int = logging.DEBUG):
        """Initialize the debug logger with enhanced crash detection."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Enable Python fault handler for segmentation faults
        faulthandler.enable()

        # Set up signal handlers for crash detection
        signal.signal(signal.SIGSEGV, self._crash_handler)
        signal.signal(signal.SIGABRT, self._crash_handler)
        if hasattr(signal, 'SIGBUS'):
            signal.signal(signal.SIGBUS, self._crash_handler)

        # Create timestamp for this session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set up multiple loggers
        self.setup_loggers(log_level)

        # Log session start
        self.main_logger.info("="*60)
        self.main_logger.info(
            f"Pore Network Visualizer Debug Session: {self.session_id}")
        self.main_logger.info("="*60)
        self.main_logger.info(f"Python version: {sys.version}")
        self.main_logger.info(f"Platform: {sys.platform}")
        self.main_logger.info(f"Working directory: {os.getcwd()}")

    def setup_loggers(self, log_level: int):
        """Set up multiple specialized loggers."""

        # Main application logger
        self.main_logger = self._create_logger(
            'main',
            f"main_{self.session_id}.log",
            log_level,
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )

        # C++ extension logger
        self.cpp_logger = self._create_logger(
            'cpp_ext',
            f"cpp_ext_{self.session_id}.log",
            log_level,
            "%(asctime)s - CPP - %(levelname)s - %(message)s"
        )

        # PyQt5 logger
        self.gui_logger = self._create_logger(
            'gui',
            f"gui_{self.session_id}.log",
            log_level,
            "%(asctime)s - GUI - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )

        # Visualization logger
        self.viz_logger = self._create_logger(
            'visualization',
            f"viz_{self.session_id}.log",
            log_level,
            "%(asctime)s - VIZ - %(levelname)s - %(message)s"
        )

        # Error logger (for all errors)
        self.error_logger = self._create_logger(
            'errors',
            f"errors_{self.session_id}.log",
            logging.ERROR,
            "%(asctime)s - ERROR - %(name)s - %(funcName)s:%(lineno)d - %(message)s"
        )

    def _create_logger(self, name: str, filename: str, level: int, format_str: str) -> logging.Logger:
        """Create a specialized logger with file and console output."""
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # File handler
        file_handler = logging.FileHandler(self.log_dir / filename, mode='w')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler (only for main logger and errors)
        if name in ['main', 'errors'] or level >= logging.ERROR:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(max(level, logging.INFO))
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        return logger

    def _crash_handler(self, signum, frame):
        """Handle crashes and log detailed information."""
        crash_msg = f"CRASH DETECTED: Signal {signum} received"
        print(f"\n💥 {crash_msg}")

        # Write crash log
        crash_file = self.log_dir / f"CRASH_{self.session_id}.log"
        with open(crash_file, 'w') as f:
            f.write(f"{crash_msg}\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Signal: {signum}\n")
            f.write("="*50 + "\n")
            f.write("Python Stack Trace:\n")
            f.write("="*50 + "\n")

            if frame:
                traceback.print_stack(frame, file=f)
            else:
                traceback.print_stack(file=f)

            f.write("\n" + "="*50 + "\n")
            f.write("Fault Handler Traceback:\n")
            f.write("="*50 + "\n")

        # Write fault handler info to crash file
        with open(crash_file, 'a') as f:
            faulthandler.dump_traceback(file=f)

        print(f"💾 Crash log saved to: {crash_file}")

        # Try to cleanup
        try:
            self.main_logger.critical(crash_msg)
            self.error_logger.critical(crash_msg)
        except:
            pass

        # Exit gracefully
        sys.exit(1)

    def log_import_attempt(self, module_name: str, success: bool, error: Optional[str] = None):
        """Log module import attempts."""
        if success:
            self.main_logger.info(f"✓ Successfully imported {module_name}")
        else:
            self.main_logger.error(
                f"✗ Failed to import {module_name}: {error}")
            self.error_logger.error(f"Import failure - {module_name}: {error}")

    def log_cpp_operation(self, operation: str, success: bool, details: str = ""):
        """Log C++ extension operations."""
        if success:
            self.cpp_logger.info(f"✓ C++ {operation} successful: {details}")
        else:
            self.cpp_logger.error(f"✗ C++ {operation} failed: {details}")
            self.error_logger.error(
                f"C++ operation failed - {operation}: {details}")

    def log_gui_event(self, event: str, details: str = ""):
        """Log GUI events."""
        self.gui_logger.debug(f"GUI Event: {event} - {details}")

    def log_visualization_step(self, step: str, details: str = ""):
        """Log visualization pipeline steps."""
        self.viz_logger.info(f"Viz Step: {step} - {details}")

    def log_exception(self, exc: Exception, context: str = ""):
        """Log exceptions with full traceback."""
        exc_str = f"Exception in {context}: {str(exc)}"
        self.main_logger.exception(exc_str)
        self.error_logger.exception(exc_str)

        # Write detailed exception info
        exc_file = self.log_dir / \
            f"exception_{self.session_id}_{datetime.now().strftime('%H%M%S')}.log"
        with open(exc_file, 'w') as f:
            f.write(f"Exception Details\n")
            f.write(f"Time: {datetime.now()}\n")
            f.write(f"Context: {context}\n")
            f.write(f"Exception: {exc}\n")
            f.write("="*50 + "\n")
            traceback.print_exc(file=f)

    def close(self):
        """Close all loggers and handlers."""
        self.main_logger.info("="*60)
        self.main_logger.info(f"Debug session {self.session_id} ended")
        self.main_logger.info("="*60)

        for logger_name in ['main', 'cpp_ext', 'gui', 'visualization', 'errors']:
            logger = logging.getLogger(logger_name)
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

    # Standard logging method aliases for compatibility
    def debug(self, message: str):
        """Debug level logging."""
        self.main_logger.debug(message)

    def info(self, message: str):
        """Info level logging."""
        self.main_logger.info(message)

    def warning(self, message: str):
        """Warning level logging."""
        self.main_logger.warning(message)

    def error(self, message: str):
        """Error level logging."""
        self.main_logger.error(message)
        self.error_logger.error(message)

    def critical(self, message: str):
        """Critical level logging."""
        self.main_logger.critical(message)
        self.error_logger.critical(message)

    def exception(self, message: str):
        """Exception level logging with traceback."""
        self.main_logger.exception(message)
        self.error_logger.exception(message)


# Global logger instance
debug_logger: Optional[DebugLogger] = None


def init_debug_logging(log_dir: str = None) -> DebugLogger:
    """Initialize the global debug logger."""
    global debug_logger

    if log_dir is None:
        # Create logs directory in app/out/logs folder
        log_dir = Path(__file__).parent.parent / "out" / "logs"

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    debug_logger = DebugLogger(log_dir)
    return debug_logger


def get_logger() -> DebugLogger:
    """Get the global debug logger."""
    global debug_logger
    if debug_logger is None:
        debug_logger = init_debug_logging()
    return debug_logger


def close_debug_logging():
    """Close the global debug logger."""
    global debug_logger
    if debug_logger:
        debug_logger.close()
        debug_logger = None
