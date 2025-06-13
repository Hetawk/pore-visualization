#!/usr/bin/env python3
"""
MIST-like PyQt GUI Application for Interactive 3D Pore Network Visualization
Professional interface for scientific data analysis and visualization
"""

from pathlib import Path
import os
import sys
from core.pore_analyzer import PoreAnalyzer
from core.visualization_engine import VisualizationEngine
from core.data_manager import DataManager
from core.high_performance_renderer import HighPerformanceRenderer, create_spheres_from_pore_data, generate_bonds_from_spheres
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import numpy as np
import time
import traceback
import logging
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPalette, QColor, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSplitter, QTabWidget, QGroupBox, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QPushButton, QCheckBox, QFileDialog, QTextEdit, QProgressBar,
    QStatusBar, QMenuBar, QAction, QToolBar, QColorDialog, QMessageBox,
    QScrollArea, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib
# Configure matplotlib to avoid threading issues
matplotlib.use('Qt5Agg')
# Set matplotlib to thread-safe mode
plt.ioff()  # Turn off interactive mode to prevent GUI issues in threads

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


# Configure matplotlib backend before importing PyQt5
matplotlib.use('Qt5Agg')


class VisualizationThread(QThread):
    """Background thread for heavy visualization tasks."""

    # Changed from Figure to object (dict with plot data)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, engine: VisualizationEngine, data_path: str, viz_type: str):
        super().__init__()
        self.engine = engine
        self.data_path = data_path
        self.viz_type = viz_type

    def run(self):
        try:
            self.progress.emit(25)
            print(f"Creating visualization of type: {self.viz_type}")

            # Always defer matplotlib figure creation to main thread to avoid threading issues
            plot_data = {
                'viz_type': self.viz_type,
                'data_path': self.data_path,
                'status': 'needs_main_thread_plotting'
            }

            self.progress.emit(100)
            self.finished.emit(plot_data)
        except Exception as e:
            import traceback
            error_msg = f"Visualization error: {str(e)}\n{traceback.format_exc()}"

            # Enhanced error logging
            try:
                from core.logger import get_logger
                logger = get_logger()
                if logger:
                    logger.log_exception(e, "VisualizationThread")
                    logger.log_visualization_step(
                        "thread_error", error_msg[:200])
            except Exception as log_err:
                print(f"Failed to log thread error: {log_err}")

            # Print to console for immediate visibility
            print(f"THREAD ERROR: {error_msg}")

            self.error.emit(error_msg)


class ParameterControlWidget(QWidget):
    """Widget for controlling visualization parameters."""

    parameter_changed = pyqtSignal(str, object)
    preset_mode_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize parameter control UI with better layout and spacing."""
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Sphere Parameters
        sphere_group = QGroupBox("Sphere Parameters")
        sphere_layout = QGridLayout()
        sphere_layout.setSpacing(6)

        # Number of spheres
        sphere_layout.addWidget(QLabel("Number of Spheres:"), 0, 0)
        self.num_spheres = QSpinBox()
        self.num_spheres.setRange(100, 5000)
        self.num_spheres.setValue(1000)
        self.num_spheres.setMinimumHeight(25)
        self.num_spheres.valueChanged.connect(
            lambda v: self.parameter_changed.emit('num_spheres', v))
        sphere_layout.addWidget(self.num_spheres, 0, 1)

        # Opacity
        sphere_layout.addWidget(QLabel("Opacity:"), 1, 0)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.1, 1.0)
        self.opacity.setSingleStep(0.1)
        self.opacity.setValue(0.7)
        self.opacity.setMinimumHeight(25)
        self.opacity.valueChanged.connect(
            lambda v: self.parameter_changed.emit('opacity', v))
        sphere_layout.addWidget(self.opacity, 1, 1)

        # Size multiplier - increased range and default
        sphere_layout.addWidget(QLabel("Size Multiplier:"), 2, 0)
        self.size_multiplier = QDoubleSpinBox()
        self.size_multiplier.setRange(0.5, 10.0)  # Increased max range
        self.size_multiplier.setSingleStep(0.5)
        self.size_multiplier.setValue(3.0)  # Increased default value
        self.size_multiplier.setMinimumHeight(25)
        self.size_multiplier.valueChanged.connect(
            lambda v: self.parameter_changed.emit('size_multiplier', v))
        sphere_layout.addWidget(self.size_multiplier, 2, 1)

        # Colormap
        sphere_layout.addWidget(QLabel("Colormap:"), 3, 0)
        self.colormap = QComboBox()
        self.colormap.addItems(
            ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo'])
        self.colormap.setMinimumHeight(25)
        self.colormap.currentTextChanged.connect(
            lambda v: self.parameter_changed.emit('colormap', v))
        sphere_layout.addWidget(self.colormap, 3, 1)

        sphere_group.setLayout(sphere_layout)
        layout.addWidget(sphere_group)

        # Axis Dimension Controls Group
        axis_group = QGroupBox("Axis Dimension Controls")
        axis_layout = QGridLayout()
        axis_layout.setSpacing(6)

        # X-axis scale
        axis_layout.addWidget(QLabel("X-Axis Scale:"), 0, 0)
        self.axis_x_scale = QDoubleSpinBox()
        self.axis_x_scale.setRange(0.1, 5.0)
        self.axis_x_scale.setSingleStep(0.1)
        self.axis_x_scale.setValue(1.0)
        self.axis_x_scale.setMinimumHeight(25)
        self.axis_x_scale.valueChanged.connect(
            lambda v: self.parameter_changed.emit('axis_x_scale', v))
        axis_layout.addWidget(self.axis_x_scale, 0, 1)

        # Y-axis scale
        axis_layout.addWidget(QLabel("Y-Axis Scale:"), 1, 0)
        self.axis_y_scale = QDoubleSpinBox()
        self.axis_y_scale.setRange(0.1, 5.0)
        self.axis_y_scale.setSingleStep(0.1)
        self.axis_y_scale.setValue(1.0)
        self.axis_y_scale.setMinimumHeight(25)
        self.axis_y_scale.valueChanged.connect(
            lambda v: self.parameter_changed.emit('axis_y_scale', v))
        axis_layout.addWidget(self.axis_y_scale, 1, 1)

        # Z-axis scale
        axis_layout.addWidget(QLabel("Z-Axis Scale:"), 2, 0)
        self.axis_z_scale = QDoubleSpinBox()
        self.axis_z_scale.setRange(0.1, 5.0)
        self.axis_z_scale.setSingleStep(0.1)
        self.axis_z_scale.setValue(1.0)
        self.axis_z_scale.setMinimumHeight(25)
        self.axis_z_scale.valueChanged.connect(
            lambda v: self.parameter_changed.emit('axis_z_scale', v))
        axis_layout.addWidget(self.axis_z_scale, 2, 1)

        # Sphere base size control
        axis_layout.addWidget(QLabel("Sphere Base Size:"), 3, 0)
        self.sphere_base_size = QDoubleSpinBox()
        self.sphere_base_size.setRange(10.0, 500.0)
        self.sphere_base_size.setSingleStep(10.0)
        self.sphere_base_size.setValue(100.0)  # Increased default
        self.sphere_base_size.setMinimumHeight(25)
        self.sphere_base_size.valueChanged.connect(
            lambda v: self.parameter_changed.emit('sphere_base_size', v))
        axis_layout.addWidget(self.sphere_base_size, 3, 1)

        axis_group.setLayout(axis_layout)
        layout.addWidget(axis_group)

        # Space Dimension Controls Group
        dimension_group = QGroupBox("Space Dimensions")
        dimension_layout = QGridLayout()
        dimension_layout.setSpacing(6)

        # Space width
        dimension_layout.addWidget(QLabel("Width:"), 0, 0)
        self.space_width = QDoubleSpinBox()
        self.space_width.setRange(5.0, 100.0)
        self.space_width.setSingleStep(1.0)
        self.space_width.setValue(20.0)
        self.space_width.setMinimumHeight(25)
        self.space_width.setSuffix(" units")
        self.space_width.valueChanged.connect(
            lambda v: self.parameter_changed.emit('space_width', v))
        dimension_layout.addWidget(self.space_width, 0, 1)

        # Space height
        dimension_layout.addWidget(QLabel("Height:"), 1, 0)
        self.space_height = QDoubleSpinBox()
        self.space_height.setRange(5.0, 100.0)
        self.space_height.setSingleStep(1.0)
        self.space_height.setValue(20.0)
        self.space_height.setMinimumHeight(25)
        self.space_height.setSuffix(" units")
        self.space_height.valueChanged.connect(
            lambda v: self.parameter_changed.emit('space_height', v))
        dimension_layout.addWidget(self.space_height, 1, 1)

        # Space depth
        dimension_layout.addWidget(QLabel("Depth:"), 2, 0)
        self.space_depth = QDoubleSpinBox()
        self.space_depth.setRange(5.0, 100.0)
        self.space_depth.setSingleStep(1.0)
        self.space_depth.setValue(20.0)
        self.space_depth.setMinimumHeight(25)
        self.space_depth.setSuffix(" units")
        self.space_depth.valueChanged.connect(
            lambda v: self.parameter_changed.emit('space_depth', v))
        dimension_layout.addWidget(self.space_depth, 2, 1)

        # Maintain aspect ratio checkbox
        self.maintain_aspect_ratio = QCheckBox("Maintain Aspect Ratio")
        self.maintain_aspect_ratio.setChecked(True)
        self.maintain_aspect_ratio.toggled.connect(
            lambda v: self.parameter_changed.emit('maintain_aspect_ratio', v))
        dimension_layout.addWidget(self.maintain_aspect_ratio, 3, 0, 1, 2)

        # Aspect ratio controls (initially disabled if maintain_aspect_ratio is checked)
        dimension_layout.addWidget(QLabel("Aspect Ratio W:H:D:"), 4, 0)

        # Create horizontal layout for aspect ratio spinboxes
        aspect_widget = QWidget()
        aspect_layout = QHBoxLayout()
        aspect_layout.setContentsMargins(0, 0, 0, 0)
        aspect_layout.setSpacing(2)

        self.aspect_width = QDoubleSpinBox()
        self.aspect_width.setRange(0.1, 10.0)
        self.aspect_width.setSingleStep(0.1)
        self.aspect_width.setValue(1.0)
        self.aspect_width.setMinimumHeight(25)
        self.aspect_width.setMaximumWidth(60)
        self.aspect_width.valueChanged.connect(self.update_aspect_ratio)
        aspect_layout.addWidget(self.aspect_width)

        aspect_layout.addWidget(QLabel(":"))

        self.aspect_height = QDoubleSpinBox()
        self.aspect_height.setRange(0.1, 10.0)
        self.aspect_height.setSingleStep(0.1)
        self.aspect_height.setValue(1.0)
        self.aspect_height.setMinimumHeight(25)
        self.aspect_height.setMaximumWidth(60)
        self.aspect_height.valueChanged.connect(self.update_aspect_ratio)
        aspect_layout.addWidget(self.aspect_height)

        aspect_layout.addWidget(QLabel(":"))

        self.aspect_depth = QDoubleSpinBox()
        self.aspect_depth.setRange(0.1, 10.0)
        self.aspect_depth.setSingleStep(0.1)
        self.aspect_depth.setValue(1.0)
        self.aspect_depth.setMinimumHeight(25)
        self.aspect_depth.setMaximumWidth(60)
        self.aspect_depth.valueChanged.connect(self.update_aspect_ratio)
        aspect_layout.addWidget(self.aspect_depth)

        aspect_widget.setLayout(aspect_layout)
        dimension_layout.addWidget(aspect_widget, 4, 1)

        # Initially disable aspect ratio controls if maintain_aspect_ratio is checked
        self.toggle_aspect_ratio_controls(True)
        self.maintain_aspect_ratio.toggled.connect(
            self.toggle_aspect_ratio_controls)

        dimension_group.setLayout(dimension_layout)
        layout.addWidget(dimension_group)

        # Bond Parameters
        bond_group = QGroupBox("Bond Parameters")
        bond_layout = QGridLayout()
        bond_layout.setSpacing(6)

        # Show bonds
        self.show_bonds = QCheckBox("Show Atomic Bonds")
        self.show_bonds.setChecked(True)
        self.show_bonds.toggled.connect(
            lambda v: self.parameter_changed.emit('show_bonds', v))
        bond_layout.addWidget(self.show_bonds, 0, 0, 1, 2)

        # Bond thickness
        bond_layout.addWidget(QLabel("Bond Thickness:"), 1, 0)
        self.bond_thickness = QDoubleSpinBox()
        self.bond_thickness.setRange(0.1, 2.0)
        self.bond_thickness.setSingleStep(0.1)
        self.bond_thickness.setValue(0.5)
        self.bond_thickness.setMinimumHeight(25)
        self.bond_thickness.valueChanged.connect(
            lambda v: self.parameter_changed.emit('bond_thickness', v))
        bond_layout.addWidget(self.bond_thickness, 1, 1)

        bond_group.setLayout(bond_layout)
        layout.addWidget(bond_group)

        # Quick Visualization Presets Group
        presets_group = QGroupBox("Quick Visualization Modes")
        presets_layout = QGridLayout()
        presets_layout.setSpacing(6)

        # Visualization mode buttons
        self.ultra_realistic_btn = QPushButton("Ultra Realistic")
        self.ultra_realistic_btn.setMinimumHeight(30)
        self.ultra_realistic_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #3498db; color: white; }")
        self.ultra_realistic_btn.clicked.connect(
            lambda: self.preset_mode_changed.emit('ultra_realistic'))
        presets_layout.addWidget(self.ultra_realistic_btn, 0, 0)

        self.scientific_btn = QPushButton("Scientific Analysis")
        self.scientific_btn.setMinimumHeight(30)
        self.scientific_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #2ecc71; color: white; }")
        self.scientific_btn.clicked.connect(
            lambda: self.preset_mode_changed.emit('scientific_analysis'))
        presets_layout.addWidget(self.scientific_btn, 0, 1)

        self.presentation_btn = QPushButton("Presentation")
        self.presentation_btn.setMinimumHeight(30)
        self.presentation_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #e74c3c; color: white; }")
        self.presentation_btn.clicked.connect(
            lambda: self.preset_mode_changed.emit('presentation'))
        presets_layout.addWidget(self.presentation_btn, 1, 0)

        self.cross_section_btn = QPushButton("Cross Section")
        self.cross_section_btn.setMinimumHeight(30)
        self.cross_section_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #9b59b6; color: white; }")
        self.cross_section_btn.clicked.connect(
            lambda: self.preset_mode_changed.emit('cross_section'))
        presets_layout.addWidget(self.cross_section_btn, 1, 1)

        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

        # DEM Particle Visualization Group
        dem_group = QGroupBox("DEM Particle Visualization")
        dem_layout = QGridLayout()
        dem_layout.setSpacing(6)

        # DEM particle type selection
        dem_layout.addWidget(QLabel("Particle Type:"), 0, 0)
        self.particle_type = QComboBox()
        self.particle_type.addItems(['spherical', 'cubic', 'mixed'])
        self.particle_type.setCurrentText('mixed')
        self.particle_type.setMinimumHeight(25)
        self.particle_type.currentTextChanged.connect(
            lambda v: self.parameter_changed.emit('particle_type', v))
        dem_layout.addWidget(self.particle_type, 0, 1)

        # DEM visualization button
        self.dem_visualization_btn = QPushButton("Create DEM Visualization")
        self.dem_visualization_btn.setMinimumHeight(30)
        self.dem_visualization_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #ff6b35; color: white; }")
        self.dem_visualization_btn.clicked.connect(
            lambda: self.preset_mode_changed.emit('dem_particles'))
        dem_layout.addWidget(self.dem_visualization_btn, 1, 0, 1, 2)

        dem_group.setLayout(dem_layout)
        layout.addWidget(dem_group)

        # Live Rendering Controls Group
        live_group = QGroupBox("Live 3D Rendering")
        live_layout = QGridLayout()
        live_layout.setSpacing(6)

        # Start/Stop live rendering
        self.start_live_btn = QPushButton("Start Live Rendering")
        self.start_live_btn.setMinimumHeight(25)
        self.start_live_btn.setStyleSheet(
            "QPushButton { background-color: #27ae60; color: white; }")
        self.start_live_btn.clicked.connect(
            lambda: self.parameter_changed.emit('start_live_rendering', True))
        live_layout.addWidget(self.start_live_btn, 0, 0)

        self.stop_live_btn = QPushButton("Stop Live Rendering")
        self.stop_live_btn.setMinimumHeight(25)
        self.stop_live_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; }")
        self.stop_live_btn.clicked.connect(
            lambda: self.parameter_changed.emit('stop_live_rendering', True))
        live_layout.addWidget(self.stop_live_btn, 0, 1)

        # Camera controls
        live_layout.addWidget(QLabel("Camera Elevation:"), 1, 0)
        self.camera_elevation = QSlider(Qt.Horizontal)
        self.camera_elevation.setRange(-90, 90)
        self.camera_elevation.setValue(20)
        self.camera_elevation.setMinimumHeight(25)
        self.camera_elevation.valueChanged.connect(
            lambda v: self.parameter_changed.emit('camera_elevation', v))
        live_layout.addWidget(self.camera_elevation, 1, 1)

        live_layout.addWidget(QLabel("Camera Azimuth:"), 2, 0)
        self.camera_azimuth = QSlider(Qt.Horizontal)
        self.camera_azimuth.setRange(0, 360)
        self.camera_azimuth.setValue(45)
        self.camera_azimuth.setMinimumHeight(25)
        self.camera_azimuth.valueChanged.connect(
            lambda v: self.parameter_changed.emit('camera_azimuth', v))
        live_layout.addWidget(self.camera_azimuth, 2, 1)

        live_group.setLayout(live_layout)
        layout.addWidget(live_group)

        # MIST Analysis Group
        mist_group = QGroupBox("MIST-like Analysis")
        mist_layout = QGridLayout()
        mist_layout.setSpacing(6)

        # Analysis buttons
        self.connectivity_analysis_btn = QPushButton("Connectivity Analysis")
        self.connectivity_analysis_btn.setMinimumHeight(25)
        self.connectivity_analysis_btn.clicked.connect(
            lambda: self.parameter_changed.emit('run_mist_analysis', 'connectivity'))
        mist_layout.addWidget(self.connectivity_analysis_btn, 0, 0)

        self.size_distribution_btn = QPushButton("Size Distribution")
        self.size_distribution_btn.setMinimumHeight(25)
        self.size_distribution_btn.clicked.connect(
            lambda: self.parameter_changed.emit('run_mist_analysis', 'size_distribution'))
        mist_layout.addWidget(self.size_distribution_btn, 0, 1)

        self.clustering_analysis_btn = QPushButton("Clustering Analysis")
        self.clustering_analysis_btn.setMinimumHeight(25)
        self.clustering_analysis_btn.clicked.connect(
            lambda: self.parameter_changed.emit('run_mist_analysis', 'clustering'))
        mist_layout.addWidget(self.clustering_analysis_btn, 1, 0)

        self.full_mist_analysis_btn = QPushButton("Full MIST Analysis")
        self.full_mist_analysis_btn.setMinimumHeight(25)
        self.full_mist_analysis_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #8e44ad; color: white; }")
        self.full_mist_analysis_btn.clicked.connect(
            lambda: self.parameter_changed.emit('run_mist_analysis', 'full'))
        mist_layout.addWidget(self.full_mist_analysis_btn, 1, 1)

        mist_group.setLayout(mist_layout)
        layout.addWidget(mist_group)

        # Add stretch and set layout
        layout.addStretch()
        self.setLayout(layout)

    def choose_prism_color(self):
        """Choose prism color."""
        color = QColorDialog.getColor()
        if color.isValid():
            self.prism_color_btn.setStyleSheet(
                f"background-color: {color.name()}")
            self.parameter_changed.emit('prism_color', color.name())

    def toggle_aspect_ratio_controls(self, maintain_aspect):
        """Enable/disable aspect ratio controls based on maintain_aspect_ratio setting."""
        self.aspect_width.setEnabled(not maintain_aspect)
        self.aspect_height.setEnabled(not maintain_aspect)
        self.aspect_depth.setEnabled(not maintain_aspect)

    def update_aspect_ratio(self):
        """Update the custom aspect ratio when any aspect ratio spinbox changes."""
        if not self.maintain_aspect_ratio.isChecked():
            aspect_ratio = [
                self.aspect_width.value(),
                self.aspect_height.value(),
                self.aspect_depth.value()
            ]
            self.parameter_changed.emit('custom_aspect_ratio', aspect_ratio)

    def get_current_parameters(self):
        """Get current parameter values from the UI controls."""
        return {
            'num_spheres': self.num_spheres.value(),
            'opacity': self.opacity.value(),
            'size_multiplier': self.size_multiplier.value(),
            'colormap': self.colormap.currentText(),
            'axis_x_scale': self.axis_x_scale.value(),
            'axis_y_scale': self.axis_y_scale.value(),
            'axis_z_scale': self.axis_z_scale.value(),
            'sphere_base_size': self.sphere_base_size.value(),
            'space_width': self.space_width.value(),
            'space_height': self.space_height.value(),
            'space_depth': self.space_depth.value(),
            'maintain_aspect_ratio': self.maintain_aspect_ratio.isChecked(),
            'custom_aspect_ratio': [
                self.aspect_width.value(),
                self.aspect_height.value(),
                self.aspect_depth.value()
            ],
            'show_bonds': self.show_bonds.isChecked(),
            'bond_thickness': self.bond_thickness.value(),
            # DEM and live rendering parameters
            'particle_type': self.particle_type.currentText(),
            'camera_elevation': self.camera_elevation.value(),
            'camera_azimuth': self.camera_azimuth.value()
        }


class PoreVisualizerGUI(QMainWindow):
    """Main GUI application for pore network visualization."""

    def __init__(self):
        super().__init__()
        self.settings = QSettings('PoreVisualizer', 'MainApp')

        # Initialize comprehensive logging system first
        try:
            from core.logger import init_debug_logging, get_logger
            self.debug_logger = init_debug_logging("logs")
            self.logger = get_logger()
            self.logger.log_gui_event(
                "application_start", "Initializing Pore Visualizer GUI")
        except Exception as e:
            # Fallback to standard logging if debug logger fails
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
            self.logger.warning(f"Failed to initialize debug logger: {e}")
            self.debug_logger = None

        # Initialize core components
        self.data_manager = DataManager()
        self.visualization_engine = VisualizationEngine()
        self.pore_analyzer = PoreAnalyzer()

        # GUI state
        self.current_data_path = None
        self.current_figure = None
        self.visualization_thread = None
        self.auto_refresh_enabled = False  # Real-time parameter updates

        self.init_ui()
        self.restore_settings()

        # Set up real-time parameter callback
        self.visualization_engine.add_parameter_change_callback(
            self._on_parameter_changed)

        if self.logger:
            self.logger.log_gui_event(
                "application_ready", "GUI initialization completed")

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Professional Pore Network Visualizer")
        self.setGeometry(100, 100, 1600, 1000)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout()

        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)

        # Left panel - parameters
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)

        # Right panel - visualization
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set splitter proportions (30% left, 70% right)
        main_splitter.setSizes([400, 1200])

        main_layout.addWidget(main_splitter)
        central_widget.setLayout(main_layout)

        # Create menus and toolbars
        self.create_menus()
        self.create_toolbar()

        # Create status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def create_left_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout()

        # File operations
        file_group = QGroupBox("Data File")
        file_layout = QVBoxLayout()

        self.load_btn = QPushButton("Load Pore Data")
        self.load_btn.setMinimumHeight(35)
        self.load_btn.clicked.connect(self.load_data_file)
        file_layout.addWidget(self.load_btn)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet(
            "QLabel { color: #666; font-style: italic; }")
        file_layout.addWidget(self.file_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Visualization type
        viz_group = QGroupBox("Visualization Type")
        viz_layout = QVBoxLayout()

        self.viz_type = QComboBox()
        self.viz_type.addItems([
            'pore_network', 'size_distribution', 'clustering_analysis',
            'ultra_realistic', 'scientific_analysis', 'presentation', 'cross_section'
        ])
        self.viz_type.setMinimumHeight(30)
        viz_layout.addWidget(self.viz_type)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Parameter controls in scroll area
        scroll_area = QScrollArea()
        self.parameter_widget = ParameterControlWidget()
        self.parameter_widget.parameter_changed.connect(
            self.update_visualization_parameter)
        self.parameter_widget.preset_mode_changed.connect(
            self.apply_preset_visualization_mode)
        scroll_area.setWidget(self.parameter_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(400)
        layout.addWidget(scroll_area)

        # Generate button
        self.generate_btn = QPushButton("Generate Visualization")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.generate_visualization)
        layout.addWidget(self.generate_btn)

        # Auto-refresh toggle
        self.auto_refresh_checkbox = QCheckBox("Enable Real-time Auto-refresh")
        self.auto_refresh_checkbox.setChecked(False)
        self.auto_refresh_checkbox.toggled.connect(self.toggle_auto_refresh)
        layout.addWidget(self.auto_refresh_checkbox)

        # Export button
        self.export_btn = QPushButton("Export Visualization")
        self.export_btn.setMinimumHeight(35)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_visualization)
        layout.addWidget(self.export_btn)

        layout.addStretch()
        panel.setLayout(layout)
        return panel

    def create_right_panel(self):
        """Create the right visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        panel.setLayout(layout)
        return panel

    def create_menus(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        load_action = QAction('Load Data...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_data_file)
        file_menu.addAction(load_action)

        export_action = QAction('Export Visualization...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_visualization)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis menu
        analysis_menu = menubar.addMenu('Analysis')

        run_analysis_action = QAction('Run Pore Analysis', self)
        run_analysis_action.triggered.connect(self.run_analysis)
        analysis_menu.addAction(run_analysis_action)

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        """Create toolbar."""
        toolbar = self.addToolBar('Main')
        toolbar.setMovable(False)

        # Add quick actions to toolbar
        load_action = QAction('Load Data', self)
        load_action.triggered.connect(self.load_data_file)
        toolbar.addAction(load_action)

        toolbar.addSeparator()

        generate_action = QAction('Generate', self)
        generate_action.triggered.connect(self.generate_visualization)
        toolbar.addAction(generate_action)

        export_action = QAction('Export', self)
        export_action.triggered.connect(self.export_visualization)
        toolbar.addAction(export_action)

    def load_data_file(self):
        """Load pore data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Pore Data", "",
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )

        if file_path:
            try:
                # Load and validate data
                self.data_manager.load_pore_data(file_path)
                self.current_data_path = file_path

                # Update UI
                self.file_label.setText(f"Loaded: {file_path.split('/')[-1]}")
                self.file_label.setStyleSheet(
                    "QLabel { color: #2ecc71; font-weight: bold; }")
                self.generate_btn.setEnabled(True)
                self.status_bar.showMessage(f"Data loaded: {file_path}")
                self.logger.info(f"Data loaded from {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Load Error",
                                     f"Failed to load data:\n{str(e)}")
                self.logger.error(f"Failed to load data: {e}")

    def generate_visualization(self):
        """Generate 3D visualization."""
        if not self.current_data_path:
            QMessageBox.warning(
                self, "Warning", "Please load a data file first.")
            return

        viz_type = self.viz_type.currentText()
        self.generate_visualization_with_type(viz_type)

    def generate_visualization_with_type(self, viz_type):
        """Generate visualization with specific type using thread."""
        if not self.current_data_path:
            QMessageBox.warning(
                self, "Warning", "Please load a data file first.")
            return

        # Clear previous plot
        self.figure.clear()
        self.canvas.draw()

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.generate_btn.setEnabled(False)

        # Start visualization in background thread
        self.visualization_thread = VisualizationThread(
            self.visualization_engine, self.current_data_path, viz_type
        )
        self.visualization_thread.finished.connect(
            self.on_visualization_finished)
        self.visualization_thread.error.connect(self.on_visualization_error)
        self.visualization_thread.progress.connect(self.progress_bar.setValue)
        self.visualization_thread.start()

    def apply_preset_visualization_mode(self, mode):
        """Apply a preset visualization mode with optimized parameters."""
        if not self.current_data_path:
            QMessageBox.warning(
                self, "Warning", "Please load a data file first.")
            return

        try:
            self.status_bar.showMessage(
                f"Generating {mode.replace('_', ' ').title()} visualization...")

            # Clear previous plot
            self.figure.clear()
            self.canvas.draw()

            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(25)

            # Handle different visualization modes
            if mode == 'dem_particles':
                # Create DEM particle visualization
                particle_type = self.parameter_widget.particle_type.currentText()
                fig = self.visualization_engine.create_dem_visualization(
                    self.current_data_path, particle_type)
                self.on_visualization_finished(fig)

            elif mode in ['ultra_realistic', 'scientific_analysis', 'presentation', 'cross_section']:
                # Use advanced visualization engine
                if hasattr(self.visualization_engine, 'create_advanced_visualization'):
                    fig = self.visualization_engine.create_advanced_visualization(
                        self.current_data_path, mode)
                    self.on_visualization_finished(fig)
                else:
                    # Fallback to regular visualization - use thread
                    self.generate_visualization_with_type(mode)
            else:
                # Fallback to regular visualization - use thread
                self.generate_visualization_with_type(mode)

        except Exception as e:
            QMessageBox.critical(self, "Visualization Error",
                                 f"Failed to create {mode} visualization:\n{str(e)}")
            self.progress_bar.setVisible(False)

    def on_visualization_finished(self, plot_data):
        """Handle completed visualization."""
        try:
            # Clear the current figure
            self.figure.clear()

            # Check if we need to create the figure in main thread
            if isinstance(plot_data, dict):
                if plot_data.get('status') == 'needs_main_thread_plotting':
                    # Create visualization in main thread to avoid matplotlib threading issues
                    print(
                        f"Creating {plot_data['viz_type']} visualization in main thread")
                    figure = self.visualization_engine.create_pore_network_visualization(
                        plot_data['data_path'], plot_data['viz_type'])
                elif plot_data.get('status') == 'complete':
                    figure = plot_data.get('figure')
                else:
                    figure = None
            else:
                # Backward compatibility - assume it's a figure
                figure = plot_data

            # Copy the generated figure's content to our canvas figure
            if figure and len(figure.axes) > 0:
                # Get the source axis from generated figure
                source_ax = figure.axes[0]

                # Create new axis in our canvas figure with same projection
                if hasattr(source_ax, 'zaxis'):
                    new_ax = self.figure.add_subplot(111, projection='3d')
                else:
                    new_ax = self.figure.add_subplot(111)

                # Copy all the plot elements from source to destination
                for collection in source_ax.collections:
                    if hasattr(collection, '_offsets3d'):
                        # 3D scatter plot
                        xs, ys, zs = collection._offsets3d
                        if hasattr(xs, '__len__') and len(xs) > 0:
                            colors = collection.get_facecolors()
                            sizes = collection.get_sizes()
                            alpha = collection.get_alpha() if collection.get_alpha() is not None else 1.0

                            # Ensure all arrays have consistent lengths
                            min_length = min(len(xs), len(ys), len(zs))
                            if len(colors) > 0:
                                min_length = min(min_length, len(colors))
                            if len(sizes) > 0:
                                min_length = min(min_length, len(sizes))

                            # Trim arrays to consistent length
                            xs_trimmed = xs[:min_length]
                            ys_trimmed = ys[:min_length]
                            zs_trimmed = zs[:min_length]
                            colors_trimmed = colors[:min_length] if len(
                                colors) > 0 else None
                            sizes_trimmed = sizes[:min_length] if len(
                                sizes) > 0 else None

                            # Create scatter plot with consistent arrays
                            scatter_kwargs = {
                                'alpha': alpha
                            }
                            if colors_trimmed is not None and len(colors_trimmed) > 0:
                                scatter_kwargs['c'] = colors_trimmed
                            if sizes_trimmed is not None and len(sizes_trimmed) > 0:
                                scatter_kwargs['s'] = sizes_trimmed

                            new_ax.scatter(xs_trimmed, ys_trimmed,
                                           zs_trimmed, **scatter_kwargs)

                # Copy line plots
                for line in source_ax.lines:
                    if hasattr(line, '_verts3d'):
                        # 3D line
                        xs, ys, zs = line._verts3d
                        new_ax.plot(xs, ys, zs, color=line.get_color(),
                                    alpha=line.get_alpha(), linewidth=line.get_linewidth())

                # Copy axis labels and title
                new_ax.set_xlabel(source_ax.get_xlabel())
                new_ax.set_ylabel(source_ax.get_ylabel())
                if hasattr(source_ax, 'set_zlabel'):
                    new_ax.set_zlabel(source_ax.get_zlabel())
                new_ax.set_title(source_ax.get_title())

                # Copy view angle for 3D plots
                if hasattr(source_ax, 'view_init') and hasattr(new_ax, 'view_init'):
                    elev = source_ax.elev
                    azim = source_ax.azim
                    new_ax.view_init(elev=elev, azim=azim)

            # Force canvas update
            self.canvas.draw()
            self.canvas.flush_events()

            # Update state
            self.current_figure = self.figure
            self.export_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.generate_btn.setEnabled(True)
            self.status_bar.showMessage("Visualization generated successfully")

        except Exception as e:
            # Enhanced error logging
            error_msg = f"Error displaying visualization: {str(e)}"

            # Log error to console and files
            try:
                from core.logger import get_logger
                logger = get_logger()
                if logger:
                    logger.log_exception(e, "GUI Visualization Display")
                    logger.log_gui_event(
                        "visualization_display_error", error_msg)
            except Exception as log_err:
                print(f"Failed to log error: {log_err}")

            # Print to console for immediate visibility
            print(f"VISUALIZATION DISPLAY ERROR: {error_msg}")
            import traceback
            print(traceback.format_exc())

            QMessageBox.critical(self, "Visualization Error",
                                 f"Error displaying visualization:\n{str(e)}")
            self.progress_bar.setVisible(False)
            self.generate_btn.setEnabled(True)

    def on_visualization_error(self, error_msg):
        """Handle visualization error."""
        # Log error to console and files using the logger system
        try:
            from core.logger import get_logger
            logger = get_logger()
            if logger:
                logger.log_exception(Exception(error_msg), "GUI Visualization")
                # Log first 200 chars
                logger.log_gui_event("visualization_error", error_msg[:200])
        except Exception as log_err:
            print(f"Failed to log error: {log_err}")

        # Print to console for immediate visibility
        print(f"VISUALIZATION ERROR: {error_msg}")

        # Show GUI error dialog
        QMessageBox.critical(self, "Visualization Error",
                             f"Failed to generate visualization:\n{error_msg}")
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_bar.showMessage("Visualization failed")

    def update_visualization_parameter(self, param_name: str, value):
        """Update visualization parameter and trigger real-time rendering."""
        # Handle special parameter cases
        if param_name == 'start_live_rendering':
            self.start_live_rendering()
            return
        elif param_name == 'stop_live_rendering':
            self.stop_live_rendering()
            return
        elif param_name == 'camera_elevation' or param_name == 'camera_azimuth':
            self.update_live_camera()
            return
        elif param_name == 'run_mist_analysis':
            self.run_mist_analysis(value)
            return

        # Standard parameter update
        self.visualization_engine.update_parameters({param_name: value})
        self.status_bar.showMessage(f"Updated {param_name}: {value}")

        # Auto-refresh visualization if data is loaded and auto-refresh is enabled
        if self.current_data_path and hasattr(self, 'auto_refresh_enabled') and self.auto_refresh_enabled:
            # Debounce rapid parameter changes by using a timer
            if not hasattr(self, 'refresh_timer'):
                self.refresh_timer = QTimer()
                self.refresh_timer.setSingleShot(True)
                self.refresh_timer.timeout.connect(self._do_real_time_refresh)

            # Restart timer on each parameter change to debounce
            self.refresh_timer.stop()
            # 500ms delay to debounce rapid changes
            self.refresh_timer.start(500)

    def _do_real_time_refresh(self):
        """Perform real-time visualization refresh."""
        try:
            if self.current_data_path:
                viz_type = self.viz_type.currentText()
                self.status_bar.showMessage("Auto-refreshing visualization...")

                # Use current visualization type for refresh
                if viz_type in ['ultra_realistic', 'scientific_analysis', 'presentation', 'cross_section']:
                    figure = self.visualization_engine.create_advanced_visualization(
                        self.current_data_path, viz_type)
                else:
                    figure = self.visualization_engine.create_pore_network_visualization(
                        self.current_data_path, viz_type)

                if figure:
                    self.on_visualization_finished(figure)
                    self.status_bar.showMessage("Auto-refresh completed")
                else:
                    self.status_bar.showMessage("Auto-refresh failed")
        except Exception as e:
            self.logger.error(f"Real-time refresh error: {e}")
            self.status_bar.showMessage("Auto-refresh error")

    def toggle_auto_refresh(self, enabled: bool):
        """Toggle auto-refresh mode for real-time parameter updates."""
        self.auto_refresh_enabled = enabled
        if enabled:
            self.status_bar.showMessage(
                "Auto-refresh enabled - visualizations will update automatically")
        else:
            self.status_bar.showMessage("Auto-refresh disabled")
        self.logger.info(
            f"Auto-refresh {'enabled' if enabled else 'disabled'}")

    def _on_parameter_changed(self, new_params: Dict[str, Any], old_params: Dict[str, Any]):
        """Callback for when visualization parameters change."""
        # This could be used for additional real-time effects or logging
        if self.logger:
            param_names = list(new_params.keys())
            self.logger.debug(f"Parameters changed: {param_names}")

    def export_visualization(self):
        """Export current visualization."""
        if not self.current_figure:
            QMessageBox.warning(self, "Warning", "No visualization to export.")
            return

        # Default to out/ directory
        out_dir = Path(__file__).parent / "out"
        out_dir.mkdir(exist_ok=True)
        default_filename = out_dir / "visualization.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Visualization", str(default_filename),
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;All Files (*)"
        )

        if file_path:
            try:
                self.current_figure.savefig(
                    file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(
                    self, "Success", "Visualization exported successfully!")
                self.status_bar.showMessage(f"Exported to {file_path}")

            except Exception as e:
                # Enhanced export error logging
                error_msg = f"Failed to export visualization: {str(e)}"

                # Log error to console and files
                try:
                    from core.logger import get_logger
                    logger = get_logger()
                    if logger:
                        logger.log_exception(e, "Export Visualization")
                        logger.log_gui_event("export_error", error_msg)
                except Exception as log_err:
                    print(f"Failed to log export error: {log_err}")

                # Print to console for immediate visibility
                print(f"EXPORT ERROR: {error_msg}")
                import traceback
                print(traceback.format_exc())

                QMessageBox.critical(self, "Export Error",
                                     f"Failed to export:\n{str(e)}")

    def run_analysis(self):
        """Run pore structure analysis."""
        if not self.current_data_path:
            QMessageBox.warning(
                self, "Warning", "Please load a data file first.")
            return

        try:
            # Load data and run analysis
            df = self.data_manager.load_pore_data(self.current_data_path)
            analysis_results = self.pore_analyzer.analyze_pore_distribution(df)
            report = self.pore_analyzer.generate_analysis_report(
                analysis_results)

            # Show results in a dialog
            dialog = QTextEdit()
            dialog.setWindowTitle("Analysis Results")
            dialog.setPlainText(report)
            dialog.setReadOnly(True)
            dialog.setFont(QFont("Courier", 10))
            dialog.resize(600, 400)
            dialog.show()

            self.status_bar.showMessage("Analysis completed successfully")

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error",
                                 f"Failed to run analysis:\n{str(e)}")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Pore Network Visualizer",
            "Professional Pore Network Visualizer v2.0\n\n"
            "Advanced 3D visualization tool for pore structure analysis\n"
            "with enhanced rendering and scientific features.\n\n"
            "Features:\n"
            "• Interactive 3D visualization with larger, more visible spheres\n"
            "• Advanced pore analysis with axis dimension controls\n"
            "• Multiple visualization modes\n"
            "• Publication-quality export\n"
            "• Real-time parameter tuning\n"
            "• Enhanced sphere sizing and axis scaling"
        )

    def closeEvent(self, event):
        """Handle application closing."""
        # Clean up callbacks
        self.visualization_engine.remove_parameter_change_callback(
            self._on_parameter_changed)

        # Clean up matplotlib figures to prevent memory warnings
        try:
            self.visualization_engine.cleanup_figures()
        except Exception as e:
            print(f"Error during figure cleanup: {e}")

        self.save_settings()
        if self.visualization_thread and self.visualization_thread.isRunning():
            self.visualization_thread.terminate()
            self.visualization_thread.wait()
        event.accept()

    def save_settings(self):
        """Save application settings."""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())

    def restore_settings(self):
        """Restore application settings."""
        try:
            geometry = self.settings.value('geometry')
            if geometry:
                self.restoreGeometry(geometry)

            state = self.settings.value('windowState')
            if state:
                self.restoreState(state)

            self.logger.info("Application settings restored")
        except Exception as e:
            self.logger.warning(f"Failed to restore settings: {e}")

    def start_live_rendering(self):
        """Start live 3D rendering."""
        if not self.current_data_path:
            QMessageBox.warning(
                self, "Warning", "Please load a data file first.")
            return

        try:
            fig = self.visualization_engine.start_live_rendering(
                self.current_data_path)
            if fig:
                self.status_bar.showMessage("Live rendering started")
                self.on_visualization_finished(fig)
            else:
                self.status_bar.showMessage("Failed to start live rendering")
        except Exception as e:
            QMessageBox.critical(self, "Live Rendering Error",
                                 f"Failed to start live rendering:\n{str(e)}")

    def stop_live_rendering(self):
        """Stop live 3D rendering."""
        try:
            self.visualization_engine.stop_live_rendering()
            self.status_bar.showMessage("Live rendering stopped")
        except Exception as e:
            QMessageBox.critical(self, "Live Rendering Error",
                                 f"Failed to stop live rendering:\n{str(e)}")

    def update_live_camera(self):
        """Update live camera parameters."""
        try:
            elevation = self.parameter_widget.camera_elevation.value()
            azimuth = self.parameter_widget.camera_azimuth.value()
            self.visualization_engine.update_live_camera(elevation, azimuth)
            self.status_bar.showMessage(
                f"Camera updated: elev={elevation}, azim={azimuth}")
        except Exception as e:
            self.logger.error(f"Error updating camera: {e}")

    def run_mist_analysis(self, analysis_type: str):
        """Run MIST-like analysis."""
        if not self.current_data_path:
            QMessageBox.warning(
                self, "Warning", "Please load a data file first.")
            return

        try:
            self.status_bar.showMessage(f"Running {analysis_type} analysis...")
            results = self.visualization_engine.analyze_with_mist(
                self.current_data_path)

            if results:
                # Display results in a dialog
                results_text = self.format_analysis_results(
                    results, analysis_type)
                QMessageBox.information(self, f"MIST Analysis Results - {analysis_type.title()}",
                                        results_text)
                self.status_bar.showMessage(
                    f"{analysis_type} analysis completed")
            else:
                self.status_bar.showMessage(f"{analysis_type} analysis failed")
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error",
                                 f"Failed to run {analysis_type} analysis:\n{str(e)}")

    def format_analysis_results(self, results: dict, analysis_type: str) -> str:
        """Format analysis results for display."""
        if not results:
            return "No results available."

        formatted = f"MIST Analysis Results ({analysis_type.title()})\n" + \
            "="*50 + "\n\n"

        if analysis_type == 'connectivity':
            connectivity = results.get('connectivity_analysis', {})
            formatted += f"Average Coordination Number: {connectivity.get('avg_coordination', 'N/A')}\n"
            formatted += f"Max Coordination Number: {connectivity.get('max_coordination', 'N/A')}\n"
            formatted += f"Min Coordination Number: {connectivity.get('min_coordination', 'N/A')}\n"

        elif analysis_type == 'size_distribution':
            size_dist = results.get('size_distribution', {})
            formatted += f"Mean Size: {size_dist.get('mean', 'N/A'):.3f}\n"
            formatted += f"Standard Deviation: {size_dist.get('std', 'N/A'):.3f}\n"
            formatted += f"25th Percentile: {size_dist.get('percentile_25', 'N/A'):.3f}\n"
            formatted += f"75th Percentile: {size_dist.get('percentile_75', 'N/A'):.3f}\n"

        elif analysis_type == 'clustering':
            clustering = results.get('clustering_analysis', {})
            formatted += f"Number of Clusters: {clustering.get('n_clusters', 'N/A')}\n"
            formatted += f"Silhouette Score: {clustering.get('silhouette_score', 'N/A'):.3f}\n"

        elif analysis_type == 'full':
            # Show summary of all analyses
            for key, value in results.items():
                if isinstance(value, dict):
                    formatted += f"{key.title()}:\n"
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, float):
                            formatted += f"  {subkey}: {subvalue:.3f}\n"
                        else:
                            formatted += f"  {subkey}: {subvalue}\n"
                    formatted += "\n"

        return formatted


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Professional Pore Network Visualizer")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Scientific Visualization Lab")

    # Set application icon if available
    try:
        app.setWindowIcon(QIcon("assets/icon.png"))
    except:
        pass

    # Apply dark theme
    app.setStyle('Fusion')

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

    # Create and show main window
    window = PoreVisualizerGUI()
    window.show()

    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
