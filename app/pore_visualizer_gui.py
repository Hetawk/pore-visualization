#!/usr/bin/env python3
"""
MIST-like PyQt GUI Application for Interactive 3D Pore Network Visualization
Professional interface for scientific data analysis and visualization
"""

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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
import matplotlib
# Configure matplotlib to prevent popup windows
matplotlib.use('Agg')
# Disable interactive mode completely
plt.ioff()
# Set backend to non-interactive
matplotlib.rcParams['backend'] = 'Agg'

matplotlib.rcParams['interactive'] = False


# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


# Configure matplotlib backend before importing PyQt5
matplotlib.use('Agg')


class VisualizationThread(QThread):
    """Background thread for heavy visualization tasks."""

    # Changed from Figure to object (dict with plot data)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, engine: VisualizationEngine, data_path: str, viz_type: str, **kwargs):
        super().__init__()
        self.engine = engine
        self.data_path = data_path
        self.viz_type = viz_type
        self.kwargs = kwargs

    def run(self):
        try:
            self.progress.emit(25)
            # Handle volumetric visualization in background thread to prevent GUI freezing
            print(
                f"VisualizationThread: Creating visualization of type: {self.viz_type}")
            if self.viz_type == '3d_volumetric':
                print(
                    "VisualizationThread: Processing 3D volumetric visualization in background thread")
                self.progress.emit(50)
                # Get sample type from kwargs, default to 'all'
                sample_type = self.kwargs.get('sample_type', 'all')
                print(f"VisualizationThread: Sample type: {sample_type}")

                # Import matplotlib threading fix
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend in thread

                # Call the method that exists in the engine
                figure = self.engine.create_volumetric_pore_visualization_with_options(
                    self.data_path, sample_type)

                plot_data = {
                    'viz_type': self.viz_type,
                    'data_path': self.data_path,
                    'status': 'complete',
                    'figure': figure
                }
                self.progress.emit(100)
                print(
                    "VisualizationThread: Volumetric visualization completed, emitting finished signal")
                self.finished.emit(plot_data)
            else:
                # For other visualizations, defer to main thread to avoid threading issues
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
                # Enhanced error logging
                print(f"Failed to log thread error: {log_err}")
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

        # 3D Volumetric Visualization Group
        volumetric_group = QGroupBox("3D Volumetric Visualization")
        volumetric_layout = QGridLayout()
        volumetric_layout.setSpacing(6)

        # Sample selection for volumetric visualization
        volumetric_layout.addWidget(QLabel("Sample Type:"), 0, 0)
        self.volumetric_sample_combo = QComboBox()
        self.volumetric_sample_combo.addItems(
            ["All Samples", "T1 Only", "T2 Only", "T3 Only"])
        self.volumetric_sample_combo.setMinimumHeight(25)
        volumetric_layout.addWidget(self.volumetric_sample_combo, 0, 1)

        # Volumetric visualization buttons
        self.create_volumetric_btn = QPushButton("Create 3D Volumetric View")
        self.create_volumetric_btn.setMinimumHeight(30)
        self.create_volumetric_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #3498db; color: white; }")
        self.create_volumetric_btn.clicked.connect(
            lambda: self.preset_mode_changed.emit('3d_volumetric'))
        volumetric_layout.addWidget(self.create_volumetric_btn, 1, 0, 1, 2)

        # Individual sample buttons
        self.volumetric_t1_btn = QPushButton("T1 Volumetric")
        self.volumetric_t1_btn.setMinimumHeight(25)
        self.volumetric_t1_btn.setStyleSheet(
            "QPushButton { background-color: #e74c3c; color: white; }")
        self.volumetric_t1_btn.clicked.connect(
            lambda: self.preset_mode_changed.emit('volumetric_t1'))
        volumetric_layout.addWidget(self.volumetric_t1_btn, 2, 0)

        self.volumetric_t2_btn = QPushButton("T2 Volumetric")
        self.volumetric_t2_btn.setMinimumHeight(25)
        self.volumetric_t2_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; }")
        self.volumetric_t2_btn.clicked.connect(
            lambda: self.preset_mode_changed.emit('volumetric_t2'))
        volumetric_layout.addWidget(self.volumetric_t2_btn, 2, 1)

        self.volumetric_t3_btn = QPushButton("T3 Volumetric")
        self.volumetric_t3_btn.setMinimumHeight(25)
        self.volumetric_t3_btn.setStyleSheet(
            "QPushButton { background-color: #f39c12; color: white; }")
        self.volumetric_t3_btn.clicked.connect(
            lambda: self.preset_mode_changed.emit('volumetric_t3'))
        volumetric_layout.addWidget(self.volumetric_t3_btn, 3, 0, 1, 2)

        volumetric_group.setLayout(volumetric_layout)
        layout.addWidget(volumetric_group)

        # Add stretch and set layout
        layout.addStretch()
        self.setLayout(layout)

        # Add Interactive 3D Controls Group
        interactive_group = QGroupBox("Interactive 3D Controls")
        interactive_layout = QGridLayout()
        interactive_layout.setSpacing(6)

        # View Controls
        view_label = QLabel("Camera & View Controls")
        view_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        interactive_layout.addWidget(view_label, 0, 0, 1, 2)

        # Elevation angle (vertical rotation)
        interactive_layout.addWidget(QLabel("Elevation (°):"), 1, 0)
        self.elevation_slider = QSlider(Qt.Horizontal)
        self.elevation_slider.setRange(-90, 90)
        self.elevation_slider.setValue(30)
        self.elevation_slider.setTickPosition(QSlider.TicksBelow)
        self.elevation_slider.setTickInterval(30)
        self.elevation_slider.valueChanged.connect(
            lambda v: self.on_view_changed('elevation', v))
        interactive_layout.addWidget(self.elevation_slider, 1, 1)

        # Azimuth angle (horizontal rotation)
        interactive_layout.addWidget(QLabel("Azimuth (°):"), 2, 0)
        self.azimuth_slider = QSlider(Qt.Horizontal)
        self.azimuth_slider.setRange(-180, 180)
        self.azimuth_slider.setValue(-60)
        self.azimuth_slider.setTickPosition(QSlider.TicksBelow)
        self.azimuth_slider.setTickInterval(60)
        self.azimuth_slider.valueChanged.connect(
            lambda v: self.on_view_changed('azimuth', v))
        interactive_layout.addWidget(self.azimuth_slider, 2, 1)

        # Roll angle (camera roll)
        interactive_layout.addWidget(QLabel("Roll (°):"), 3, 0)
        self.roll_slider = QSlider(Qt.Horizontal)
        self.roll_slider.setRange(-180, 180)
        self.roll_slider.setValue(0)
        self.roll_slider.setTickPosition(QSlider.TicksBelow)
        self.roll_slider.setTickInterval(60)
        self.roll_slider.valueChanged.connect(
            lambda v: self.on_view_changed('roll', v))
        interactive_layout.addWidget(self.roll_slider, 3, 1)

        # Distance/Zoom
        interactive_layout.addWidget(QLabel("Zoom Level:"), 4, 0)
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 200)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setTickPosition(QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(25)
        self.zoom_slider.valueChanged.connect(
            lambda v: self.on_view_changed('zoom', v))
        interactive_layout.addWidget(self.zoom_slider, 4, 1)

        # Flip Controls
        flip_label = QLabel("Flip & Mirror Controls")
        flip_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        interactive_layout.addWidget(flip_label, 5, 0, 1, 2)

        # Flip buttons layout
        flip_widget = QWidget()
        flip_layout = QHBoxLayout()
        flip_layout.setContentsMargins(0, 0, 0, 0)
        flip_layout.setSpacing(4)

        self.flip_x_btn = QPushButton("Flip X")
        self.flip_x_btn.setCheckable(True)
        self.flip_x_btn.clicked.connect(lambda: self.on_flip_changed('x'))
        flip_layout.addWidget(self.flip_x_btn)

        self.flip_y_btn = QPushButton("Flip Y")
        self.flip_y_btn.setCheckable(True)
        self.flip_y_btn.clicked.connect(lambda: self.on_flip_changed('y'))
        flip_layout.addWidget(self.flip_y_btn)

        self.flip_z_btn = QPushButton("Flip Z")
        self.flip_z_btn.setCheckable(True)
        self.flip_z_btn.clicked.connect(lambda: self.on_flip_changed('z'))
        flip_layout.addWidget(self.flip_z_btn)

        flip_widget.setLayout(flip_layout)
        interactive_layout.addWidget(flip_widget, 6, 0, 1, 2)

        # Preset View Buttons
        preset_view_label = QLabel("Preset Views")
        preset_view_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        interactive_layout.addWidget(preset_view_label, 7, 0, 1, 2)

        preset_widget = QWidget()
        preset_layout = QGridLayout()
        preset_layout.setContentsMargins(0, 0, 0, 0)
        preset_layout.setSpacing(4)

        # View preset buttons
        self.front_view_btn = QPushButton("Front")
        self.front_view_btn.clicked.connect(
            lambda: self.set_preset_view('front'))
        preset_layout.addWidget(self.front_view_btn, 0, 0)

        self.back_view_btn = QPushButton("Back")
        self.back_view_btn.clicked.connect(
            lambda: self.set_preset_view('back'))
        preset_layout.addWidget(self.back_view_btn, 0, 1)

        self.left_view_btn = QPushButton("Left")
        self.left_view_btn.clicked.connect(
            lambda: self.set_preset_view('left'))
        preset_layout.addWidget(self.left_view_btn, 1, 0)

        self.right_view_btn = QPushButton("Right")
        self.right_view_btn.clicked.connect(
            lambda: self.set_preset_view('right'))
        preset_layout.addWidget(self.right_view_btn, 1, 1)

        self.top_view_btn = QPushButton("Top")
        self.top_view_btn.clicked.connect(lambda: self.set_preset_view('top'))
        preset_layout.addWidget(self.top_view_btn, 2, 0)

        self.bottom_view_btn = QPushButton("Bottom")
        self.bottom_view_btn.clicked.connect(
            lambda: self.set_preset_view('bottom'))
        preset_layout.addWidget(self.bottom_view_btn, 2, 1)

        self.isometric_view_btn = QPushButton("Isometric")
        self.isometric_view_btn.clicked.connect(
            lambda: self.set_preset_view('isometric'))
        preset_layout.addWidget(self.isometric_view_btn, 3, 0, 1, 2)

        preset_widget.setLayout(preset_layout)
        interactive_layout.addWidget(preset_widget, 8, 0, 1, 2)

        # Real-time update checkbox
        self.realtime_update = QCheckBox("Real-time Updates")
        self.realtime_update.setChecked(True)
        self.realtime_update.setToolTip(
            "Update visualization in real-time as you move sliders")
        interactive_layout.addWidget(self.realtime_update, 9, 0, 1, 2)

        # Animation controls
        animation_label = QLabel("Animation Controls")
        animation_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        interactive_layout.addWidget(animation_label, 10, 0, 1, 2)

        # Animation buttons
        anim_widget = QWidget()
        anim_layout = QHBoxLayout()
        anim_layout.setContentsMargins(0, 0, 0, 0)
        anim_layout.setSpacing(4)

        self.rotate_anim_btn = QPushButton("Auto Rotate")
        self.rotate_anim_btn.setCheckable(True)
        self.rotate_anim_btn.clicked.connect(self.toggle_auto_rotation)
        anim_layout.addWidget(self.rotate_anim_btn)

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view_to_default)
        anim_layout.addWidget(self.reset_view_btn)

        anim_widget.setLayout(anim_layout)
        interactive_layout.addWidget(anim_widget, 11, 0, 1, 2)

        interactive_group.setLayout(interactive_layout)
        layout.addWidget(interactive_group)

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

        # Initialize enhanced visualization engine for realistic rendering
        try:
            from core.live_3d_renderer import LivePore3DRenderer
            self.live_3d_renderer = LivePore3DRenderer()
            self.logger.info("✓ Live 3D renderer initialized")
        except Exception as e:
            self.logger.warning(f"Live 3D renderer not available: {e}")
            self.live_3d_renderer = None

        # Try enhanced visualization engine
        try:
            from core.enhanced_visualization_engine import EnhancedPoreVisualizationEngine
            self.enhanced_viz_engine = EnhancedPoreVisualizationEngine()
            self.embedded_opengl_widget = self.enhanced_viz_engine.get_embedded_widget()
            self.logger.info("✓ Enhanced visualization engine initialized")
        except Exception as e:
            self.logger.warning(f"Enhanced visualization not available: {e}")
            self.enhanced_viz_engine = None
            self.embedded_opengl_widget = None

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
        layout.addWidget(file_group)        # Visualization type
        viz_group = QGroupBox("Visualization Type")
        viz_layout = QVBoxLayout()

        self.viz_type = QComboBox()
        self.viz_type.addItems([
            'pore_network', 'size_distribution', 'clustering_analysis',
            'ultra_realistic', 'scientific_analysis', 'presentation', 'cross_section',
            '3d_volumetric'
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
        """Create the right visualization panel with tabbed interface."""
        panel = QWidget()
        layout = QVBoxLayout()

        # Create tabbed interface for different visualization modes
        self.viz_tabs = QTabWidget()

        # Tab 1: Traditional matplotlib visualizations
        matplotlib_tab = QWidget()
        matplotlib_layout = QVBoxLayout()        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Create a scroll area for the canvas to handle large visualizations
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.canvas)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        matplotlib_layout.addWidget(self.toolbar)
        matplotlib_layout.addWidget(scroll_area)
        matplotlib_tab.setLayout(matplotlib_layout)

        self.viz_tabs.addTab(matplotlib_tab, "2D/3D Plots")

        # Tab 2: Live realistic 3D rendering (always available)
        if self.live_3d_renderer:
            self.viz_tabs.addTab(self.live_3d_renderer, "Live 3D View")

        # Tab 3: Enhanced realistic 3D rendering (if OpenGL available)
        if self.embedded_opengl_widget:
            self.viz_tabs.addTab(
                self.embedded_opengl_widget, "Advanced 3D View")

            # Add controls for the realistic view
            realistic_controls = QWidget()
            controls_layout = QHBoxLayout()

            # Connection threshold control
            controls_layout.addWidget(QLabel("Bond Threshold:"))
            self.bond_threshold_slider = QSlider(Qt.Horizontal)
            self.bond_threshold_slider.setRange(10, 50)
            self.bond_threshold_slider.setValue(20)
            self.bond_threshold_slider.valueChanged.connect(
                self.update_realistic_view)
            controls_layout.addWidget(self.bond_threshold_slider)

            # Pore size range controls
            controls_layout.addWidget(QLabel("Min Size (nm):"))
            self.min_size_spin = QSpinBox()
            self.min_size_spin.setRange(1, 1000)
            self.min_size_spin.setValue(10)
            self.min_size_spin.valueChanged.connect(self.update_realistic_view)
            controls_layout.addWidget(self.min_size_spin)

            controls_layout.addWidget(QLabel("Max Size (nm):"))
            self.max_size_spin = QSpinBox()
            self.max_size_spin.setRange(100, 10000)
            self.max_size_spin.setValue(1000)
            self.max_size_spin.valueChanged.connect(self.update_realistic_view)
            controls_layout.addWidget(self.max_size_spin)

            # Board type selection for realistic view
            controls_layout.addWidget(QLabel("Board Type:"))
            self.realistic_board_combo = QComboBox()
            self.realistic_board_combo.addItems(["T1", "T2", "T3"])
            self.realistic_board_combo.currentTextChanged.connect(
                self.update_realistic_view)
            controls_layout.addWidget(self.realistic_board_combo)

            # Update button
            update_realistic_btn = QPushButton("Update Realistic View")
            update_realistic_btn.clicked.connect(
                self.create_realistic_visualization)
            controls_layout.addWidget(update_realistic_btn)

            controls_layout.addStretch()
            realistic_controls.setLayout(controls_layout)

            # Add controls above the OpenGL widget
            realistic_tab_layout = QVBoxLayout()
            realistic_tab_layout.addWidget(realistic_controls)
            realistic_tab_layout.addWidget(self.embedded_opengl_widget, 1)

            realistic_tab = QWidget()
            realistic_tab.setLayout(realistic_tab_layout)
            self.viz_tabs.removeTab(1)  # Remove the direct widget
            self.viz_tabs.addTab(realistic_tab, "Realistic 3D View")

        layout.addWidget(self.viz_tabs)
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

        # Add realistic visualization actions if available
        if self.enhanced_viz_engine:
            toolbar.addSeparator()

            realistic_action = QAction('Realistic 3D', self)
            realistic_action.setToolTip(
                'Create realistic 3D pore network visualization')
            realistic_action.triggered.connect(
                self.create_realistic_visualization)
            toolbar.addAction(realistic_action)

            auto_refresh_action = QAction('Auto-Refresh', self)
            auto_refresh_action.setCheckable(True)
            auto_refresh_action.setToolTip(
                'Toggle automatic refresh of realistic view')
            auto_refresh_action.triggered.connect(self.toggle_auto_refresh)
            toolbar.addAction(auto_refresh_action)

            stats_action = QAction('Network Stats', self)
            stats_action.setToolTip('Show network statistics')
            stats_action.triggered.connect(self.show_network_statistics)
            toolbar.addAction(stats_action)

        toolbar.addSeparator()

        export_action = QAction('Export', self)
        export_action.triggered.connect(self.export_visualization)
        toolbar.addAction(export_action)

        if self.enhanced_viz_engine:
            export_realistic_action = QAction('Export 3D', self)
            export_realistic_action.setToolTip(
                'Export high-resolution realistic view')
            export_realistic_action.triggered.connect(
                self.export_realistic_view)
            toolbar.addAction(export_realistic_action)

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

                # Load data into live 3D renderer
                if self.live_3d_renderer:
                    if self.live_3d_renderer.load_data(file_path):
                        self.logger.info("✓ Live 3D renderer data loaded")
                        # Switch to Live 3D tab and show it
                        if self.viz_tabs.count() > 1:
                            self.viz_tabs.setCurrentIndex(1)  # Live 3D tab

                # Load data into enhanced engine if available
                if self.enhanced_viz_engine:
                    self.enhanced_viz_engine.load_data(file_path)

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

    def generate_visualization_with_type(self, viz_type, **kwargs):
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

        # For volumetric visualization, get sample type from combo box
        if viz_type == '3d_volumetric' and 'sample_type' not in kwargs:
            if hasattr(self, 'volumetric_sample_combo'):
                combo_text = self.volumetric_sample_combo.currentText()
                if combo_text == "All Samples":
                    kwargs['sample_type'] = 'all'
                elif combo_text == "T1 Only":
                    kwargs['sample_type'] = 'T1'
                elif combo_text == "T2 Only":
                    kwargs['sample_type'] = 'T2'
                elif combo_text == "T3 Only":
                    kwargs['sample_type'] = 'T3'
                else:
                    kwargs['sample_type'] = 'all'
            else:
                kwargs['sample_type'] = 'all'

        # Start visualization in background thread
        self.visualization_thread = VisualizationThread(
            self.visualization_engine, self.current_data_path, viz_type, **kwargs
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

            elif mode == 'ultra_realistic':
                # Use enhanced realistic visualization
                if self.enhanced_viz_engine:
                    self.progress_bar.setValue(50)
                    self.create_realistic_visualization()
                    self.progress_bar.setVisible(False)
                    return
                else:                    # Fallback to regular visualization
                    self.generate_visualization_with_type(mode)

            elif mode in ['scientific_analysis', 'presentation', 'cross_section']:
                # Use advanced visualization engine
                if hasattr(self.visualization_engine, 'create_advanced_visualization'):
                    fig = self.visualization_engine.create_advanced_visualization(
                        self.current_data_path, mode)
                    self.on_visualization_finished(fig)
                else:
                    # Fallback to regular visualization - use thread
                    self.generate_visualization_with_type(mode)

            elif mode == '3d_volumetric':
                # Create 3D volumetric visualization for all samples using background thread
                self.generate_visualization_with_type(
                    '3d_volumetric', sample_type='all')

            elif mode == 'volumetric_t1':
                # Create 3D volumetric visualization for T1 sample only using background thread
                self.generate_visualization_with_type(
                    '3d_volumetric', sample_type='T1')

            elif mode == 'volumetric_t2':
                # Create 3D volumetric visualization for T2 sample only using background thread
                self.generate_visualization_with_type(
                    '3d_volumetric', sample_type='T2')

            elif mode == 'volumetric_t3':
                # Create 3D volumetric visualization for T3 sample only using background thread
                self.generate_visualization_with_type(
                    '3d_volumetric', sample_type='T3')

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
                        f"MainThread: Creating {plot_data['viz_type']} visualization in main thread")

                    # Handle different visualization types correctly
                    viz_type = plot_data['viz_type']
                    if viz_type == '3d_volumetric':
                        print(
                            "MainThread: ERROR - Volumetric visualization should never run on main thread!")
                        print(
                            "MainThread: This indicates a bug in the threading logic.")
                        # Don't process volumetric on main thread - this should not happen
                        figure = None
                    else:
                        # Default to pore network visualization for other types
                        figure = self.visualization_engine.create_pore_network_visualization(
                            plot_data['data_path'], viz_type)
                elif plot_data.get('status') == 'complete':
                    print(
                        "MainThread: Received completed visualization from background thread")
                    figure = plot_data.get('figure')
                else:
                    figure = None
            else:
                # Backward compatibility - assume it's a figure
                figure = plot_data

            # Handle the generated figure
            # Check if this is a volumetric visualization with multiple subplots
            if figure and len(figure.axes) > 0:
                if len(figure.axes) > 1:
                    # For volumetric visualizations with multiple subplots,
                    # replace the entire canvas figure instead of copying elements
                    self.figure = figure

                    # Update the canvas to use the new figure
                    self.canvas.figure = self.figure

                    # Ensure the canvas size is appropriate for the figure
                    self.canvas.updateGeometry()

                    # Force canvas refresh and resize
                    self.canvas.draw()
                    self.canvas.flush_events()

                    print(
                        f"MainThread: Loaded volumetric visualization with {len(figure.axes)} subplots")
                else:  # For single subplot figures, copy elements as before
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

        # Handle new interactive controls
        elif param_name.startswith('view_'):
            self.handle_view_change(param_name, value)
            return
        elif param_name.startswith('flip_'):
            self.handle_flip_change(param_name, value)
            return
        elif param_name == 'preset_view':
            self.handle_preset_view(value)
            return
        elif param_name == 'auto_rotation':
            self.handle_auto_rotation(value)
            return
        elif param_name == 'reset_view':
            self.handle_reset_view()
            return

        # Standard parameter update
        self.visualization_engine.update_parameters({param_name: value})
        self.status_bar.showMessage(f"Updated {param_name}: {value}")

        # Auto-refresh visualization if data is loaded and real-time updates are enabled
        if (self.current_data_path and
            hasattr(self.parameter_widget, 'realtime_update') and
                self.parameter_widget.realtime_update.isChecked()):
            # Debounce rapid parameter changes by using a timer
            if not hasattr(self, 'refresh_timer'):
                self.refresh_timer = QTimer()
                self.refresh_timer.setSingleShot(True)
                self.refresh_timer.timeout.connect(self._do_real_time_refresh)

            # Restart timer on each parameter change to debounce
            self.refresh_timer.stop()
            # 300ms delay to debounce rapid changes (reduced for more responsive feel)
            self.refresh_timer.start(300)

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

    def create_realistic_visualization(self):
        """Create realistic 3D pore network visualization."""
        if not self.enhanced_viz_engine or not self.current_data_path:
            QMessageBox.warning(self, "Warning",
                                "Enhanced visualization not available or no data loaded.")
            return

        try:
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)

            # Load data into enhanced engine
            if self.enhanced_viz_engine.load_data(self.current_data_path):
                self.progress_bar.setValue(30)

                # Get parameters from UI
                board_type = self.realistic_board_combo.currentText()
                connection_threshold = self.bond_threshold_slider.value() / 10.0
                min_size = self.min_size_spin.value()
                max_size = self.max_size_spin.value()

                self.progress_bar.setValue(50)

                # Create realistic network
                pore_data, bond_data = self.enhanced_viz_engine.create_realistic_pore_network(
                    board_type=board_type,
                    connection_threshold=connection_threshold,
                    min_pore_size=min_size,
                    max_pore_size=max_size
                )

                self.progress_bar.setValue(80)

                # Create ultra-realistic view
                if self.enhanced_viz_engine.create_ultra_realistic_view():
                    self.progress_bar.setValue(100)

                    # Update status with network statistics
                    stats = self.enhanced_viz_engine.get_network_statistics()
                    status_msg = f"Realistic view: {stats.get('total_pores', 0)} pores, {stats.get('total_bonds', 0)} bonds"
                    self.status_bar.showMessage(status_msg)

                    # Switch to realistic view tab
                    self.viz_tabs.setCurrentIndex(1)

                    self.logger.info(
                        "✓ Realistic visualization created successfully")
                else:
                    QMessageBox.warning(
                        self, "Warning", "Failed to create realistic view.")
            else:
                QMessageBox.warning(
                    self, "Warning", "Failed to load data for realistic visualization.")

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to create realistic visualization:\n{str(e)}")
            self.logger.error(f"Realistic visualization error: {e}")
        finally:
            self.progress_bar.setVisible(False)

    def update_realistic_view(self):
        """Update realistic view when parameters change."""
        if self.enhanced_viz_engine and self.current_data_path:
            # Only update if auto-refresh is enabled or explicitly requested
            if self.auto_refresh_enabled:
                self.create_realistic_visualization()

    def toggle_auto_refresh(self):
        """Toggle automatic refresh of realistic view."""
        self.auto_refresh_enabled = not self.auto_refresh_enabled
        status = "enabled" if self.auto_refresh_enabled else "disabled"
        self.status_bar.showMessage(f"Auto-refresh {status}")

    def export_realistic_view(self):
        """Export high-resolution realistic visualization."""
        if not self.enhanced_viz_engine:
            QMessageBox.warning(
                self, "Warning", "Enhanced visualization not available.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Realistic View",
            "realistic_pore_visualization.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )

        if file_path:
            try:
                if self.enhanced_viz_engine.export_realistic_visualization(file_path, (1920, 1080)):
                    QMessageBox.information(
                        self, "Success", f"Realistic view exported to:\n{file_path}")
                else:
                    QMessageBox.warning(
                        self, "Warning", "Failed to export realistic view.")
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Export failed:\n{str(e)}")

    def show_network_statistics(self):
        """Show detailed network statistics."""
        if not self.enhanced_viz_engine:
            return

        stats = self.enhanced_viz_engine.get_network_statistics()
        if not stats:
            QMessageBox.information(self, "Info", "No network data available.")
            return

        stats_text = f"""
Network Statistics:

Total Pores: {stats.get('total_pores', 0)}
Total Bonds: {stats.get('total_bonds', 0)}

Average Pore Size: {stats.get('average_pore_size', 0):.2f} nm
Pore Size Std Dev: {stats.get('pore_size_std', 0):.2f} nm

Average Bond Length: {stats.get('average_bond_length', 0):.3f}
Network Connectivity: {stats.get('connectivity', 0):.3f}
Network Density: {stats.get('network_density', 0):.6f}
        """

        QMessageBox.information(self, "Network Statistics", stats_text)

    def on_view_changed(self, param, value):
        """Handle view parameter changes (elevation, azimuth, roll, zoom)."""
        if self.realtime_update.isChecked():
            self.parameter_changed.emit(f'view_{param}', value)

    def on_flip_changed(self, axis):
        """Handle flip/mirror operations."""
        button = getattr(self, f'flip_{axis}_btn')
        is_flipped = button.isChecked()
        self.parameter_changed.emit(f'flip_{axis}', is_flipped)

    def set_preset_view(self, view_name):
        """Set camera to a preset view position."""
        # Define preset view angles
        preset_views = {
            'front': {'elevation': 0, 'azimuth': 0, 'roll': 0},
            'back': {'elevation': 0, 'azimuth': 180, 'roll': 0},
            'left': {'elevation': 0, 'azimuth': -90, 'roll': 0},
            'right': {'elevation': 0, 'azimuth': 90, 'roll': 0},
            'top': {'elevation': 90, 'azimuth': 0, 'roll': 0},
            'bottom': {'elevation': -90, 'azimuth': 0, 'roll': 0},
            'isometric': {'elevation': 30, 'azimuth': -60, 'roll': 0}
        }

        if view_name in preset_views:
            view = preset_views[view_name]

            # Update sliders without triggering signals
            self.elevation_slider.blockSignals(True)
            self.azimuth_slider.blockSignals(True)
            self.roll_slider.blockSignals(True)

            self.elevation_slider.setValue(view['elevation'])
            self.azimuth_slider.setValue(view['azimuth'])
            self.roll_slider.setValue(view['roll'])

            self.elevation_slider.blockSignals(False)
            self.azimuth_slider.blockSignals(False)
            self.roll_slider.blockSignals(False)

            # Emit the preset view change
            self.parameter_changed.emit('preset_view', view_name)

    def toggle_auto_rotation(self):
        """Toggle automatic rotation animation."""
        is_enabled = self.rotate_anim_btn.isChecked()
        self.parameter_changed.emit('auto_rotation', is_enabled)

    def reset_view_to_default(self):
        """Reset all view parameters to default values."""
        # Reset sliders to default positions
        self.elevation_slider.setValue(30)
        self.azimuth_slider.setValue(-60)
        self.roll_slider.setValue(0)
        self.zoom_slider.setValue(100)

        # Reset flip buttons
        self.flip_x_btn.setChecked(False)
        self.flip_y_btn.setChecked(False)
        self.flip_z_btn.setChecked(False)

        # Emit reset signal
        self.parameter_changed.emit('reset_view', True)


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
