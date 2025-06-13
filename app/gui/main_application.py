#!/usr/bin/env python3
"""
MIST-like PyQt GUI Application for Interactive 3D Pore Network Visualization
Professional interface for scientific data analysis and visualization
"""

from core.pore_analyzer import PoreAnalyzer
from core.visualization_engine import VisualizationEngine
from core.data_manager import DataManager
from core.high_performance_renderer import HighPerformanceRenderer, create_spheres_from_pore_data, generate_bonds_from_spheres
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QSplitter, QTabWidget, QGroupBox, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QPushButton, QCheckBox, QFileDialog, QTextEdit, QProgressBar,
    QStatusBar, QMenuBar, QAction, QToolBar, QColorDialog, QMessageBox,
    QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSettings
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPalette, QColor, QImage
import matplotlib
import os
import logging
import traceback
import time
import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports BEFORE any other imports
sys.path.append(str(Path(__file__).parent.parent))


# Configure matplotlib backend before importing PyQt5
matplotlib.use('Qt5Agg')


# Import our core modules after path setup


class VisualizationThread(QThread):
    """Background thread for heavy visualization tasks."""

    finished = pyqtSignal(Figure)
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
            # Create visualization with debug info
            print(f"Creating visualization of type: {self.viz_type}")
            fig = self.engine.create_pore_network_visualization(
                self.data_path, self.viz_type)

            # Debug: Check if figure has data
            if fig and len(fig.axes) > 0:
                ax = fig.axes[0]
                print(
                    f"Figure created with {len(ax.collections)} collections and {len(ax.lines)} lines")
                for i, collection in enumerate(ax.collections):
                    if hasattr(collection, '_offsets3d'):
                        xs, ys, zs = collection._offsets3d
                        print(
                            f"Collection {i}: 3D scatter with {len(xs)} points")
                    elif hasattr(collection, '_offsets'):
                        print(
                            f"Collection {i}: 2D scatter with {len(collection._offsets)} points")
            else:
                print("Warning: Figure is None or has no axes")

            self.progress.emit(100)
            self.finished.emit(fig)
        except Exception as e:
            import traceback
            error_msg = f"Visualization error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.error.emit(error_msg)


class ParameterControlWidget(QWidget):
    """Widget for controlling visualization parameters."""

    parameter_changed = pyqtSignal(str, object)

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

        # Size multiplier
        sphere_layout.addWidget(QLabel("Size Multiplier:"), 2, 0)
        self.size_multiplier = QDoubleSpinBox()
        self.size_multiplier.setRange(0.1, 3.0)
        self.size_multiplier.setSingleStep(0.1)
        self.size_multiplier.setValue(1.0)
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

        # Prism Parameters
        prism_group = QGroupBox("Prism Parameters")
        prism_layout = QGridLayout()
        prism_layout.setSpacing(6)

        # Prism color
        prism_layout.addWidget(QLabel("Prism Color:"), 0, 0)
        self.prism_color_btn = QPushButton("Orange")
        self.prism_color_btn.setStyleSheet("background-color: orange")
        self.prism_color_btn.setMinimumHeight(25)
        self.prism_color_btn.clicked.connect(self.choose_prism_color)
        prism_layout.addWidget(self.prism_color_btn, 0, 1)

        # Prism opacity
        prism_layout.addWidget(QLabel("Prism Opacity:"), 1, 0)
        self.prism_opacity = QDoubleSpinBox()
        self.prism_opacity.setRange(0.1, 1.0)
        self.prism_opacity.setSingleStep(0.1)
        self.prism_opacity.setValue(0.3)
        self.prism_opacity.setMinimumHeight(25)
        self.prism_opacity.valueChanged.connect(
            lambda v: self.parameter_changed.emit('prism_opacity', v))
        prism_layout.addWidget(self.prism_opacity, 1, 1)

        prism_group.setLayout(prism_layout)
        layout.addWidget(prism_group)

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

        # View Parameters - Ensure this section is fully visible
        view_group = QGroupBox("View Parameters")
        view_layout = QGridLayout()
        view_layout.setSpacing(6)

        # Lighting intensity
        view_layout.addWidget(QLabel("Lighting Intensity:"), 0, 0)
        self.lighting_intensity = QDoubleSpinBox()
        self.lighting_intensity.setRange(0.1, 2.0)
        self.lighting_intensity.setSingleStep(0.1)
        self.lighting_intensity.setValue(0.8)
        self.lighting_intensity.setMinimumHeight(25)
        self.lighting_intensity.valueChanged.connect(
            lambda v: self.parameter_changed.emit('lighting_intensity', v))
        view_layout.addWidget(self.lighting_intensity, 0, 1)

        # Show coordinate system
        self.show_coords = QCheckBox("Show Coordinate System")
        self.show_coords.setChecked(True)
        self.show_coords.toggled.connect(
            lambda v: self.parameter_changed.emit('show_coordinate_system', v))
        view_layout.addWidget(self.show_coords, 1, 0, 1, 2)

        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # Enhanced Visual Effects Group
        enhanced_group = QGroupBox("Enhanced Visual Effects")
        enhanced_layout = QGridLayout()
        enhanced_layout.setSpacing(6)

        # Color scheme selection
        enhanced_layout.addWidget(QLabel("Color Scheme:"), 0, 0)
        self.color_scheme = QComboBox()
        self.color_scheme.addItems(
            ['scientific', 'artistic', 'rainbow', 'thermal', 'depth'])
        self.color_scheme.setCurrentText('scientific')
        self.color_scheme.setMinimumHeight(25)
        self.color_scheme.currentTextChanged.connect(
            lambda v: self.parameter_changed.emit('color_scheme', v))
        enhanced_layout.addWidget(self.color_scheme, 0, 1)

        # Sphere style
        enhanced_layout.addWidget(QLabel("Sphere Style:"), 1, 0)
        self.sphere_style = QComboBox()
        self.sphere_style.addItems(['glossy', 'matte', 'metallic', 'glass'])
        self.sphere_style.setCurrentText('glossy')
        self.sphere_style.setMinimumHeight(25)
        self.sphere_style.currentTextChanged.connect(
            lambda v: self.parameter_changed.emit('sphere_style', v))
        enhanced_layout.addWidget(self.sphere_style, 1, 1)

        # Bond style
        enhanced_layout.addWidget(QLabel("Bond Style:"), 2, 0)
        self.bond_style = QComboBox()
        self.bond_style.addItems(['tubes', 'lines', 'cylinders'])
        self.bond_style.setCurrentText('tubes')
        self.bond_style.setMinimumHeight(25)
        self.bond_style.currentTextChanged.connect(
            lambda v: self.parameter_changed.emit('bond_style', v))
        enhanced_layout.addWidget(self.bond_style, 2, 1)

        # Lighting model
        enhanced_layout.addWidget(QLabel("Lighting Model:"), 3, 0)
        self.lighting_model = QComboBox()
        self.lighting_model.addItems(['enhanced', 'basic', 'dramatic'])
        self.lighting_model.setCurrentText('enhanced')
        self.lighting_model.setMinimumHeight(25)
        self.lighting_model.currentTextChanged.connect(
            lambda v: self.parameter_changed.emit('lighting_model', v))
        enhanced_layout.addWidget(self.lighting_model, 3, 1)

        # Visual effects checkboxes
        self.depth_cueing = QCheckBox("Depth Cueing (Distance Fade)")
        self.depth_cueing.setChecked(True)
        self.depth_cueing.toggled.connect(
            lambda v: self.parameter_changed.emit('depth_cueing', v))
        enhanced_layout.addWidget(self.depth_cueing, 4, 0, 1, 2)

        self.edge_enhancement = QCheckBox("Edge Enhancement")
        self.edge_enhancement.setChecked(True)
        self.edge_enhancement.toggled.connect(
            lambda v: self.parameter_changed.emit('edge_enhancement', v))
        enhanced_layout.addWidget(self.edge_enhancement, 5, 0, 1, 2)

        # Size variance
        enhanced_layout.addWidget(QLabel("Size Variance:"), 6, 0)
        self.size_variance = QDoubleSpinBox()
        self.size_variance.setRange(0.0, 0.5)
        self.size_variance.setSingleStep(0.05)
        self.size_variance.setValue(0.2)
        self.size_variance.setMinimumHeight(25)
        self.size_variance.valueChanged.connect(
            lambda v: self.parameter_changed.emit('size_variance', v))
        enhanced_layout.addWidget(self.size_variance, 6, 1)

        enhanced_group.setLayout(enhanced_layout)
        layout.addWidget(enhanced_group)

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
            lambda: self.apply_preset_mode('ultra_realistic'))
        presets_layout.addWidget(self.ultra_realistic_btn, 0, 0)

        self.scientific_btn = QPushButton("Scientific Analysis")
        self.scientific_btn.setMinimumHeight(30)
        self.scientific_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #2ecc71; color: white; }")
        self.scientific_btn.clicked.connect(
            lambda: self.apply_preset_mode('scientific_analysis'))
        presets_layout.addWidget(self.scientific_btn, 0, 1)

        self.presentation_btn = QPushButton("Presentation")
        self.presentation_btn.setMinimumHeight(30)
        self.presentation_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #e74c3c; color: white; }")
        self.presentation_btn.clicked.connect(
            lambda: self.apply_preset_mode('presentation'))
        presets_layout.addWidget(self.presentation_btn, 1, 0)

        self.cross_section_btn = QPushButton("Cross Section")
        self.cross_section_btn.setMinimumHeight(30)
        self.cross_section_btn.setStyleSheet(
            "QPushButton { font-weight: bold; background-color: #9b59b6; color: white; }")
        self.cross_section_btn.clicked.connect(
            lambda: self.apply_preset_mode('cross_section'))
        presets_layout.addWidget(self.cross_section_btn, 1, 1)

        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

        # Quality Presets Group
        quality_group = QGroupBox("Performance & Quality")
        quality_layout = QGridLayout()
        quality_layout.setSpacing(6)

        # Quality preset buttons
        quality_layout.addWidget(QLabel("Quality Level:"), 0, 0)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(
            ['Low (Fast)', 'Medium', 'High', 'Ultra (Slow)'])
        self.quality_combo.setCurrentText('High')
        self.quality_combo.setMinimumHeight(25)
        self.quality_combo.currentTextChanged.connect(
            self.apply_quality_preset)
        quality_layout.addWidget(self.quality_combo, 0, 1)

        # Animation controls
        self.animation_mode = QCheckBox("Enable Rotation Animation")
        self.animation_mode.toggled.connect(
            lambda v: self.parameter_changed.emit('animation_mode', v))
        quality_layout.addWidget(self.animation_mode, 1, 0, 1, 2)

        # Export options
        self.export_hd_btn = QPushButton("Export HD Image")
        self.export_hd_btn.setMinimumHeight(25)
        self.export_hd_btn.clicked.connect(self.export_hd_visualization)
        quality_layout.addWidget(self.export_hd_btn, 2, 0)

        self.export_animation_btn = QPushButton("Export Animation")
        self.export_animation_btn.setMinimumHeight(25)
        self.export_animation_btn.clicked.connect(self.export_animation)
        quality_layout.addWidget(self.export_animation_btn, 2, 1)

        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

        # Advanced Features Group
        advanced_group = QGroupBox("Advanced Features")
        advanced_layout = QGridLayout()
        advanced_layout.setSpacing(6)

        # Surface rendering
        self.surface_rendering = QCheckBox("Surface Mesh Rendering")
        self.surface_rendering.toggled.connect(
            lambda v: self.parameter_changed.emit('surface_rendering', v))
        advanced_layout.addWidget(self.surface_rendering, 0, 0, 1, 2)

        # Volumetric effects
        self.volumetric_effects = QCheckBox("Volumetric Fog Effects")
        self.volumetric_effects.toggled.connect(
            lambda v: self.parameter_changed.emit('volumetric_effects', v))
        advanced_layout.addWidget(self.volumetric_effects, 1, 0, 1, 2)

        # Measurement tools
        self.measurement_tools = QCheckBox("Show Measurement Tools")
        self.measurement_tools.setChecked(True)
        self.measurement_tools.toggled.connect(
            lambda v: self.parameter_changed.emit('measurement_tools', v))
        advanced_layout.addWidget(self.measurement_tools, 2, 0, 1, 2)

        # Scientific annotations
        self.scientific_annotations = QCheckBox("Scientific Annotations")
        self.scientific_annotations.setChecked(True)
        self.scientific_annotations.toggled.connect(
            lambda v: self.parameter_changed.emit('scientific_annotations', v))
        advanced_layout.addWidget(self.scientific_annotations, 3, 0, 1, 2)

        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)

        # Add stretch and set layout
        layout.addStretch()
        self.setLayout(layout)

    # Signal for preset mode changes
    preset_mode_changed = pyqtSignal(str)

    def apply_preset_mode(self, mode):
        """Apply a preset visualization mode."""
        self.preset_mode_changed.emit(mode)

    def apply_quality_preset(self, quality_text):
        """Apply quality preset based on selection."""
        quality_map = {
            'Low (Fast)': 'low',
            'Medium': 'medium',
            'High': 'high',
            'Ultra (Slow)': 'ultra'
        }

        quality_level = quality_map.get(quality_text, 'high')

        # Update parameters based on quality level
        if quality_level == 'low':
            self.num_spheres.setValue(300)
            self.bond_thickness.setValue(0.3)
            self.depth_cueing.setChecked(False)
            self.edge_enhancement.setChecked(False)
            self.size_variance.setValue(0.1)
        elif quality_level == 'medium':
            self.num_spheres.setValue(600)
            self.bond_thickness.setValue(0.5)
            self.depth_cueing.setChecked(True)
            self.edge_enhancement.setChecked(False)
            self.size_variance.setValue(0.15)
        elif quality_level == 'high':
            self.num_spheres.setValue(1000)
            self.bond_thickness.setValue(0.7)
            self.depth_cueing.setChecked(True)
            self.edge_enhancement.setChecked(True)
            self.size_variance.setValue(0.2)
        elif quality_level == 'ultra':
            self.num_spheres.setValue(2000)
            self.bond_thickness.setValue(1.0)
            self.depth_cueing.setChecked(True)
            self.edge_enhancement.setChecked(True)
            self.size_variance.setValue(0.25)

    def export_hd_visualization(self):
        """Export high-definition visualization."""
        if not self.current_figure:
            QMessageBox.warning(self, "Warning", "No visualization to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export HD Visualization", "",
            "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;TIFF Files (*.tiff);;All Files (*)"
        )

        if file_path:
            try:
                # Export at high DPI for publication quality
                dpi = 300
                if file_path.lower().endswith('.png'):
                    dpi = 600  # Extra high for PNG

                self.visualization_engine.export_visualization(
                    file_path, dpi=dpi)
                QMessageBox.information(
                    self, "Success", f"HD visualization exported successfully!\nDPI: {dpi}")
                self.status_bar.showMessage(f"HD exported to {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Export Error",
                                     f"Failed to export HD visualization:\n{str(e)}")

    def export_animation(self):
        """Export rotating animation of the visualization."""
        if not self.current_data_path:
            QMessageBox.warning(
                self, "Warning", "Please load a data file first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Animation", "",
            "GIF Files (*.gif);;MP4 Files (*.mp4);;All Files (*)"
        )

        if file_path:
            try:
                self.status_bar.showMessage("Creating animation frames...")
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(0)

                # Create animation frames
                frames = self.visualization_engine.create_animation_frames(
                    self.current_data_path, num_frames=36)

                self.progress_bar.setValue(50)

                # Save animation
                if file_path.lower().endswith('.gif'):
                    self._save_gif_animation(frames, file_path)
                elif file_path.lower().endswith('.mp4'):
                    self._save_mp4_animation(frames, file_path)
                else:
                    # Default to GIF
                    self._save_gif_animation(frames, file_path + '.gif')

                self.progress_bar.setValue(100)
                self.progress_bar.setVisible(False)

                QMessageBox.information(
                    self, "Success", "Animation exported successfully!")
                self.status_bar.showMessage(
                    f"Animation exported to {file_path}")

            except Exception as e:
                self.progress_bar.setVisible(False)
                QMessageBox.critical(self, "Export Error",
                                     f"Failed to export animation:\n{str(e)}")

    def _save_gif_animation(self, frames, file_path):
        """Save frames as GIF animation."""
        try:
            import matplotlib.animation as animation
            from matplotlib.animation import PillowWriter

            # Create animation
            fig = frames[0]

            def animate(frame_num):
                fig.clear()
                # Copy frame content
                source_ax = frames[frame_num].axes[0]
                new_ax = fig.add_subplot(111, projection='3d')

                # Copy visualization data (simplified)
                for collection in source_ax.collections:
                    if hasattr(collection, '_offsets3d'):
                        xs, ys, zs = collection._offsets3d
                        colors = collection.get_facecolors()
                        sizes = collection.get_sizes()
                        new_ax.scatter(xs, ys, zs, c=colors, s=sizes,
                                       alpha=collection.get_alpha() or 0.7)

                # Set view angle for rotation
                angle = frame_num * 10
                new_ax.view_init(elev=20, azim=angle)
                new_ax.set_title(f'3D Pore Network - Frame {frame_num + 1}')

                return new_ax.collections

            anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                           interval=200, blit=False)

            writer = PillowWriter(fps=5)
            anim.save(file_path, writer=writer)

        except ImportError:
            # Fallback: save individual frames
            import os
            base_name = os.path.splitext(file_path)[0]
            for i, frame in enumerate(frames):
                frame.savefig(f"{base_name}_frame_{i:03d}.png", dpi=150)
            QMessageBox.information(
                self, "Info", f"Saved {len(frames)} individual frames (GIF export requires pillow)")

    def _save_mp4_animation(self, frames, file_path):
        """Save frames as MP4 animation."""
        try:
            import matplotlib.animation as animation
            from matplotlib.animation import FFMpegWriter

            # Similar to GIF but with FFMpeg writer
            fig = frames[0]

            def animate(frame_num):
                fig.clear()
                source_ax = frames[frame_num].axes[0]
                new_ax = fig.add_subplot(111, projection='3d')

                for collection in source_ax.collections:
                    if hasattr(collection, '_offsets3d'):
                        xs, ys, zs = collection._offsets3d
                        colors = collection.get_facecolors()
                        sizes = collection.get_sizes()
                        new_ax.scatter(xs, ys, zs, c=colors, s=sizes,
                                       alpha=collection.get_alpha() or 0.7)

                angle = frame_num * 10
                new_ax.view_init(elev=20, azim=angle)
                new_ax.set_title(f'3D Pore Network - Frame {frame_num + 1}')

                return new_ax.collections

            anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                           interval=100, blit=False)

            writer = FFMpegWriter(fps=10, metadata=dict(
                artist='Pore Network Visualizer'))
            anim.save(file_path, writer=writer)

        except ImportError:
            # Fallback to GIF
            gif_path = file_path.replace('.mp4', '.gif')
            self._save_gif_animation(frames, gif_path)
            QMessageBox.information(
                self, "Info", f"Saved as GIF instead (MP4 export requires ffmpeg): {gif_path}")

    def create_quality_comparison_view(self):
        """Create a comparison view showing different quality levels."""
        if not self.current_data_path:
            QMessageBox.warning(
                self, "Warning", "Please load a data file first.")
            return

        try:
            self.status_bar.showMessage("Creating quality comparison...")
            self.progress_bar.setVisible(True)

            # Create comparison figure
            fig = self.visualization_engine.create_comparison_view(
                self.current_data_path)
            self.on_visualization_finished(fig)

            self.status_bar.showMessage("Quality comparison created")

        except Exception as e:
            self.on_visualization_error(
                f"Failed to create comparison: {str(e)}")

    def toggle_measurement_mode(self, enabled):
        """Toggle measurement overlay mode."""
        self.visualization_engine.update_parameters(
            {'measurement_tools': enabled})
        if self.auto_refresh_enabled and self.current_data_path:
            self.generate_visualization()

    def apply_scientific_preset(self):
        """Apply scientific publication preset."""
        preset_params = {
            'color_scheme': 'scientific',
            'sphere_style': 'matte',
            'bond_style': 'lines',
            'lighting_model': 'enhanced',
            'depth_cueing': True,
            'edge_enhancement': False,
            'size_variance': 0.1,
            'background_color': 'white',
            'scientific_annotations': True,
            'measurement_tools': True
        }

        self.visualization_engine.update_parameters(preset_params)

        # Update GUI controls to match preset
        if hasattr(self.parameter_widget, 'color_scheme'):
            self.parameter_widget.color_scheme.setCurrentText('scientific')
            self.parameter_widget.sphere_style.setCurrentText('matte')
            self.parameter_widget.bond_style.setCurrentText('lines')
            self.parameter_widget.lighting_model.setCurrentText('enhanced')
            self.parameter_widget.depth_cueing.setChecked(True)
            self.parameter_widget.edge_enhancement.setChecked(False)
            self.parameter_widget.size_variance.setValue(0.1)

    def apply_presentation_preset(self):
        """Apply presentation-ready preset."""
        preset_params = {
            'color_scheme': 'artistic',
            'sphere_style': 'glossy',
            'bond_style': 'tubes',
            'lighting_model': 'dramatic',
            'depth_cueing': True,
            'edge_enhancement': True,
            'size_variance': 0.2,
            'background_color': '#1a1a1a',
            'scientific_annotations': False,
            'measurement_tools': False,
            'num_spheres': 800  # Optimized for clarity
        }

        self.visualization_engine.update_parameters(preset_params)

        # Update GUI controls
        if hasattr(self.parameter_widget, 'color_scheme'):
            self.parameter_widget.color_scheme.setCurrentText('artistic')
            self.parameter_widget.sphere_style.setCurrentText('glossy')
            self.parameter_widget.bond_style.setCurrentText('tubes')
            self.parameter_widget.lighting_model.setCurrentText('dramatic')
            self.parameter_widget.depth_cueing.setChecked(True)
            self.parameter_widget.edge_enhancement.setChecked(True)
            self.parameter_widget.size_variance.setValue(0.2)
            self.parameter_widget.num_spheres.setValue(800)

    def apply_preset_visualization_mode(self, mode):
        """Apply a preset visualization mode with optimized parameters."""
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

        # Create advanced visualization based on mode
        try:
            self.status_bar.showMessage(
                f"Generating {mode.replace('_', ' ').title()} visualization...")

            # Use the advanced visualization engine
            if hasattr(self.visualization_engine, 'create_advanced_visualization'):
                fig = self.visualization_engine.create_advanced_visualization(
                    self.current_data_path, mode)
                self.on_visualization_finished(fig)
            else:
                # Fallback to regular visualization
                self.visualization_thread = VisualizationThread(
                    self.visualization_engine, self.current_data_path, mode
                )
                self.visualization_thread.finished.connect(
                    self.on_visualization_finished)
                self.visualization_thread.error.connect(
                    self.on_visualization_error)
                self.visualization_thread.progress.connect(
                    self.progress_bar.setValue)
                self.visualization_thread.start()

        except Exception as e:
            self.on_visualization_error(
                f"Failed to create {mode} visualization: {str(e)}")

    def closeEvent(self, event):
        """Handle application closing."""
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
        """Restore application settings with better window positioning."""
        try:
            # Only restore geometry if it's reasonable
            geometry = self.settings.value('geometry')
            if geometry:
                self.restoreGeometry(geometry)
                # Ensure window is visible on screen
                screen = QApplication.desktop().screenGeometry()
                if (self.x() < 0 or self.y() < 0 or
                        self.x() > screen.width() or self.y() > screen.height()):
                    # Reset to center if position is invalid
                    self.move(screen.width()//4, screen.height()//4)

            state = self.settings.value('windowState')
            if state:
                self.restoreState(state)

            self.logger.info("Application settings restored")
        except Exception as e:
            self.logger.warning(f"Failed to restore settings: {e}")

    def setup_realtime_connections(self):
        """Set up connections between parameter controls and real-time rendering."""
        if hasattr(self, 'realtime_widget') and hasattr(self, 'parameter_widget'):
            # Connect parameter changes to real-time updates
            self.parameter_widget.parameter_changed.connect(
                self.realtime_widget.update_parameters)
            self.logger.info("Real-time parameter connections established")

    def toggle_auto_refresh(self, enabled):
        """Toggle auto-refresh of main visualization on parameter changes."""
        self.auto_refresh_enabled = enabled
        self.logger.info(
            f"Auto-refresh {'enabled' if enabled else 'disabled'}")

    def apply_preset_visualization_mode(self, mode):
        """Apply a preset visualization mode with optimized parameters."""
        if not self.current_data_path:
            QMessageBox.warning(
                self, "Warning", "Please load a data file first.")
            return

        try:
            self.status_bar.showMessage(
                f"Generating {mode.replace('_', ' ').title()} visualization...")

            # Use the advanced visualization engine
            if hasattr(self.visualization_engine, 'create_advanced_visualization'):
                # Clear previous plot
                self.figure.clear()
                self.canvas.draw()

                # Show progress bar
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(25)

                fig = self.visualization_engine.create_advanced_visualization(
                    self.current_data_path, mode)
                self.on_visualization_finished(fig)
            else:
                # Fallback to regular visualization - use thread
                self.generate_visualization_with_type(mode)

        except Exception as e:
            QMessageBox.critical(self, "Visualization Error",
                                 f"Failed to create {mode} visualization:\n{str(e)}")
            self.progress_bar.setVisible(False)

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

    def generate_visualization(self):
        """Generate 3D visualization."""
        viz_type = self.viz_type.currentText()
        self.generate_visualization_with_type(viz_type)

    def on_visualization_finished(self, figure):
        """Handle completed visualization."""
        try:
            # Clear the current figure
            self.figure.clear()

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
                            new_ax.scatter(xs, ys, zs, c=colors,
                                           s=sizes, alpha=alpha)

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
            QMessageBox.critical(self, "Visualization Error",
                                 f"Error displaying visualization:\n{str(e)}")
            self.progress_bar.setVisible(False)
            self.generate_btn.setEnabled(True)

    def on_visualization_error(self, error_msg):
        """Handle visualization error."""
        QMessageBox.critical(self, "Visualization Error",
                             f"Failed to generate visualization:\n{error_msg}")
        self.progress_bar.setVisible(False)
        self.generate_btn.setEnabled(True)
        self.status_bar.showMessage("Visualization failed")

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
                self.generate_btn.setEnabled(True)
                self.status_bar.showMessage(f"Data loaded: {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Load Error",
                                     f"Failed to load data:\n{str(e)}")

    def update_visualization_parameter(self, param_name: str, value):
        """Update visualization parameter and trigger real-time rendering."""
        self.visualization_engine.update_parameters({param_name: value})
        if self.auto_refresh_enabled and self.current_data_path:
            self.generate_visualization()

    def export_visualization(self):
        """Export current visualization."""
        if not self.current_figure:
            QMessageBox.warning(self, "Warning", "No visualization to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Visualization", "",
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

            # Display results
            self.analysis_widget.display_analysis(report)
            self.status_bar.showMessage("Analysis completed successfully")

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error",
                                 f"Failed to run analysis:\n{str(e)}")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Pore Network Visualizer",
            "Pore Network Visualizer v1.0\n\n"
            "Professional 3D visualization tool for pore structure analysis\n"
            "with enhanced rendering and advanced features.\n\n"
            "Features:\n"
            "• Interactive 3D visualization\n"
            "• Advanced pore analysis\n"
            "• Multiple visualization modes\n"
            "• Publication-quality export\n"
            "• Real-time parameter tuning"
        )

    def closeEvent(self, event):
        """Handle application closing."""
        self.save_settings()
        if self.visualization_thread and self.visualization_thread.isRunning():
            self.visualization_thread.terminate()
            self.visualization_thread.wait()
        event.accept()

    def save_settings(self):
        """Save application settings."""
        self.settings.setValue('geometry', self.saveGeometry())
        self.settings.setValue('windowState', self.saveState())


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Pore Network Visualizer")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Scientific Visualization Lab")

    # Set application icon if available
    try:
        app.setWindowIcon(QIcon("assets/icon.png"))
    except:
        pass

    # Create and show main window
    window = PoreVisualizerGUI()
    window.show()

    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
